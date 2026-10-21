"""Persistent claim controller: reviewed assessment -> validated adjudication."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import TypedDict

# Keep this workflow and its checkpoints local.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import adjudicate_batch
import retrieval_controller
import prepare_assessment_draft as draft_preparer
from adjudicate_batch import decide, sha
from verify_claim_agent import dump, exclusive_run


class ClaimState(TypedDict):
    claim_id: str
    claim_text: str
    claim_spec: dict
    retrieval: dict | None
    screening: dict | None
    verification: dict | None
    retrieval_allowed: bool
    input_hash: str
    status: str
    pending_action: str
    source_review_status: str
    assessment: dict | None
    submissions: int
    max_submissions: int
    result: dict | None
    last_error: str | None
    history: list[dict]


def digest_json(value):
    return sha(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    )


def artifact_ref(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": sha(path.read_bytes()),
    }


def append_event(state, action, **details):
    return state["history"] + [
        {"action": action, **details}
    ]


def plan(state, retrieval_settings=None, verification_settings=None):
    # Readiness routing; scientific planning is a separate component.
    if state["submissions"] >= state["max_submissions"]:
        return {
            "status": "blocked_submission_limit",
            "result": None,
        }

    if verification_settings and not state["assessment"]:
        if state.get("screening") and state.get("verification") is None:
            return {"status": "verifying_scientific_evidence"}
        if not state.get("screening"):
            outcome = state["claim_spec"].get("outcome", "")
            if not isinstance(outcome, str) or not outcome.strip() or retrieval_controller._generic_outcome(outcome):
                return {"status": "claim_clarification_required", "pending_action": "clarify_claim_outcome"}

    if (retrieval_settings and state.get("retrieval_allowed")
            and state.get("retrieval") is None and not state["assessment"]):
        return {"status": "retrieving_evidence"}

    return {
        "status": (
            "validating_assessment"
            if state["assessment"]
            else (state.get("verification") or {}).get("status", "awaiting_reviewed_assessment")
        )
    }


def route(state):
    if state["status"] == "blocked_submission_limit":
        return END

    if state["status"] == "claim_clarification_required":
        return END
    if state["status"] == "verifying_scientific_evidence":
        return "verify"
    if state["status"] == "retrieving_evidence":
        return "retrieve"

    return (
        "adjudicate"
        if state["assessment"]
        else "await_assessment"
    )


def await_assessment(state):
    payload = interrupt(
        {
            "claim_id": state["claim_id"],
            "claim_text": state["claim_text"],
            "pending_action": state["pending_action"],
            "last_error": state["last_error"],
            "retrieval": state.get("retrieval"),
            "required": (
                "A completed, reviewed claim-level assessment "
                "in the existing assisted or "
                "adjudication-ready-v1 format."
            ),
        }
    )

    if (
        not isinstance(payload, dict)
        or set(payload) != {"path", "sha256"}
        or not all(
            isinstance(value, str) and value
            for value in payload.values()
        )
    ):
        return {
            "assessment": None,
            "submissions": state["submissions"] + 1,
            "last_error": "Invalid assessment reference",
            "result": None,
        }

    return {
        "assessment": payload,
        "last_error": None,
        "history": append_event(
            state,
            "assessment_attached",
            sha256=payload["sha256"],
        ),
    }


def adjudicate_node(state):
    attempt = state["submissions"] + 1

    try:
        ref = state["assessment"]

        if ref is None:
            raise ValueError("Missing assessment reference")

        raw = Path(ref["path"]).read_bytes()

        if sha(raw) != ref["sha256"]:
            raise ValueError(
                "Assessment changed after attachment"
            )

        data = json.loads(raw)

        if (
            not isinstance(data, dict)
            or not isinstance(data.get("claim"), dict)
        ):
            raise ValueError(
                "Assessment has no claim identity"
            )

        claim = data["claim"]

        supplied_text = " ".join(
            str(claim.get("claim_text", "")).split()
        )
        expected_text = " ".join(
            state["claim_text"].split()
        )

        if (
            claim.get("claim_id") != state["claim_id"]
            or supplied_text != expected_text
        ):
            raise ValueError(
                "Assessment belongs to a different "
                "claim or wording"
            )

        result = decide(data, ref["sha256"])

        if (
            result["status"] == "blocked"
            or result.get("verdict")
            not in {
                "supported",
                "refuted",
                "misleading",
                "insufficient_evidence",
            }
        ):
            raise ValueError(
                result.get(
                    "reason",
                    "Assessment is not ready for adjudication",
                )
            )

        result.update(
            claim_id=state["claim_id"],
            claim_text=state["claim_text"],
            input_sha256=ref["sha256"],
            input_path=ref["path"],
        )

        return {
            "status": "completed_from_reviewed_assessment",
            "result": result,
            "submissions": attempt,
            "last_error": None,
            "pending_action": "finished",
            "history": append_event(
                state,
                "adjudicated",
                input_sha256=ref["sha256"],
                verdict=result["verdict"],
            ),
        }

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as error:
        return {
            "status": "assessment_rejected",
            "assessment": None,
            "result": None,
            "submissions": attempt,
            "last_error": str(error),
            "pending_action": (
                "repair_or_complete_scientific_assessment"
            ),
            "history": append_event(
                state,
                "assessment_rejected",
                reason=str(error),
            ),
        }


def after_adjudication(state):
    return (
        END
        if state["result"] is not None
        else "plan"
    )


def retrieval_node(state, settings):
    print("Retrieving:", state["claim_id"], flush=True)

    try:
        result = retrieval_controller.retrieve(
            state["claim_spec"],
            Path(settings["root"]) / sha(
                state["claim_id"].encode()
            ),
            settings["model"],
            os.environ.get("NCBI_EMAIL"),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise retrieval_controller.RetrievalPaused(
            str(error), "retrieval_input_error"
        ) from error

    action = {
        "claim_clarification_required": "clarify_claim_outcome",
        "retrieval_no_usable_abstracts": (
            "review_search_coverage_and_source_availability"
        ),
    }.get(
        result["status"],
        "rank_candidates_and_complete_suitability_verification_and_review",
    )

    return {
        "retrieval": result,
        "status": result["status"],
        "pending_action": action,
        "last_error": None,
        "history": append_event(
            state, "retrieval_finished", result=result
        ),
    }


def make_graph(checkpointer, retrieval_settings=None, verification_settings=None):
    graph = StateGraph(ClaimState)

    graph.add_node("plan", lambda state: plan(state, retrieval_settings, verification_settings))
    if verification_settings:
        graph.add_node("verify", lambda state: verification_node(state, verification_settings))
        graph.add_edge("verify", "await_assessment")
    graph.add_node(
        "retrieve",
        lambda state: retrieval_node(state, retrieval_settings),
    )
    graph.add_node(
        "await_assessment",
        await_assessment,
    )
    graph.add_node(
        "adjudicate",
        adjudicate_node,
    )

    graph.add_edge(START, "plan")
    graph.add_edge("retrieve", "await_assessment")
    graph.add_conditional_edges("plan", route)
    graph.add_edge("await_assessment", "plan")
    graph.add_conditional_edges(
        "adjudicate",
        after_adjudication,
    )

    return graph.compile(
        checkpointer=checkpointer
    )


def seeds(worklist_path, source_review_path, screening=None):
    worklist = json.loads(
        Path(worklist_path).read_bytes()
    )

    reviews = {}

    if (
        source_review_path
        and Path(source_review_path).exists()
    ):
        report = json.loads(
            Path(source_review_path).read_bytes()
        )

        for row in report["claims"]:
            cid = row["claim"]["claim_id"]

            if cid in reviews:
                raise ValueError(
                    "Duplicate source-review claim: " + cid
                )

            reviews[cid] = row

    output = []
    seen = set()

    for row in worklist["claims"]:
        claim = row["claim"]
        cid = claim["claim_id"]
        text = claim["claim_text"]

        if (
            not isinstance(cid, str)
            or not cid.strip()
            or cid in seen
            or row["claim_id"] != cid
        ):
            raise ValueError(
                "Missing or duplicate claim identity"
            )

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Missing claim wording")

        seen.add(cid)
        review = reviews.get(cid)

        if review:
            if review["claim"]["claim_text"] != text:
                raise ValueError(
                    "Source review wording differs "
                    "from extracted claim"
                )

            for card in review["source_cards"].values():
                if (
                    "text" in card
                    and sha(card["text"].encode())
                    != card["metadata"]["text_sha256"]
                ):
                    raise ValueError(
                        "Source snapshot was modified"
                    )

        candidates = row.get("candidate_reports", [])

        ref = (
            artifact_ref(candidates[0])
            if len(candidates) == 1
            else None
        )

        action = (
            "select_one_reviewed_assessment"
            if len(candidates) > 1
            else (
                "complete_relevance_screening"
                "_verification_and_review"
            )
        )

        source_status = (
            review["status"]
            if review
            else "not_supplied"
        )

        output.append(
            {
                "claim_id": cid,
                "claim_text": text,
                "claim_spec": claim,
                "retrieval": None,
                "screening": screening,
                "verification": None,
                "retrieval_allowed": (
                    row.get("status") == "verification_pending"
                    and not candidates
                ),
                "input_hash": digest_json(
                    {
                        "claim": claim,
                        "screening": screening,
                        "source_review": review,
                        "initial_assessment": ref,
                        "candidates": candidates,
                    }
                ),
                "status": "new",
                "pending_action": action,
                "source_review_status": source_status,
                "assessment": ref,
                "submissions": 0,
                "max_submissions": 2,
                "result": None,
                "last_error": None,
                "history": [],
            }
        )

    if not output:
        raise ValueError("No claims supplied")

    return output


def export_state(folder, state, interrupted):
    result = state["result"]

    row = {
        "claim_id": state["claim_id"],
        "claim_text": state["claim_text"],
        "status": state["status"],
        "interrupted": interrupted,
        "pending_action": state["pending_action"],
        "source_review_status": state[
            "source_review_status"
        ],
        "retrieval": state.get("retrieval"),
        "verification": state.get("verification"),
        "verdict": (
            result["verdict"]
            if result
            else None
        ),
        "last_error": state["last_error"],
        "submissions": state["submissions"],
        "history": state["history"],
        "adjudication": result,
    }

    filename = (
        sha(state["claim_id"].encode())[:16]
        + ".json"
    )

    dump(folder / "claims" / filename, row)
    return row


def run(args):
    snapshot_path = getattr(args, "screening_review", None)
    screening = artifact_ref(snapshot_path) if snapshot_path else None
    initial = seeds(args.worklist, args.source_review, screening)
    if screening:
        snapshot_rows, _, _ = draft_preparer.snapshot_claims(snapshot_path)
        known = {s["claim_id"] for s in initial}
        if not set(snapshot_rows) <= known:
            raise ValueError("Snapshot contains claims outside the canonical worklist")
        for state in initial:
            row = snapshot_rows.get(state["claim_id"])
            if row and row["claim"] != state["claim_spec"]:
                raise ValueError("Snapshot claim specification differs from the worklist")
            if not row:
                state["screening"] = None

    folder = Path(args.run_dir).resolve()
    retrieval_settings = (
        {
            "root": str(folder / "retrieval"),
            "model": getattr(
                args, "retrieval_model", "qwen3:4b"
            ),
            "controller_sha256": sha(
                Path(retrieval_controller.__file__).read_bytes()
            ),
        }
        if getattr(args, "retrieve", False)
        else None
    )

    verification_settings = None
    if screening:
        from screen_relevance_batch import model_digest
        import local_quote_backend
        import verification_schemas
        verification_settings = {
            "root": str(folder / "verification"), "snapshot": screening,
            "model": args.verification_model, "model_digest": model_digest(args.verification_model),
            "source_limit": args.max_verify_sources,
            "helper_sha256": sha(Path(draft_preparer.__file__).read_bytes()),
            "anchors_sha256": sha(Path(local_quote_backend.__file__).read_bytes()),
            "schema_sha256": sha(Path(verification_schemas.__file__).read_bytes()),
            "remaining": [args.max_new_calls],
        }
    binding = {
        "verification_settings": ({k: v for k, v in verification_settings.items() if k != "remaining"} if verification_settings else None),
        "retrieval_settings": retrieval_settings,
        "profile": "claim-controller-v1",
        "script_sha256": sha(
            Path(__file__).read_bytes()
        ),
        "adjudicator_sha256": sha(
            Path(adjudicate_batch.__file__).read_bytes()
        ),
        "framework_versions": {
            name: version(name)
            for name in (
                "langgraph",
                "langgraph-checkpoint-sqlite",
                "langgraph-checkpoint",
                "pydantic",
            )
        },
        "claim_inputs": {
            state["claim_id"]: state["input_hash"]
            for state in initial
        },
    }

    with exclusive_run(folder):
        binding_path = folder / "run-binding.json"

        if (
            binding_path.exists()
            and json.loads(binding_path.read_bytes())
            != binding
        ):
            raise ValueError(
                "Inputs/code changed; "
                "use a new run directory"
            )

        if (
            args.resume
            and args.resume not in binding["claim_inputs"]
        ):
            raise ValueError("Unknown claim ID")

        resume_ref = (
            artifact_ref(args.assessment)
            if args.assessment
            else None
        )

        dump(binding_path, binding)

        checkpoint_path = str(
            folder / "checkpoints.sqlite"
        )

        with SqliteSaver.from_conn_string(
            checkpoint_path
        ) as saver:
            graph = make_graph(saver, retrieval_settings, verification_settings)

            if args.resume:
                config = {
                    "configurable": {
                        "thread_id": args.resume
                    },
                    "recursion_limit": 30,
                }

                snapshot = graph.get_state(config)

                if (
                    not snapshot.values
                    or not any(
                        task.interrupts
                        for task in snapshot.tasks
                    )
                ):
                    raise ValueError(
                        "Claim is not waiting "
                        "for an assessment"
                    )

                graph.invoke(
                    Command(resume=resume_ref),
                    config,
                )

            rows = []

            for state in initial:
                config = {
                    "configurable": {
                        "thread_id": state["claim_id"]
                    },
                    "recursion_limit": 30,
                }

                snapshot = graph.get_state(config)

                if (verification_settings and not args.resume and snapshot.values
                        and (snapshot.values.get("verification") or {}).get("retryable")):
                    graph.update_state(config, {"verification": None}, as_node="await_assessment")
                    snapshot = graph.get_state(config)
                paused = None
                try:
                    if not snapshot.values:
                        graph.invoke(state, config)
                    elif (
                        snapshot.next
                        and not any(
                            task.interrupts
                            for task in snapshot.tasks
                        )
                    ):
                        graph.invoke(None, config)
                except retrieval_controller.RetrievalPaused as error:
                    paused = error

                snapshot = graph.get_state(config)
                saved = dict(snapshot.values)

                if paused is not None:
                    saved.update(
                        status=paused.status,
                        last_error=str(paused),
                        pending_action=(
                            "inspect_retrieval_state_then_rerun_same_command"
                        ),
                    )

                retrieval_ref = saved.get("retrieval") or {}

                if (
                    retrieval_ref.get("evidence_path")
                    and saved["result"] is None
                ):
                    try:
                        current_hash = sha(
                            Path(
                                retrieval_ref["evidence_path"]
                            ).read_bytes()
                        )
                        if (
                            current_hash
                            != retrieval_ref["evidence_sha256"]
                        ):
                            raise ValueError(
                                "Saved retrieval evidence was modified"
                            )
                    except (OSError, ValueError) as error:
                        saved.update(
                            status="retrieval_artifact_invalid",
                            last_error=str(error),
                            pending_action=(
                                "restore_original_retrieval_artifact"
                            ),
                        )

                if (
                    saved["assessment"] is not None
                    and saved["result"] is not None
                ):
                    current_ref = artifact_ref(
                        saved["assessment"]["path"]
                    )

                    if current_ref != saved["assessment"]:
                        raise ValueError(
                            "Completed assessment file "
                            "changed; preserve this run "
                            "and start a new one"
                        )

                row = export_state(
                    folder,
                    saved,
                    any(
                        task.interrupts
                        for task in snapshot.tasks
                    ),
                )

                rows.append(row)

                print(
                    row["claim_id"],
                    "|",
                    row["status"],
                    "|",
                    row["verdict"] or "-",
                )

                if not row["verdict"]:
                    print(
                        "  NEXT:",
                        row["pending_action"],
                    )

                    if row["last_error"]:
                        print(
                            "  REASON:",
                            row["last_error"],
                        )

            dump(
                folder / "workflow-summary.json",
                {
                    "binding": binding,
                    "claims": rows,
                    "scope": (
                        "Persistent orchestration and "
                        "validated adjudication of supplied "
                        "reviewed assessments. Missing "
                        "scientific stages pause. "
                        "Local query planning and PubMed retrieval run "
                        "only with --retrieve; candidates require scientific review."
                    ),
                },
            )

            if verification_settings:
                export_verification_review(folder, rows)
            print(
                "SUMMARY:",
                folder / "workflow-summary.json",
            )



# Bounded snapshot verification v1: drafts always require recorded review.
def verification_node(state, settings):
    ref = state["screening"]
    try:
        if artifact_ref(ref["path"]) != ref:
            raise ValueError("Screening snapshot changed")
        result = draft_preparer.bounded_snapshot_draft(
            ref["path"], state["claim_spec"],
            Path(settings["root"]) / sha(state["claim_id"].encode())[:16], settings,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"status": "verification_input_blocked", "error": str(error), "retryable": False}
    return {"verification": result, "status": result["status"],
            "pending_action": ("critical_review_and_submit_reviewed_assessment" if result.get("draft_path")
                               else "resume_bounded_verification" if result.get("retryable") else "inspect_verification_blocker"),
            "last_error": result.get("error"),
            "history": append_event(state, "bounded_verification", result=result)}


def export_verification_review(folder, rows):
    items = []
    for row in rows:
        result = row.get("verification") or {}
        if result.get("draft_path"):
            path = Path(result["draft_path"])
            items.append({"draft_path": str(path), "draft": json.loads(path.read_bytes()),
                          "provenance": json.loads(path.with_name("provenance.json").read_bytes()),
                          "source_selection": json.loads(path.with_name("source-selection.json").read_bytes())})
    dump(folder / "verification-review-bundle.json", {
        "schema_version": "bounded-verification-review-v1", "assessments": items,
        "scope": "Unapproved model drafts for critical review; no new verdicts or scientific approvals.",
    })
    print("REVIEW BUNDLE:", folder / "verification-review-bundle.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--worklist",
        default=(
            "data/processed/"
            "project-claim-status.json"
        ),
    )

    parser.add_argument(
        "--source-review",
        default=(
            "data/processed/"
            "source-review-batch-v1/"
            "source-review.json"
        ),
    )

    parser.add_argument(
        "--run-dir",
        default=(
            "data/processed/"
            "langgraph-controller-v1"
        ),
    )

    parser.add_argument(
        "--resume",
        help="Claim ID waiting at a checkpoint",
    )

    parser.add_argument(
        "--assessment",
        help="Reviewed assessment JSON for --resume",
    )

    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="Plan searches with Ollama and retrieve PubMed candidates",
    )
    parser.add_argument(
        "--retrieval-model", default="qwen3:4b"
    )

    parser.add_argument("--screening-review", help="Saved screening-review checkpoint export")
    parser.add_argument("--verification-model", default="qwen3:4b")
    parser.add_argument("--max-verify-sources", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-new-calls", type=int, default=3)
    args = parser.parse_args()
    if args.max_new_calls < 1:
        parser.error("max-new-calls must be positive")
    if args.screening_review and args.retrieve:
        parser.error("Use cached verification separately from new retrieval")

    if bool(args.resume) != bool(args.assessment):
        parser.error(
            "Supply --resume and --assessment together"
        )

    run(args)