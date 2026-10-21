"""Build a project-wide worklist from claims and reviewed reports."""
import argparse
from collections import Counter
import json
from pathlib import Path

from adjudicate_batch import decide, sha
from verify_claim_agent import dump


def normalized(text):
    return " ".join(str(text or "").split())


# Consolidation change 3: discover and classify assessment artifacts.
def build(claims_dir, experiments, summary_path=None):
    claims = {}

    for path in sorted(Path(claims_dir).rglob("*.json")):
        document = json.loads(path.read_bytes())
        rows = (
            document.get("claims")
            if isinstance(document, dict)
            else document
        )

        if not isinstance(rows, list):
            raise ValueError(f"Expected a claims list: {path}")

        for claim in rows:
            if (
                not isinstance(claim, dict)
                or not claim.get("claim_id")
                or not claim.get("claim_text")
            ):
                raise ValueError(f"Missing claim ID or text: {path}")

            cid = claim["claim_id"]

            if not isinstance(cid, str):
                raise ValueError(f"Claim ID must be a string: {path}")

            cid = cid.strip()

            if not cid or not normalized(claim["claim_text"]):
                raise ValueError(f"Empty claim ID or text: {path}")

            if cid in claims and claims[cid]["claim"] != claim:
                raise ValueError(
                    f"Conflicting extracted records for {cid}"
                )

            claims[cid] = {
                "claim": claim,
                "claims_file": str(path.resolve()),
            }

    if not claims:
        raise ValueError("No extracted claims found")

    artifacts = Path(experiments)
    candidates = {cid: [] for cid in claims}
    awaiting_review = {cid: [] for cid in claims}
    invalid = {cid: [] for cid in claims}
    issues = []
    saved_results = []

    summary_paths = {
        p.resolve()
        for name in ("batch-summary.json", "workflow-summary.json")
        for p in artifacts.rglob(name)
    }

    if summary_path and Path(summary_path).exists():
        summary_paths.add(Path(summary_path).resolve())

    for summary_file in sorted(summary_paths):
        try:
            document = json.loads(summary_file.read_bytes())
            if not isinstance(document, dict):
                raise ValueError("Expected a summary object")

            if isinstance(document.get("results"), list):
                rows = document["results"]
            elif isinstance(document.get("claims"), list):
                rows = [
                    row.get("adjudication")
                    for row in document["claims"]
                    if isinstance(row, dict)
                    and row.get("adjudication") is not None
                ]
            else:
                raise ValueError(
                    "Expected batch results or workflow claims"
                )

            if any(not isinstance(row, dict) for row in rows):
                raise ValueError("Invalid adjudication result")

            saved_results.extend(rows)

        except (OSError, ValueError, TypeError) as exc:
            issues.append({
                "path": str(summary_file),
                "problem": str(exc),
            })

    paths = sorted({
        p.resolve()
        for name in (
            "assessment-reviewed.json",
            "assessment-draft.json",
            "verification-assisted.json",
            "verification-ready.json",
        )
        for p in artifacts.rglob(name)
    })

    # Discover recorded assessment paths v2.
    discovered = set(paths)
    for saved in saved_results:
        name = saved.get("input_path")
        expected = saved.get("input_sha256")
        if saved.get("status") == "blocked" or not isinstance(name, str) or not name.strip():
            continue
        reference = Path(name).resolve()
        try:
            if not reference.is_relative_to(artifacts.resolve()):
                raise ValueError("Recorded assessment is outside the artifact root")
            if not isinstance(expected, str) or sha(reference.read_bytes()) != expected:
                raise ValueError("Recorded assessment hash differs from its adjudication")
        except (OSError, ValueError) as exc:
            issues.append({"path": str(reference), "problem": str(exc)})
            continue
        discovered.add(reference)
    paths = sorted(discovered)

    for path in paths:
        cid = None
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
            claim = data["claim"]
            cid = claim["claim_id"].strip()

            if cid not in claims:
                issues.append({
                    "path": str(path),
                    "problem": (
                        "Claim ID is outside the extracted-claims inventory"
                    ),
                })
                continue

            if normalized(claim["claim_text"]) != normalized(
                claims[cid]["claim"]["claim_text"]
            ):
                raise ValueError(
                    "Report claim wording differs from extracted claim"
                )

            decision = decide(data, sha(raw))

            if decision["status"] == "blocked":
                if data.get("schema_version") != "adjudication-ready-v1":
                    raise ValueError(decision["reason"])

                awaiting_review[cid].append({
                    "path": str(path),
                    "sha256": sha(raw),
                    "reason": decision["reason"],
                    "next_action": decision["next_action"],
                })
                continue

            digest = sha(raw)
            duplicate = next(
                (
                    item for item in candidates[cid]
                    if item["sha256"] == digest
                ),
                None,
            )

            if duplicate is not None:
                duplicate["copies"].append(str(path))
            else:
                candidates[cid].append({
                    "path": str(path),
                    "sha256": digest,
                    "copies": [str(path)],
                    "decision": decision,
                })

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as exc:
            issue = {
                "path": str(path),
                "problem": str(exc),
            }
            issues.append(issue)
            if cid in invalid:
                invalid[cid].append(issue)

    rows = []
    pending = []

    for cid, entry in sorted(claims.items()):
        reports = candidates[cid]
        status = "verification_pending"
        verdict = None

        if not reports and awaiting_review[cid]:
            status = "assessment_review_pending"
        elif not reports and invalid[cid]:
            status = "artifact_repair_required"

        if len(reports) > 1:
            status = "select_verification_run"

        elif reports:
            report = reports[0]
            status = "ready_for_adjudication"

            for saved in saved_results:
                if (
                    saved.get("claim_id") == cid
                    and saved.get("status") != "blocked"
                    and saved.get("input_sha256") == report["sha256"]
                    and saved.get("verdict")
                    == report["decision"]["verdict"]
                    and saved.get("status")
                    == report["decision"]["status"]
                ):
                    status = "adjudication_recorded"
                    verdict = saved["verdict"]
                    break

            if status == "ready_for_adjudication":
                pending.append(report["path"])

        rows.append({
            **entry,
            "claim_id": cid,
            "status": status,
            "verdict": verdict,
            "candidate_reports": [
                report["path"] for report in reports
            ],
            "candidate_report_copies": [
                {
                    "sha256": item["sha256"],
                    "paths": item["copies"],
                }
                for item in reports
            ],
            "assessments_awaiting_review": awaiting_review[cid],
            "invalid_assessments": invalid[cid],
            "claim_specification_note": (
                "Structured outcome is missing; review the wording "
                "before operationalizing it."
                if not entry["claim"].get("outcome")
                else None
            ),
        })

    return {
        "total_claims": len(rows),
        "status_counts": dict(
            Counter(row["status"] for row in rows)
        ),
        "claims": rows,
        "artifact_issues": issues,
        "adjudication_summaries": [
            str(p) for p in sorted(summary_paths)
        ],
        "pending_adjudication_inputs": pending,
        "scope": (
            "Artifact readiness and recorded adjudication; "
            "not independent scientific validation. "
            "No retrieval or model calls performed."
        ),
    }



# Video report export v1. Clinical verdicts come only from validated assessments.
def export_video_report(inventory, args):
    from verify_claim_agent import atomic_write, exclusive_run
    transcript_path = Path(args.transcript).resolve()
    transcript_raw = transcript_path.read_bytes()
    transcript = json.loads(transcript_raw)
    video_id = transcript["video_id"]
    selected = [r for r in inventory["claims"] if r["claim_id"].startswith(video_id + "-")
                or Path(r["claims_file"]).stem == video_id]
    if not selected:
        raise ValueError("No extracted claims match this transcript video_id")
    summary_path = Path(args.summary).resolve()
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    workflow_rows = {r["claim_id"]: r for r in summary["claims"]}
    scope_path = Path(args.scope_review).resolve() if args.scope_review else None
    scope_raw = scope_path.read_bytes() if scope_path else None
    scope = json.loads(scope_raw) if scope_raw else {"claims": []}
    if scope_path and (scope.get("profile") != "claim-scope-review-v1"
            or scope.get("video_id") != video_id
            or scope.get("transcript_sha256") != sha(transcript_raw)
            or scope.get("method") not in ("assisted", "human", "model")
            or not isinstance(scope.get("reviewer"), str) or not scope["reviewer"].strip()
            or not isinstance(scope.get("scope"), str) or not scope["scope"].strip()
            or type(scope.get("independently_reviewed")) is not bool
            or type(scope.get("expert_validated")) is not bool):
        raise ValueError("Scope review belongs to another transcript or has no declared method")
    if scope.get("method") != "human" and (scope.get("independently_reviewed") or scope.get("expert_validated")):
        raise ValueError("Assisted/model scope review cannot claim independent expert validation")
    scopes = {r["claim_id"]: r for r in scope["claims"]}
    if len(scopes) != len(scope["claims"]) or not set(scopes) <= {r["claim_id"] for r in selected}:
        raise ValueError("Duplicate or out-of-video scope-review claim")
    passages = [("speech_text", None, transcript.get("text", ""))]
    passages += [("speech_segment", s.get("start"), s.get("text", "")) for s in transcript.get("segments", [])]
    passages += [("on_screen_text", s.get("timestamp"), s.get("text", "")) for s in transcript.get("on_screen_text", [])]
    entries, assessment_inputs = [], []
    for item in selected:
        claim = item["claim"]
        cid, text = item["claim_id"], claim["claim_text"]
        origins = [{"kind": kind, "timestamp": stamp, "quote": text}
                   for kind, stamp, source in passages if normalized(text) in normalized(source)]
        if not origins:
            raise ValueError("Claim wording is not grounded in supplied transcript/OCR: " + cid)
        row = workflow_rows.get(cid)
        if not row or normalized(row["claim_text"]) != normalized(text):
            raise ValueError("Workflow summary claim mismatch: " + cid)
        entry = {"claim_id": cid, "claim_text": text, "claim_origin": origins,
                 "workflow_status": row["status"], "automation_state": row.get("verification"),
                 "disposition": "pending", "verdict": None}
        recorded = row.get("adjudication")
        if recorded:
            if cid in scopes:
                raise ValueError("Scope abstention must not overwrite an adjudication: " + cid)
            path = Path(recorded["input_path"]).resolve()
            raw = path.read_bytes()
            if sha(raw) != recorded["input_sha256"]:
                raise ValueError("Adjudicated assessment changed: " + cid)
            data = json.loads(raw)
            if (data["claim"]["claim_id"] != cid
                    or normalized(data["claim"]["claim_text"]) != normalized(text)):
                raise ValueError("Assessment belongs to another claim: " + cid)
            checked = decide(data, sha(raw))
            if (checked["status"] == "blocked" or checked.get("verdict") is None
                    or checked["verdict"] != recorded["verdict"] or checked["status"] != recorded["status"]):
                raise ValueError("Recorded adjudication does not match validated assessment: " + cid)
            legacy = data.get("status") == "assisted_verification_ready_for_adjudication"
            entry.update(disposition="adjudicated", verdict=checked["verdict"],
                         method="assisted_legacy" if legacy else data["review"]["method"],
                         summary=checked["summary"], evidence_gaps=checked.get("evidence_gaps", []),
                         assessment={"path": str(path), "sha256": sha(raw)},
                         limitations=checked.get("limitations", []))
            assessment_inputs.append(entry["assessment"])
        elif cid in scopes:
            reviewed = scopes[cid]
            if (row["status"] != "claim_clarification_required"
                    or reviewed.get("claim_text") != text
                    or reviewed.get("claim_text_sha256") != sha(text.encode())
                    or reviewed.get("disposition") != "scope_abstention"
                    or reviewed.get("verdict") is not None
                    or not isinstance(reviewed.get("reason"), str) or not reviewed["reason"].strip()
                    or not isinstance(reviewed.get("missing_information"), list) or not reviewed["missing_information"]
                    or any(not isinstance(s, str) or not s.strip() for s in reviewed["missing_information"])):
                raise ValueError("Invalid or unsupported scope abstention: " + cid)
            entry.update(disposition="scope_abstention", method=scope["method"],
                         reason=reviewed["reason"], missing_information=reviewed["missing_information"],
                         scope_review={k: scope[k] for k in ("reviewer", "method", "scope", "independently_reviewed", "expert_validated")})
        entries.append(entry)
    warnings = []
    for issue in inventory["artifact_issues"]:
        other_id = None
        path = Path(issue["path"]).resolve()
        if (issue["problem"] == "Claim ID is outside the extracted-claims inventory"
                and path.is_relative_to(Path(args.experiments).resolve())):
            try:
                other_id = json.loads(path.read_bytes()).get("claim", {}).get("claim_id")
            except (OSError, ValueError, AttributeError):
                pass
        outside = isinstance(other_id, str) and bool(other_id.strip()) and not other_id.startswith(video_id + "-")
        warnings.append({**issue, "referenced_claim_id": other_id,
                         "scope": "outside_current_video" if outside else "requires_attention",
                         "blocking_for_this_report": not outside})
    counts = dict(Counter(e["disposition"] for e in entries))
    report = {"profile": "video-claim-report-v1", "video_id": video_id,
              "report_status": "requires_attention" if counts.get("pending") or any(w["blocking_for_this_report"] for w in warnings)
                               else "report_with_scope_abstentions" if counts.get("scope_abstention") else "report_exported",
              "claim_count": len(entries), "disposition_counts": counts,
              "speech_text_available": any(any(c.isalnum() for c in source) for kind, _, source in passages if kind != "on_screen_text"),
              "claims": entries, "scope_review": scope if scope_path else None,
              "inventory_warnings": warnings, "model_generation_calls": 0,
              "inputs": {"transcript": {"path": str(transcript_path), "sha256": sha(transcript_raw)},
                         "workflow_summary": {"path": str(summary_path), "sha256": sha(summary_raw)},
                         "scope_review": {"path": str(scope_path), "sha256": sha(scope_raw)} if scope_path else None,
                         "assessments": assessment_inputs,
                         "claims_files": [{"path": str(p), "sha256": sha(p.read_bytes())}
                             for p in sorted({Path(r["claims_file"]).resolve() for r in selected})],
                         "exporter_sha256": sha(Path(__file__).read_bytes())},
              "limitations": ["Exports supplied assessments and reviewed scope abstentions; performs no new scientific appraisal.",
                  "Assisted assessments and legacy reports are not evidence of reliable autonomous verification.",
                  "Insufficient evidence concerns the supplied appraised sources; it does not establish that a claim is false.",
                  "Scope abstentions have no medical verdict. Underlying clarification states and all attempt histories are preserved.",
                  "No independent expert validation or held-out scientific evaluation is established by this report."]}
    lines = ["# Health claim assessment: " + video_id, "", "Report status: " + report["report_status"], "",
             "Dispositions: " + ", ".join(f"{k}: {v}" for k, v in counts.items()), ""]
    for entry in entries:
        lines += ["## " + entry["claim_id"], "", entry["claim_text"], "",
                  "Disposition: " + entry["disposition"], "",
                  "Medical verdict: " + str(entry["verdict"] or "not assigned"), "",
                  "Method: " + entry.get("method", "pending"), "",
                  entry.get("summary", entry.get("reason", "Assessment remains pending.")), ""]
        for gap in entry.get("evidence_gaps", entry.get("missing_information", [])):
            lines += ["- " + gap]
        lines += [""]
    lines += ["## Limitations", ""] + ["- " + text for text in report["limitations"]]
    lines += ["", "## Inventory warnings", ""]
    lines += ["- " + w["scope"] + ": " + w["path"] + " — " + w["problem"] for w in warnings]
    markdown = ("\n".join(lines) + "\n").encode()
    folder = Path(args.report_dir).resolve()
    protected = [transcript_path, summary_path] + ([scope_path] if scope_path else []) + [Path(r["path"]) for r in assessment_inputs]
    if any(p.is_relative_to(folder) for p in protected):
        raise ValueError("Keep report output separate from input files")
    with exclusive_run(folder):
        json_path, md_path = folder / "video-report.json", folder / "video-report.md"
        if json_path.exists() and json.loads(json_path.read_bytes()) != report:
            raise ValueError("Existing report differs; preserve it and choose a new output directory")
        if md_path.exists() and md_path.read_bytes() != markdown:
            raise ValueError("Existing Markdown report differs; original preserved")
        dump(json_path, report)
        atomic_write(md_path, markdown)
    print("VIDEO:", video_id)
    print("DISPOSITIONS:", counts)
    print("STATUS:", report["report_status"])
    print("OUT-OF-SCOPE WARNINGS:", sum(not w["blocking_for_this_report"] for w in warnings))
    print("MODEL GENERATION CALLS: 0")
    print("REPORT:", json_path)
    print("READABLE REPORT:", md_path)
    return report



# Supported reviewed finish mode v1: no model generation or clinical auto-approval.
def finish_reviewed_workflow(args):
    import hashlib
    from importlib.metadata import version
    import sys
    from types import SimpleNamespace
    import claim_workflow
    import adjudicate_batch
    from retrieval_controller import _generic_outcome
    from verify_claim_agent import atomic_write, exclusive_run

    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        return json.loads(raw), {"path": str(path), "sha256": digest(raw)}

    inventory = build(args.claims_dir, args.experiments, args.summary)
    rows = {
        r["claim_id"]: dict(r, origin="registered_claim")
        for r in inventory["claims"]
    }
    inputs, unresolved, explicit = {}, [], {}

    for path in sorted({
        Path(r["claims_file"]).resolve() for r in rows.values()
    }):
        _, ref = read(path)
        inputs[ref["path"]] = ref

    for name in args.include_reviewed:
        try:
            data, ref = read(name)
            inputs[ref["path"]] = ref
            claim = data["claim"]
            cid, text = claim["claim_id"], claim["claim_text"]

            if (
                not isinstance(cid, str) or not cid.strip()
                or not isinstance(text, str) or not text.strip()
            ):
                raise ValueError("Missing claim identity")

            if (
                cid in rows
                and normalized(rows[cid]["claim"]["claim_text"])
                != normalized(text)
            ):
                rows[cid]["selection_error"] = (
                    "Explicit assessment wording differs from the registered claim."
                )
                raise ValueError(rows[cid]["selection_error"])

            if cid not in rows:
                rows[cid] = {
                    "claim_id": cid,
                    "claim": claim,
                    "candidate_reports": [],
                    "origin": (
                        "explicit_reviewed_assessment; "
                        "video extraction not revalidated"
                    ),
                }
            explicit.setdefault(cid, {})[ref["sha256"]] = ref["path"]

        except (OSError, ValueError, KeyError, TypeError) as error:
            unresolved.append({
                "path": str(Path(name).resolve()),
                "reason": str(error),
            })

    for cid, choices in explicit.items():
        rows[cid]["candidate_reports"] = sorted(choices.values())

    ready, pending, refs = [], [], {}

    for cid, row in sorted(rows.items()):
        claim = row["claim"]
        choices = row.get("candidate_reports", [])
        problem = row.get("selection_error")

        if not problem and len(choices) == 1:
            try:
                data, ref = read(choices[0])
                inputs[ref["path"]] = ref

                if (
                    data["claim"]["claim_id"] != cid
                    or normalized(data["claim"]["claim_text"])
                    != normalized(claim["claim_text"])
                ):
                    raise ValueError("Assessment identity mismatch")

                decision = adjudicate_batch.decide(data, ref["sha256"])
                if (
                    decision["status"] == "blocked"
                    or not decision.get("verdict")
                ):
                    raise ValueError(
                        decision.get("reason", "Scientific review is incomplete")
                    )

                ready.append({
                    "claim_id": cid,
                    "claim": claim,
                    "candidate_reports": [ref["path"]],
                })
                refs[cid] = ref
                continue

            except (OSError, ValueError, KeyError, TypeError) as error:
                problem = str(error)

        outcome = claim.get("outcome")
        vague = (
            not isinstance(outcome, str)
            or not outcome.strip()
            or _generic_outcome(outcome)
        )
        pending.append({
            "claim_id": cid,
            "claim_text": claim["claim_text"],
            "verdict": None,
            "origin": row["origin"],
            "disposition": (
                "abstained"
                if vague and not problem and not choices
                else "review_required"
            ),
            "reason": problem or (
                "Choose one reviewed assessment."
                if len(choices) > 1 else
                "Define the health outcome or guideline-alignment task."
                if vague else
                "Complete verification and recorded review."
            ),
        })

    folder = Path(args.finish_dir).resolve()
    if (
        folder.is_relative_to(Path(args.claims_dir).resolve())
        or any(Path(p).is_relative_to(folder) for p in inputs)
    ):
        raise ValueError(
            "Keep finish outputs separate from original claim and assessment files."
        )

    binding = {
        "profile": "reviewed-finish-v1",
        "inputs": sorted(inputs.values(), key=lambda r: r["path"]),
        "claim_specs": {
            cid: row["claim"] for cid, row in sorted(rows.items())
        },
        "selected": refs,
        "pending": pending,
        "unresolved_inputs": unresolved,
        "code": {
            "reporter": digest(Path(__file__).read_bytes()),
            "controller": digest(Path(claim_workflow.__file__).read_bytes()),
            "adjudicator": digest(Path(adjudicate_batch.__file__).read_bytes()),
        },
        "versions": {
            name: version(name)
            for name in (
                "langgraph", "langgraph-checkpoint-sqlite", "pydantic"
            )
        },
    }

    with exclusive_run(folder):
        path = folder / "finish-binding.json"
        if path.exists() and json.loads(path.read_bytes()) != binding:
            raise ValueError(
                "Finish inputs/code changed; preserve this run "
                "and choose a new --finish-dir."
            )
        dump(path, binding)

        for ref in inputs.values():
            if digest(Path(ref["path"]).read_bytes()) != ref["sha256"]:
                raise ValueError(
                    "Input changed before execution: " + ref["path"]
                )

        finished, controller_ref = [], None

        if ready:
            worklist = folder / "reviewed-worklist.json"
            dump(worklist, {"claims": ready})

            claim_workflow.run(SimpleNamespace(
                worklist=str(worklist),
                source_review=None,
                run_dir=str(folder / "controller"),
                resume=None,
                assessment=None,
                retrieve=False,
                screening_review=None,
                max_new_calls=0,
            ))

            summary, controller_ref = read(
                folder / "controller/workflow-summary.json"
            )
            for row in summary["claims"]:
                cid = row["claim_id"]
                result = row.get("adjudication")

                if (
                    not result
                    or row["status"] != "completed_from_reviewed_assessment"
                    or result["input_sha256"] != refs[cid]["sha256"]
                ):
                    raise ValueError(
                        "Controller did not complete the selected assessment: "
                        + cid
                    )

                finished.append({
                    "claim_id": cid,
                    "claim_text": row["claim_text"],
                    "disposition": "adjudicated",
                    "verdict": result["verdict"],
                    "origin": rows[cid]["origin"],
                    "review_method": result.get("review", {}).get(
                        "method", "assisted_legacy"
                    ),
                    "reason": result.get(
                        "summary", result.get("reason", "")
                    ),
                    "assessment": refs[cid],
                    "adjudication": result,
                })

        claims = sorted(
            finished + pending, key=lambda r: r["claim_id"]
        )
        counts = dict(Counter(
            r["disposition"] for r in claims
        ))
        needs_review = bool(
            counts.get("review_required") or unresolved
        )

        report = {
            "profile": "reviewed-workflow-release-v1",
            "mode": "reviewed",
            "status": (
                "needs_review" if needs_review else
                "complete_with_abstentions" if counts.get("abstained") else
                "complete_reviewed_run"
            ),
            "scope": (
                "Existing reviewed assessments -> persistent LangGraph "
                "adjudication -> report. No new extraction, retrieval "
                "or scientific verification."
            ),
            "model_generation_calls": 0,
            "autonomous_accuracy_validated": False,
            "dispositions": counts,
            "claims": claims,
            "unresolved_inputs": unresolved,
            "controller_summary": controller_ref,
            "binding": binding,
            "limitations": [
                "Assisted or model reviews are not independent expert validation.",
                "Insufficient evidence applies to the supplied appraised "
                "sources and does not mean false.",
                "Abstentions and review-required cases have no medical verdict.",
                "Explicit external assessments are reported as their exact "
                "reviewed claims; source-video extraction is not revalidated.",
                "Original assessments, model attempts and earlier "
                "checkpoints remain unchanged.",
            ],
        }

        lines = [
            "# Reviewed health-claim verdicts", "",
            "Status: " + report["status"], "",
            report["scope"], "",
        ]

        for row in claims:
            lines += [
                "## " + row["claim_id"], "",
                row["claim_text"], "",
                "Result: " + (row["verdict"] or row["disposition"]), "",
                "Review: " + row.get("review_method", "pending"), "",
                row["reason"], "",
            ]
            for criterion in row.get("adjudication", {}).get("criteria", []):
                lines += [
                    "- " + criterion["wording"]
                    + " — " + criterion["conclusion"]
                ]
                for citation in criterion["citations"]:
                    lines += [
                        "  - " + citation["source_id"]
                        + ': "' + citation["quote"] + '"'
                    ]
            lines += [""]

        lines += ["## Limits", ""] + [
            "- " + s for s in report["limitations"]
        ]
        if unresolved:
            lines += ["", "## Unresolved inputs", ""] + [
                "- " + r["path"] + ": " + r["reason"]
                for r in unresolved
            ]

        markdown = ("\n".join(lines) + "\n").encode()
        target = folder / "final-report.json"
        readable = folder / "final-report.md"

        if target.exists() and json.loads(target.read_bytes()) != report:
            raise ValueError("Existing final report differs; preserved.")
        if readable.exists() and readable.read_bytes() != markdown:
            raise ValueError("Existing Markdown report differs; preserved.")

        dump(target, report)
        atomic_write(readable, markdown)

        print("FINAL STATUS:", report["status"])
        print("DISPOSITIONS:", counts)
        for row in claims:
            print(
                row["claim_id"], "|",
                row["verdict"] or row["disposition"]
            )
        print("MODEL GENERATION CALLS: 0")
        print("REPORT:", readable)
        print("JSON:", target)

        if args.open_report and sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", str(readable)], check=False)

    return 2 if needs_review else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--claims-dir",
        default="data/processed/claims",
    )
    parser.add_argument(
        "--experiments", "--artifacts-dir",
        dest="experiments",
        default="data",
        help="Root containing experiment, processed and diagnostic artifacts",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional extra adjudication summary outside the artifact root",
    )
    parser.add_argument(
        "--out",
        default="data/processed/project-claim-status.json",
    )

    parser.add_argument("--transcript")
    parser.add_argument("--scope-review")
    parser.add_argument("--report-dir")
    parser.add_argument("--finish-reviewed", action="store_true", help="Run accepted reviews through the controller and report")
    parser.add_argument("--include-reviewed", action="append", default=[], metavar="PATH")
    parser.add_argument("--finish-dir", default="data/processed/reviewed-release-v1")
    parser.add_argument("--open-report", action="store_true")
    args = parser.parse_args()

    if args.finish_reviewed:
        if any(getattr(args, k, None) for k in ("transcript", "scope_review", "report_dir")):
            parser.error("Use --finish-dir in reviewed finish mode")
        raise SystemExit(finish_reviewed_workflow(args))
    if args.include_reviewed or args.open_report:
        parser.error("include-reviewed and open-report require --finish-reviewed")
    if args.report_dir and not (args.transcript and args.summary):
        parser.error("report-dir requires transcript and summary")

    result = build(
        args.claims_dir,
        args.experiments,
        args.summary,
    )

    if args.report_dir:
        export_video_report(result, args)
        raise SystemExit(0)
    dump(Path(args.out), result)

    print("TOTAL CLAIMS:", result["total_claims"])

    for row in result["claims"]:
        print(
            row["claim_id"],
            "|",
            row["status"],
            "|",
            row["verdict"] or "-",
        )
        print(" ", row["claim"]["claim_text"])

        if row["claim_specification_note"]:
            print("  NOTE:", row["claim_specification_note"])

    print("ARTIFACT ISSUES:", len(result["artifact_issues"]))
    print("REPORT:", Path(args.out).resolve())