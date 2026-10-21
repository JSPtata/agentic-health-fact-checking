"""Reusable adjudication of reviewed evidence; no model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from adjudicate_claim import adjudicate as validate_assisted
from verify_claim_agent import dump, exclusive_run


POLICY = "reviewed-criteria-adjudication-v1"

from verification_schemas import (
    Verdict, Strict, Claim, Citation, Source, Criterion,
    AssessmentObjection as Objection, Review, ReadyReport,
    sha, assessment_hash,
)


def blocked(reason, action):
    return {
        "status": "blocked",
        "verdict": None,
        "reason": reason,
        "next_action": action,
    }


def decide(data, input_hash):
    # Compatibility with existing assisted reports.
    # Preserve their provenance and conservative evidence limits.
    if data.get("status") == "assisted_verification_ready_for_adjudication":
        old = validate_assisted(data, input_hash)

        if old["verdict"] != "insufficient_evidence":
            return blocked(
                "Legacy report needs the general reviewed-criteria contract.",
                "prepare_and_review_criteria",
            )

        old.update(
            policy=POLICY,
            status="adjudicated_assisted_legacy",
            next_action="stop_with_evidence_limits",
        )
        return old

    if data.get("schema_version") != "adjudication-ready-v1":
        return blocked(
            "A reviewed claim-level assessment is required; "
            "source extraction alone is insufficient.",
            "complete_verification_and_review",
        )

    report = ReadyReport.model_validate(data)

    if report.review.assessment_sha256 != assessment_hash(data):
        raise ValueError("Stale review: assessment hash mismatch")

    criterion_ids = [
        criterion.criterion_id
        for criterion in report.criteria
    ]

    if (
        len(criterion_ids) != len(set(criterion_ids))
        or not any(c.role == "core" for c in report.criteria)
    ):
        raise ValueError(
            "Criteria need unique IDs and at least one core requirement"
        )

    for sid, source in report.sources.items():
        raw_text = data["sources"][sid]["text"]

        if sha(raw_text.encode()) != source.text_sha256:
            raise ValueError(f"Source hash mismatch: {sid}")

    for criterion in report.criteria:
        if (
            criterion.conclusion != "insufficient_evidence"
            and not criterion.citations
        ):
            raise ValueError(
                f"Missing citations: {criterion.criterion_id}"
            )

        for citation in criterion.citations:
            source = report.sources.get(citation.source_id)

            if source is None:
                raise ValueError(
                    f"Unknown source or absent quotation: "
                    f"{criterion.criterion_id}"
                )

            quote = " ".join(citation.quote.split())
            source_text = " ".join(source.text.split())

            if quote not in source_text:
                raise ValueError(
                    f"Unknown source or absent quotation: "
                    f"{criterion.criterion_id}"
                )

    if report.workflow_status != "complete":
        return blocked(
            "Verification workflow is incomplete.",
            "resume_verification",
        )

    if not report.review.claim_requirements_complete:
        return blocked(
            "Review has not confirmed coverage of the exact claim wording.",
            "review_claim_requirements",
        )

    major_objections = any(
        objection.severity in ("critical", "major")
        for objection in report.review.unresolved_objections
    )

    if report.review.decision != "accept" or major_objections:
        return blocked(
            "Scientific review requires revision.",
            "resolve_review_objections",
        )

    def adequate(criterion, conclusion):
        return (
            criterion.conclusion == conclusion
            and criterion.applicability == "direct"
            and criterion.evidence_strength == "sufficient"
            and bool(criterion.citations)
        )

    core = [
        criterion
        for criterion in report.criteria
        if criterion.role == "core"
    ]

    qualifiers = [
        criterion
        for criterion in report.criteria
        if criterion.role == "qualifier"
    ]

    if any(adequate(c, "refuted") for c in core):
        verdict = "refuted"
        reason = (
            "Reviewed direct, sufficient evidence refutes a core "
            "requirement of the exact claim."
        )

    elif all(
        adequate(c, "supported")
        for c in report.criteria
    ):
        verdict = "supported"
        reason = (
            "Reviewed direct, sufficient evidence supports every "
            "required part of the wording."
        )

    elif (
        all(adequate(c, "supported") for c in core)
        and any(adequate(c, "refuted") for c in qualifiers)
    ):
        verdict = "misleading"
        reason = (
            "The core is supported, but direct, sufficient evidence "
            "refutes a material qualifier."
        )

    else:
        verdict = "insufficient_evidence"
        reason = (
            "The reviewed evidence does not establish support "
            "or refutation under this policy."
        )

    if (
        report.proposed_verdict
        and report.proposed_verdict != verdict
    ):
        return blocked(
            "Proposed verdict conflicts with the reviewed criteria and policy.",
            "reconcile_verdict_and_criteria",
        )

    return {
        "status": "adjudicated_from_reviewed_assessments",
        "verdict": verdict,
        "reason": reason,
        "next_action": (
            "stop_with_evidence_limits"
            if verdict == "insufficient_evidence"
            else "finish"
        ),
        "summary": report.summary,
        "criteria": [
            criterion.model_dump()
            for criterion in report.criteria
        ],
        "evidence_gaps": report.evidence_gaps,
        "review": report.review.model_dump(),
        "source_hashes": {
            sid: source.text_sha256
            for sid, source in report.sources.items()
        },
        "decision_scope": (
            "Rule-based synthesis of supplied reviewed assessments; "
            "citation presence does not prove entailment or "
            "scientific correctness."
        ),
    }


def run(args):
    manifest = Path(args.manifest).resolve()
    entries = json.loads(manifest.read_text())

    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Manifest must be a nonempty list of verification file paths"
        )

    inputs = []

    for name in entries:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each manifest entry must be a file path")

        path = (manifest.parent / name).resolve()

        try:
            raw = path.read_bytes()
            data = json.loads(raw)

            if not isinstance(data, dict):
                raise ValueError("Verification must be a JSON object")

            error = None

        except (OSError, ValueError) as exc:
            raw = None
            data = {}
            error = str(exc)

        claim = data.get("claim", {})
        claim = claim if isinstance(claim, dict) else {}

        claim_id = claim.get("claim_id")
        claim_id = (
            claim_id.strip()
            if isinstance(claim_id, str) and claim_id.strip()
            else None
        )

        inputs.append({
            "path": str(path),
            "sha256": sha(raw) if raw is not None else None,
            "data": data,
            "claim_id": claim_id,
            "error": error,
        })

    counts = Counter(
        item["claim_id"]
        for item in inputs
        if item["claim_id"]
    )

    binding = {
        "policy": POLICY,
        "script_sha256": sha(Path(__file__).read_bytes()),
        "inputs": [
            {
                key: item[key]
                for key in (
                    "path",
                    "sha256",
                    "claim_id",
                    "error",
                )
            }
            for item in inputs
        ],
    }

    folder = Path(args.out_dir).resolve()

    with exclusive_run(folder):
        state_path = folder / "batch-binding.json"

        if (
            state_path.exists()
            and json.loads(state_path.read_text()) != binding
        ):
            raise ValueError(
                "Batch inputs/code changed; choose a new output directory"
            )

        dump(state_path, binding)
        results = []

        for index, item in enumerate(inputs, 1):
            if item["error"]:
                result = blocked(
                    item["error"],
                    "repair_input_file",
                )

            elif not item["claim_id"]:
                result = blocked(
                    "Missing claim identity.",
                    "repair_input_file",
                )

            elif counts[item["claim_id"]] > 1:
                result = blocked(
                    "Duplicate claim ID: select exactly one run per claim.",
                    "select_claim_run",
                )

            else:
                try:
                    result = decide(
                        item["data"],
                        item["sha256"],
                    )

                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                ) as exc:
                    result = blocked(
                        str(exc),
                        "repair_or_review_assessment",
                    )

            result.update(
                claim_id=item["claim_id"],
                policy=POLICY,
                input_path=item["path"],
                input_sha256=item["sha256"],
            )

            claim = item["data"].get("claim", {})
            result["claim_text"] = (
                claim.get("claim_text")
                if isinstance(claim, dict)
                else None
            )

            output = folder / "claims" / f"{index:04d}.json"

            if (
                output.exists()
                and json.loads(output.read_text()) != result
            ):
                raise ValueError(
                    "Saved decision differs; choose a new output directory"
                )

            dump(output, result)
            results.append(result)

            print(
                item["claim_id"],
                "|",
                result["status"],
                "|",
                result["verdict"],
            )

        total_blocked = sum(
            result["status"] == "blocked"
            for result in results
        )

        summary = {
            "policy": POLICY,
            "total_inputs": len(results),
            "adjudicated": len(results) - total_blocked,
            "blocked": total_blocked,
            "verdict_counts": dict(
                Counter(
                    result["verdict"]
                    for result in results
                    if result["verdict"]
                )
            ),
            "results": results,
        }

        dump(folder / "batch-summary.json", summary)

        print("SUMMARY:", folder / "batch-summary.json")
        return 1 if total_blocked else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--out-dir")
    parser.add_argument("--print-schema", action="store_true")
    args = parser.parse_args()

    if args.print_schema:
        print(
            json.dumps(
                ReadyReport.model_json_schema(),
                indent=2,
            )
        )

    else:
        if not args.manifest or not args.out_dir:
            parser.error("--manifest and --out-dir are required")

        raise SystemExit(run(args))