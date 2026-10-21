"""Adjudicate a supplied assisted verification using explicit rules."""
import argparse
import hashlib
import json
from pathlib import Path

from verification_schemas import Verification
from verify_claim_agent import (
    dump,
    exclusive_run,
    validate_verification,
)


POLICY = "component-adjudication-v1"


def adjudicate(data, input_sha256):
    if (
        data.get("status")
        != "assisted_verification_ready_for_adjudication"
    ):
        raise ValueError(
            "Use the corrected assisted verification artifact"
        )

    claim = data["claim"]
    original = data["original_report"]

    if claim != original["claim"]:
        raise ValueError(
            "Claim does not match the reviewed original report"
        )

    verification = Verification.model_validate(
        data["verification"]
    )
    texts = data["source_texts"]

    if not texts:
        raise ValueError("No source snapshots supplied")

    registry = data["source_registry"]
    expected_hashes = data["provenance"]["source_hashes"]

    if set(expected_hashes) != set(texts):
        raise ValueError("Source hash coverage mismatch")

    if (
        expected_hashes
        != original["binding"]["source_hashes"]
    ):
        raise ValueError(
            "Sources do not match the original input binding"
        )

    sources = {}

    for sid, text in texts.items():
        actual = hashlib.sha256(text.encode()).hexdigest()

        if (
            actual != expected_hashes[sid]
            or actual != registry[sid]["text_sha256"]
        ):
            raise ValueError(f"Source hash mismatch: {sid}")

        sources[sid] = {"text": text}

    validate_verification(
        verification,
        claim["claim_id"],
        sources,
    )

    pairs = {
        (item["source_id"], item["part"])
        for item in original["passage_assessments"]
    }

    resolutions = data["review_reconciliation"]
    resolved = {
        (item["source_id"], item["part"])
        for item in resolutions
    }

    if (
        resolved != pairs
        or len(resolutions) != len(resolved)
    ):
        raise ValueError(
            "Missing or duplicate review reconciliation entries"
        )

    if any(
        not str(item.get("resolution", "")).strip()
        for item in resolutions
    ):
        raise ValueError("A review resolution is empty")

    conclusions = {
        component.component: component.conclusion
        for component in verification.components
    }
    values = set(conclusions.values())
    provisional = verification.provisional_verdict

    if (
        "refuted" in values
        or provisional in ("refuted", "misleading")
    ):
        status = "requires_manual_adjudication"
        verdict = None
        reason = (
            "Refutation and misleading classifications need "
            "explicit scientific adjudication; component "
            "labels alone are insufficient."
        )

    elif (
        values == {"supported"}
        and provisional == "supported"
    ):
        status = "adjudicated_with_evidence_limits"
        verdict = "supported"
        reason = (
            "All three supplied component assessments support "
            "the wording and agree with the provisional verdict. "
            "This policy does not independently establish that "
            "the assessments are correct."
        )

    elif (
        provisional == "insufficient_evidence"
        and values.intersection({
            "mixed",
            "insufficient_evidence",
        })
    ):
        status = "adjudicated_with_evidence_limits"
        verdict = "insufficient_evidence"
        reason = (
            "The supplied verification contains unresolved or "
            "heterogeneous component evidence and concludes "
            "insufficient evidence. Missing support is not "
            "converted into refutation."
        )

    else:
        status = "requires_manual_adjudication"
        verdict = None
        reason = (
            "The component assessments and provisional "
            "verdict require reconciliation."
        )

    return {
        "status": status,
        "claim_id": claim["claim_id"],
        "claim_text": claim["claim_text"],
        "verdict": verdict,
        "reason": reason,
        "summary": verification.summary,
        "components": [
            component.model_dump()
            for component in verification.components
        ],
        "evidence_gaps": verification.unresolved_gaps,
        "policy": POLICY,
        "input_sha256": input_sha256,
        "decision_scope": (
            "Classification of the supplied assisted verification; "
            "not a new evidence appraisal, independent scientific "
            "review or clinical recommendation."
        ),
        "verification_provenance": data["provenance"],
        "review_reconciliation_entries": len(resolutions),
        "source_count": len(sources),
        "sources": {
            sid: {
                key: registry[sid].get(key)
                for key in (
                    "title",
                    "url",
                    "pmid",
                    "text_sha256",
                )
            }
            for sid in sources
        },
        "limitations": data.get("limitations", []),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--verification", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    raw = Path(args.verification).read_bytes()
    input_hash = hashlib.sha256(raw).hexdigest()
    result = adjudicate(json.loads(raw), input_hash)

    folder = Path(args.out_dir).resolve()

    with exclusive_run(folder):
        output = folder / "adjudication.json"

        if output.exists():
            old = json.loads(output.read_text())

            if (
                old.get("input_sha256") != input_hash
                or old.get("policy") != POLICY
            ):
                raise FileExistsError(
                    "Different adjudication already exists; "
                    "choose a new output directory"
                )

            if old != result:
                raise ValueError(
                    "Saved adjudication does not match "
                    "the current validated result"
                )
        else:
            dump(output, result)

        print("STATUS:", result["status"])
        print("VERDICT:", result["verdict"])
        print("REPORT:", output)


if __name__ == "__main__":
    main()