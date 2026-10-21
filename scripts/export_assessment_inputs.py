"""Export pending claims and cached abstracts. No API calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from adjudicate_batch import sha
from verify_claim_agent import dump, exclusive_run


def build(worklist, report):
    pending = {}
    seen = set()

    for row in worklist["claims"]:
        cid = row["claim_id"]
        claim = row["claim"]

        if (
            cid in seen
            or cid != claim["claim_id"]
            or not re.fullmatch(r"[A-Za-z0-9_-]+", cid)
        ):
            raise ValueError(
                "Duplicate or invalid claim identity: " + str(cid)
            )

        seen.add(cid)

        if row["status"] == "verification_pending":
            pending[cid] = claim

    bundles = {}

    for row in report["claims"]:
        cached = row["claim"]
        cid = cached["claim_id"]

        if cid not in pending:
            continue

        if cid in bundles:
            raise ValueError("Duplicate source-review claim: " + cid)

        claim = deepcopy(pending[cid])

        # Preparation annotations may differ; claim specifications must match.
        if {key: cached.get(key) for key in claim} != claim:
            raise ValueError(
                "Cached claim specification differs: " + cid
            )

        articles = {}

        for sid, card in row["source_cards"].items():
            metadata = card["metadata"]
            text = card["text"]
            pmid = str(metadata["pmid"])
            digest = sha(text.encode())

            if (
                metadata.get("kind") != "pubmed_abstract"
                or metadata.get("text_scope") != "abstract_only"
            ):
                raise ValueError(
                    "Expected a PubMed abstract snapshot: " + sid
                )

            if digest != metadata["text_sha256"]:
                raise ValueError("Source text hash mismatch: " + sid)

            expected_id = "S" + sha(
                (metadata["url"] + "\n" + text).encode()
            )[:16]

            if sid != metadata["id"] or sid != expected_id:
                raise ValueError("Source identity mismatch: " + sid)

            if (
                not re.fullmatch(r"[1-9][0-9]*", pmid)
                or pmid in articles
            ):
                raise ValueError(
                    "Invalid or duplicate PMID for " + cid
                )

            articles[pmid] = {
                **deepcopy(metadata),
                "abstract": text,
                "abstract_sha256": digest,
                "abstract_available": bool(text.strip()),
            }

        issues = []

        if not claim.get("outcome"):
            issues.append(
                "Structured outcome missing; do not invent an outcome."
            )

        if not articles:
            issues.append(
                "No cached candidates supplied; "
                "retrieval or source access is required."
            )

        bundles[cid] = {
            "claim": claim,
            "articles": articles,
            "status": "assessment_pending",
            "verdict": None,
            "claim_specification_issues": issues,
            "limitations": [
                "Existing candidate pool only; not an exhaustive literature search.",
                "Source hashes establish integrity, not scientific relevance or correctness.",
                "Previous model selections and review decisions are not imported as approved facts.",
                "Relevance screening, claim clarification where needed, assessment and review remain pending."
            ],
        }

    missing = set(pending) - set(bundles)

    if missing:
        raise ValueError(
            "Missing cached claim records: "
            + ", ".join(sorted(missing))
        )

    if not bundles:
        raise ValueError("No verification-pending claims found")

    return bundles


def run(args):
    inputs = [
        Path(path).resolve()
        for path in (args.worklist, args.source_review)
    ]
    folder = Path(args.out_dir).resolve()

    if any(path.is_relative_to(folder) for path in inputs):
        raise ValueError("Keep inputs outside the output directory")

    raw = [path.read_bytes() for path in inputs]
    bundles = build(*[json.loads(data) for data in raw])

    outputs = {
        cid + "/review-input.json": bundle
        for cid, bundle in sorted(bundles.items())
    }

    outputs["summary.json"] = {
        "profile": "pending-assessment-inputs-v1",
        "api_calls": 0,
        "inputs": [
            {"path": str(path), "sha256": sha(data)}
            for path, data in zip(inputs, raw)
        ],
        "script_sha256": sha(Path(__file__).read_bytes()),
        "claims": [
            {
                "claim_id": cid,
                "input": cid + "/review-input.json",
                "candidate_count": len(bundle["articles"]),
                "status": bundle["status"],
                "claim_specification_issues":
                    bundle["claim_specification_issues"],
                "verdict": None,
            }
            for cid, bundle in sorted(bundles.items())
        ],
    }

    with exclusive_run(folder):
        for name, value in outputs.items():
            path = folder / name
            if path.exists() and json.loads(path.read_bytes()) != value:
                raise ValueError(
                    "Existing output differs; use a new directory"
                )

        for name, value in outputs.items():
            dump(folder / name, value)

    for cid, bundle in sorted(bundles.items()):
        print(
            cid,
            "| assessment_pending | candidates:",
            len(bundle["articles"]),
        )
        for issue in bundle["claim_specification_issues"]:
            print("  NOTE:", issue)

    print("SOURCE REVIEW SHA256:", sha(raw[1]))
    print("API CALLS: 0")
    print("SUMMARY:", folder / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", required=True)
    parser.add_argument("--source-review", required=True)
    parser.add_argument("--out-dir", required=True)
    run(parser.parse_args())