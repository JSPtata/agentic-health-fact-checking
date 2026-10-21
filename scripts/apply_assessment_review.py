"""Attach a supplied, hash-bound review to a draft. No model calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Literal

from pydantic import Field
from adjudicate_batch import (
    Citation, Objection, ReadyReport, Strict,
    assessment_hash, decide, sha,
)
from verify_claim_agent import dump, exclusive_run


class CriterionReview(Strict):
    criterion_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    additional_citations: list[Citation] = Field(default_factory=list)


class Resolution(Strict):
    problem: str = Field(min_length=1)
    resolution: str = Field(min_length=1)


class ReviewRecord(Strict):
    schema_version: Literal["assessment-review-v1"]
    claim_id: str = Field(min_length=1)
    draft_assessment_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["accept", "revise"]
    reviewer: str = Field(min_length=1)
    method: Literal["human", "assisted", "model"]
    scope: str = Field(min_length=1)
    independently_reviewed: bool
    expert_validated: bool
    claim_requirements_complete: bool
    summary: str = Field(min_length=1)
    criterion_reviews: list[CriterionReview] = Field(min_length=1)
    resolved_objections: list[Resolution]
    unresolved_objections: list[Objection]


def apply_review(draft, provenance, record):
    ReadyReport.model_validate(draft)
    review = ReviewRecord.model_validate(record)
    original_hash = assessment_hash(draft)

    if (
        draft["workflow_status"] != "incomplete"
        or draft["review"]["decision"] != "revise"
    ):
        raise ValueError("Input must be an unapproved draft")

    if (
        review.claim_id != draft["claim"]["claim_id"]
        or review.draft_assessment_sha256 != original_hash
    ):
        raise ValueError("Review belongs to another claim or draft version")

    if (
        provenance["claim_id"] != review.claim_id
        or provenance["assessment_sha256"] != original_hash
    ):
        raise ValueError("Source provenance does not match this draft")

    if provenance.get("profile") == "claim-criteria-draft-v1":
        snapshot_hashes = {
            sid: item["text_sha256"]
            for sid, item in provenance["source_snapshots"].items()
        }
    else:
        # Compatibility with existing assisted PubMed assessments.
        snapshot_hashes = {
            "PMID:" + pmid: item["abstract_sha256"]
            for pmid, item in provenance["source_reviews"].items()
        }

    if snapshot_hashes != {
        sid: item["text_sha256"]
        for sid, item in draft["sources"].items()
    }:
        raise ValueError(
            "Source provenance coverage or hash mismatch"
        )

    # Validate original hashes and quotations before changing review status.
    decide(draft, "0" * 64)

    ids = [r.criterion_id for r in review.criterion_reviews]
    expected = [c["criterion_id"] for c in draft["criteria"]]

    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Review must cover every criterion exactly once")

    old_problems = {
        o["problem"] for o in draft["review"]["unresolved_objections"]
    }
    addressed = [r.problem for r in review.resolved_objections]

    if (
        len(addressed) != len(set(addressed))
        or set(addressed) != old_problems
    ):
        raise ValueError(
            "Every original objection needs an explicit resolution"
        )

    if (
        review.decision != "accept"
        or not review.claim_requirements_complete
        or any(
            o.severity in ("critical", "major")
            for o in review.unresolved_objections
        )
    ):
        raise ValueError(
            "Review still requires revision; no approved copy produced"
        )

    result = deepcopy(draft)
    by_id = {r.criterion_id: r for r in review.criterion_reviews}

    for criterion in result["criteria"]:
        for citation in by_id[criterion["criterion_id"]].additional_citations:
            value = citation.model_dump()
            if value not in criterion["citations"]:
                criterion["citations"].append(value)

    result["summary"] = review.summary
    result["workflow_status"] = "complete"
    result["review"] = {
        "decision": review.decision,
        "reviewer": review.reviewer,
        "method": review.method,
        "claim_requirements_complete": review.claim_requirements_complete,
        "assessment_sha256": assessment_hash(result),
        "unresolved_objections": [
            o.model_dump() for o in review.unresolved_objections
        ],
    }

    ReadyReport.model_validate(result)
    checked = decide(result, "0" * 64)

    if checked["status"] == "blocked":
        raise ValueError(
            "Adjudication validation blocked: " + checked["reason"]
        )

    return result, review



# Source-level review import v1. This does not approve a claim assessment.
class SourceCardReview(Strict):
    claim_id: str = Field(min_length=1)
    claim_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pmid: str = Field(pattern=r"^[0-9]+$")
    abstract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assessment: dict
    rationale: str = Field(min_length=1)
    corrections: list[str]


class SourceRoleReview(Strict):
    schema_version: Literal["source-role-review-v1"]
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer: str = Field(min_length=1)
    method: Literal["human", "assisted", "model"]
    scope: str = Field(min_length=1)
    independently_reviewed: bool
    expert_validated: bool
    source_reviews: list[SourceCardReview] = Field(min_length=1)
    evidence_gaps: dict[str, list[str]]


def apply_source_role_review(report, input_hash, record):
    import assess_source_roles as assessor
    import local_quote_backend

    review = SourceRoleReview.model_validate(record)

    if review.source_report_sha256 != input_hash:
        raise ValueError("Review belongs to another source report/version")

    if review.method != "human" and (
        review.independently_reviewed or review.expert_validated
    ):
        raise ValueError(
            "Assisted/model review cannot claim independent expert validation"
        )

    anchors_hash = sha(Path(local_quote_backend.__file__).read_bytes())
    if report["binding"]["anchors_sha256"] != anchors_hash:
        raise ValueError("Quotation-anchor code changed since source assessment")

    result = deepcopy(report)
    claims, expected = {}, {}

    for entry in result["claims"]:
        claim = entry["claim"]
        cid = claim["claim_id"]

        if cid in claims or entry.get("verdict") is not None:
            raise ValueError("Duplicate claim or unexpected source-level verdict")

        claims[cid] = entry

        for pmid, card in entry["source_cards"].items():
            source = card["source"]
            actual = sha(source["abstract"].encode())

            if actual != source["abstract_sha256"] or source["pmid"] != pmid:
                raise ValueError("Original source identity/hash mismatch: " + pmid)

            expected[(cid, pmid)] = (entry, card)

    supplied = [(r.claim_id, r.pmid) for r in review.source_reviews]

    if len(supplied) != len(set(supplied)) or set(supplied) != set(expected):
        raise ValueError(
            "Review must cover every supplied claim/source exactly once"
        )

    if set(review.evidence_gaps) != set(claims):
        raise ValueError("Evidence-gap entries must match the supplied claims")

    for item in review.source_reviews:
        entry, original = expected[(item.claim_id, item.pmid)]
        source = original["source"]

        if (
            item.claim_text_sha256
            != sha(entry["claim"]["claim_text"].encode())
            or item.abstract_sha256 != source["abstract_sha256"]
        ):
            raise ValueError("Claim wording or source changed since review")

        anchors = assessor.quote_anchors(source["abstract"])
        value = assessor.validate(deepcopy(item.assessment), anchors)
        card = assessor.make_card({"source": source}, value)

        card.update(
            status="source_interpretation_reviewed",
            semantic_review_required=False,
            adjudication_approved=False,
            source_review={
                "reviewer": review.reviewer,
                "method": review.method,
                "scope": review.scope,
                "independently_reviewed": review.independently_reviewed,
                "expert_validated": review.expert_validated,
                "rationale": item.rationale,
                "corrections": item.corrections,
                "source_report_sha256": input_hash,
            },
            original_model_interpretation={
                k: original.get(k)
                for k in ("status", "role", "fit", "reason", "error")
            },
        )
        entry["source_cards"][item.pmid] = card

    for cid, entry in claims.items():
        entry.update(
            status="source_interpretations_reviewed",
            verdict=None,
            claim_assessment_ready=False,
            evidence_gaps=review.evidence_gaps[cid],
        )
        entry["queues"] = {
            lane: [
                pmid for pmid, card in entry["source_cards"].items()
                if card["lane"] == lane
            ]
            for lane in (
                "effect_candidates",
                "guidance_candidates",
                "background_candidates",
                "needs_review",
            )
        }

    result["scope"] = (
        "Recorded source-interpretation review only. Claim coverage, scientific "
        "verification and adjudication remain pending; no claim verdict generated."
    )
    return result, review


def run_source_role_review(args):
    import local_quote_backend

    paths = [Path(p).resolve() for p in (args.source_roles, args.review)]
    folder = Path(args.out_dir).resolve()

    if any(path.is_relative_to(folder) for path in paths):
        raise ValueError("Keep input files outside the output directory")

    raw = [path.read_bytes() for path in paths]
    result, review = apply_source_role_review(
        json.loads(raw[0]), sha(raw[0]), json.loads(raw[1])
    )

    result["binding"] = {
        "profile": "source-roles-reviewed-v1",
        "parent_binding": result["binding"],
        "source_report_sha256": sha(raw[0]),
        "review_record_sha256": sha(raw[1]),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "anchors_sha256": sha(Path(local_quote_backend.__file__).read_bytes()),
    }

    outputs = {
        "source-roles.json": result,
        "source-review-record.json": json.loads(raw[1]),
    }

    with exclusive_run(folder):
        for name, value in outputs.items():
            target = folder / name
            if target.exists() and json.loads(target.read_bytes()) != value:
                raise ValueError(
                    "Existing reviewed output differs; preserve it "
                    "and use a new directory"
                )

        for name, value in outputs.items():
            dump(folder / name, value)

    print("STATUS: source_interpretations_reviewed")
    print("METHOD:", review.method)
    print("SOURCES:", len(review.source_reviews))
    print("CLAIM ASSESSMENT: pending; no verdict generated")
    print("API CALLS: 0")
    print("REPORT:", folder / "source-roles.json")


def run(args):
    paths = [
        Path(p).resolve()
        for p in (args.draft, args.provenance, args.review)
    ]
    folder = Path(args.out_dir).resolve()

    if any(p.is_relative_to(folder) for p in paths):
        raise ValueError("Keep input files outside the output directory")

    raw = [p.read_bytes() for p in paths]
    draft, provenance, record = [json.loads(b) for b in raw]
    result, review = apply_review(draft, provenance, record)

    outputs = {
        "assessment-reviewed.json": result,
        "review-record.json": record,
        "source-provenance.json": provenance,
        "manifest.json": ["assessment-reviewed.json"],
        "review-provenance.json": {
            "profile": "reviewed-assessment-import-v1",
            "inputs": [
                {"path": str(p), "sha256": sha(b)}
                for p, b in zip(paths, raw)
            ],
            "script_sha256": sha(Path(__file__).read_bytes()),
            "assessment_sha256": assessment_hash(result),
            "method": review.method,
            "scope": review.scope,
            "independently_reviewed": review.independently_reviewed,
            "expert_validated": review.expert_validated,
            "api_calls": 0,
        },
    }

    with exclusive_run(folder):
        for name, value in outputs.items():
            path = folder / name
            if path.exists() and json.loads(path.read_bytes()) != value:
                raise ValueError(
                    "Existing output differs; use a new output directory"
                )

        for name, value in outputs.items():
            dump(folder / name, value)

    print("STATUS: reviewed_assessment_ready")
    print("METHOD:", review.method)
    print("INDEPENDENT REVIEW:", review.independently_reviewed)
    print("API CALLS: 0")
    print("ASSESSMENT:", folder / "assessment-reviewed.json")
    print("MANIFEST:", folder / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft")
    parser.add_argument("--provenance")
    parser.add_argument("--review", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-roles")
    args = parser.parse_args()
    if args.source_roles:
        if args.draft or args.provenance:
            parser.error("Use --source-roles separately from --draft/--provenance")
        run_source_role_review(args)
    else:
        if not args.draft or not args.provenance:
            parser.error("Supply --source-roles or both --draft and --provenance")
        run(args)
