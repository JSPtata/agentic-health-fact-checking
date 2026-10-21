from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Action(StrictModel):
    tool: Literal[
        "search_official",
        "fetch_official",
        "search_pubmed",
        "inspect_source",
        "finish",
    ]

    rationale: str = Field(min_length=1, max_length=1200)
    target_gap: str = Field(min_length=1, max_length=1200)

    query: str | None = Field(default=None, max_length=500)
    domain: str | None = None
    candidate_id: str | None = None
    source_id: str | None = None


class Finding(StrictModel):
    source_id: str
    study_or_document_type: str
    population_and_intervention: str
    measured_outcome_or_recommendation: str
    finding: str

    relation: Literal[
        "support",
        "contradiction",
        "null_or_uncertain",
        "background",
        "unassessable",
    ]

    limitations: list[str]
    quotes: list[str] = Field(min_length=1, max_length=3)


class Component(StrictModel):
    component: Literal[
        "exposure_outcome",
        "context",
        "recommendation_strength",
    ]

    conclusion: Literal[
        "supported",
        "refuted",
        "mixed",
        "insufficient_evidence",
    ]

    source_ids: list[str]
    explanation: str


class Verification(StrictModel):
    claim_id: str
    findings: list[Finding]
    components: list[Component]

    provisional_verdict: Literal[
        "supported",
        "refuted",
        "misleading",
        "insufficient_evidence",
    ]

    summary: str
    unresolved_gaps: list[str]


class Objection(StrictModel):
    severity: Literal["critical", "major", "minor"]
    source_id: str | None = None
    source_quote: str | None = None
    problem: str
    requested_action: str


class Critique(StrictModel):
    decision: Literal[
        "accept_for_adjudication",
        "revise",
        "needs_more_evidence",
    ]

    objections: list[Objection]
    remaining_gaps: list[str]

# Shared claim-criteria contract: consolidation change 2.
import hashlib
import json

Verdict = Literal[
    "supported",
    "refuted",
    "misleading",
    "insufficient_evidence",
]


class Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class Claim(Strict):
    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)


class Citation(Strict):
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=12)


class Source(Strict):
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Criterion(Strict):
    criterion_id: str = Field(min_length=1)
    wording: str = Field(min_length=1)
    role: Literal["core", "qualifier"]
    conclusion: Literal[
        "supported",
        "refuted",
        "mixed",
        "insufficient_evidence",
    ]
    applicability: Literal["direct", "indirect", "unclear"]
    evidence_strength: Literal["sufficient", "limited"]
    rationale: str = Field(min_length=1)
    citations: list[Citation]


class AssessmentObjection(Strict):
    severity: Literal["critical", "major", "minor"]
    problem: str = Field(min_length=1)


class Review(Strict):
    decision: Literal["accept", "revise"]
    reviewer: str = Field(min_length=1)
    method: Literal["human", "assisted", "model"]
    claim_requirements_complete: bool
    assessment_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    unresolved_objections: list[AssessmentObjection]


class ReadyReport(Strict):
    schema_version: Literal["adjudication-ready-v1"]
    workflow_status: Literal["complete", "incomplete"]
    claim: Claim
    criteria: list[Criterion] = Field(min_length=1)
    sources: dict[str, Source]
    summary: str = Field(min_length=1)
    evidence_gaps: list[str]
    proposed_verdict: Verdict | None = None
    review: Review


def sha(data):
    return hashlib.sha256(data).hexdigest()


def assessment_hash(data):
    """Bind an upstream review to the exact assessment it reviewed."""
    content = {
        key: value
        for key, value in data.items()
        if key != "review"
    }
    content.setdefault("proposed_verdict", None)

    serialized = json.dumps(
        content,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha(serialized.encode())



class CriteriaVerification(StrictModel):
    claim_id: str = Field(min_length=1)
    findings: list[Finding]
    criteria: list[Criterion] = Field(min_length=1)
    summary: str = Field(min_length=1)
    unresolved_gaps: list[str]


def validate_criteria_verification(report, claim_id, sources):
    if report.claim_id != claim_id:
        raise ValueError("Wrong claim ID")

    ids = [finding.source_id for finding in report.findings]
    if len(ids) != len(set(ids)) or set(ids) != set(sources):
        raise ValueError(
            "Findings must cover every supplied source exactly once"
        )

    for finding in report.findings:
        for quote in finding.quotes:
            check_assessment_quote(quote, finding.source_id, sources)

    ids = [criterion.criterion_id for criterion in report.criteria]
    if (
        len(ids) != len(set(ids))
        or not any(c.role == "core" for c in report.criteria)
    ):
        raise ValueError(
            "Use unique criterion IDs and at least one core criterion"
        )

    for criterion in report.criteria:
        if (
            criterion.conclusion != "insufficient_evidence"
            and not criterion.citations
        ):
            raise ValueError(
                "Substantive conclusions require source quotations"
            )
        for citation in criterion.citations:
            check_assessment_quote(
                citation.quote, citation.source_id, sources
            )


def check_assessment_quote(quote, source_id, sources):
    normalized_quote = " ".join(quote.split())
    if (
        source_id not in sources
        or len(normalized_quote) < 12
        or normalized_quote not in " ".join(
            sources[source_id]["text"].split()
        )
    ):
        raise ValueError(
            "Unknown source or absent quotation: " + source_id
        )


def make_assessment_draft(claim, report, sources, critique, limitations):
    """Export evidence for review without manufacturing approval."""
    validate_criteria_verification(
        report, claim["claim_id"], sources
    )

    pending = [{
        "severity": "major",
        "problem": (
            "Claim coverage, quotation interpretation, applicability "
            "and evidence sufficiency require a recorded review."
        ),
    }]

    for objection in critique.objections:
        item = {
            "severity": objection.severity,
            "problem": objection.problem,
        }
        if item not in pending:
            pending.append(item)

    result = {
        "schema_version": "adjudication-ready-v1",
        "workflow_status": "incomplete",
        "claim": {
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
        },
        "criteria": [
            criterion.model_dump()
            for criterion in report.criteria
        ],
        "sources": {
            sid: {
                "text": source["text"],
                "text_sha256": sha(
                    source["text"].encode("utf-8")
                ),
            }
            for sid, source in sources.items()
        },
        "summary": report.summary,
        "evidence_gaps": list(dict.fromkeys(
            report.unresolved_gaps
            + critique.remaining_gaps
            + list(limitations)
        )),
        "proposed_verdict": None,
    }

    result["review"] = {
        "decision": "revise",
        "reviewer": "automated-criteria-verifier",
        "method": "model",
        "claim_requirements_complete": False,
        "assessment_sha256": assessment_hash(result),
        "unresolved_objections": pending,
    }

    ReadyReport.model_validate(result)
    return result
