
# Agentic Health Claim Verification — Project Instructions

## Objective

Build a research-oriented, genuinely agentic AI system that verifies individual health claims extracted from short-form videos.

The system must be technically substantial, evidence-grounded, modular, testable, and suitable for eventual conference-paper evaluation.

Do not reduce it to a fixed chain of LLM prompts or a basic RAG chatbot.

## Existing Pipeline

1. Video processing using FFmpeg, Whisper, OCR and frame extraction.
2. Atomic health-claim extraction with exposure, outcome and context.
3. PubMed evidence retrieval.
4. Biomedical reranking using S-PubMedBert-MS-MARCO.
5. Evidence Suitability Agent classifying articles as direct, partial or irrelevant.
6. Adaptive query planning, PubMed retrieval, PMID deduplication and query caching.
7. Reassessment of retrieved evidence.

Inspect the actual repository before assuming any module is complete.

## Current Progress

The first adaptive retrieval and suitability reassessment cycle has been tested for claim fitness-02-c001.

Original suitability: 0 direct, 0 partial, 3 irrelevant articles.

Reassessment of 13 combined papers: 0 direct, 2 partial, 11 irrelevant, according to preliminary model-generated labels.

The latest reassessment is stored at:
data/processed/adaptive_evidence/fitness-02-c001-reassessment.json

The Evidence Suitability Agent still reports missing evidence.

The next task is to build a retrieval decision controller that uses reassessment feedback, tracked retrieval state and a bounded retry budget.

## Intended Architecture

- Claim Understanding Agent
- Scientific Evidence Retrieval Planner
- PubMed and additional trustworthy biomedical tools
- Biomedical Semantic Reranker
- Evidence Suitability Agent
- Adaptive Retrieval Controller
- Scientific Verification Agent
- Critical Review Agent
- Adjudication Agent
- Stateful Orchestrator

The final verdicts are:
Supported, Refuted, Misleading, Insufficient Evidence.

These verdicts are distinct from evidence suitability labels:
direct, partial, irrelevant.

## Agentic Requirements

Agents must have meaningful responsibilities, access to appropriate tools, structured state, explicit decision rules, and observable outputs.

Implement feedback-driven retrieval, conditional branching, tool-failure recovery, query history, PMID deduplication, bounded retries and stopping criteria.

Use LLMs for language understanding and reasoning where appropriate, but use ordinary Python, biomedical models, APIs, validation and deterministic control logic when more suitable.

Do not add unnecessary agents merely to increase complexity.

Consider LangGraph for stateful orchestration after reviewing existing code and integration needs.

## Scientific Integrity

Never fabricate research papers, PMIDs, citations, experimental results or evaluation metrics.

Do not interpret high embedding similarity as scientific relevance.

Assess exposure, outcome, context, study applicability and evidence limitations.

Lack of suitable evidence is not proof a claim is false.

Evidence suitability is not the same as evidence stance or strength.

Treat preliminary LLM assessments as unverified until independently evaluated.

Preserve traceability from video to claim to source to final verdict.

## Development Rules

- Work in the existing repository.
- Inspect relevant files before modifying them.
- Reuse existing working functions.
- Avoid unnecessary rewrites and duplicated code.
- Prefer incremental, testable milestones.
- Add tests for control flow, errors and edge cases.
- Cache expensive model outputs where appropriate.
- Never print or commit API secrets.
- Ask before installing new dependencies or making major architectural changes.
- Do not commit, push or delete project files without explicit approval.
- Explain what changed, why, how it was tested, and what remains.
- Do not claim something is working unless a test or execution confirms it.

## Research Evaluation

Plan comparisons between baseline retrieval, biomedical reranking, suitability filtering, adaptive retrieval and the full verification framework.

Measure evidence relevance and final claim-level verification performance.

Use genuine annotations or validated benchmarks; do not invent results.
