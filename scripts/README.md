# Supported research workflow

This is an assisted health-claim fact-checking research prototype.
Automated scientific verification
has not been validated as reliable. Reports do not establish medical accuracy.

Video -> speech/OCR -> atomic claims -> scope clarification -> PubMed search
-> biomedical bi-encoder -> MedCPT cross-encoder -> suitability screening
-> source verification -> critical review -> adjudication -> video report.

Ranking scores measure relevance, not truth. Model errors and timeouts are
operational failures, never scientific verdicts. Insufficient evidence does
not mean false. Scope abstentions have no medical verdict.

Current assisted demonstration: data/processed/claim-workflow-demo-v1.
Read final-report/video-report.md for its dispositions and limits.
No independent expert validation is claimed. Preserve the source snapshots,
assessments, review records, original worklist, attempt logs and SQLite state.
Do not reset budgets or change hashes to force a resume. Existing artifacts
can contain absolute paths; relocation requires migration and revalidation.

Supported entrypoints:
- claim_workflow.py: persistent graph, bounded calls and review gates.
- rank_cached_passages.py --workflow ... --cross-encoder: cached ranking.
- apply_assessment_review.py: validate review imports.
- adjudicate_batch.py: policy over approved assessments.
- project_claim_status.py: inventory and report export.

Run those CLIs with --help to inspect their options. The legacy frontend
uses positional arguments; do not import extraction/test scripts as an
offline check. Some execute requests at import time. In particular, do not
include test_gemini.py in offline test discovery.

Remaining gaps: local extraction fidelity and raw-media processing
and autonomous local verification need validation; retrieval/screening/review
still need complete orchestration and held-out scientific evaluation.
Evaluate retrieval recall, ranking relevance, extraction fidelity, citation
accuracy, verdict quality, abstention, latency and manual intervention.

The report reuses the existing environment and saved artifacts without model
calls. Full operation also needs the relevant Python dependencies, FFmpeg,
Tesseract, Ollama/model, PubMed access and ranking models. The root minimal
requirements.txt is incomplete for these research scripts; clean-machine setup is not yet
verified. This demonstration is not a held-out scientific evaluation.


## Supported reviewed finish mode

Run from the project root in the existing working Python environment:

```bash
python3 scripts/project_claim_status.py --finish-reviewed --open-report
```

Add `--include-reviewed PATH` for an explicitly selected approved assessment.
Use `--finish-dir PATH` for a new version of inputs or code. The command uses
LangGraph and the existing adjudication gates, writes JSON and Markdown,
and makes zero model/retrieval calls. It preserves old runs and attempt logs.
It does not revalidate source-video extraction for external assessments.

A medical verdict requires a valid accepted review, unchanged source hashes,
valid citations and the exact assessed claim. Missing scope leads to an
abstention; missing/invalid reviews lead to review_required with no verdict.
Exit status 2 means review remains; a report is still produced.

This supports the assisted reviewed-assessment-to-report prototype.
Reliable autonomous new-video processing and independent scientific evaluation
remain unfinished. No accuracy or expert-validation claim is established by
successful execution. Do not include legacy test_gemini.py in offline test discovery.
