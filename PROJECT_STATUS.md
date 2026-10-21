# Agentic Health Claim Fact-Checking

## Release status

The reviewed demonstration is complete. It runs existing reviewed assessments through persistent LangGraph adjudication and produces a per-claim report.

Scientific assessments include assisted review. Reliable autonomous verification of unseen videos has not been established.

## Observed evaluation

Report: [final-report.md](data/processed/reviewed-release-v1/final-report.md)

Machine-readable report SHA-256: `7c373c64b623e26157068c9fd0fbeb529a0536ffb573beeebb37ff493c311db2`.

| Measure | Observed value | Meaning |
|---|---:|---|
| Claims accounted for | 8/8 | Disposition coverage |
| Adjudicated | 5/8 (62.5%) | Reviewed verdict coverage, not accuracy |
| Abstained | 3/8 (37.5%) | No medical verdict assigned |
| Open review cases | 0 | This saved release only |
| Model generation calls during finish stage | 0 | Earlier stages used models |
| Independent scientific accuracy | Unmeasured | No independently labelled held-out evaluation completed |

| Claim | Result | Review method |
|---|---|---|
| fitness-02-c001 | insufficient_evidence | assisted_legacy |
| fitness-02-c002 | insufficient_evidence | assisted |
| fitness-02-c003 | insufficient_evidence | assisted |
| fitness-02-c004 | abstained | scope abstention |
| fitness-02-c005 | abstained | scope abstention |
| fitness-02-c006 | insufficient_evidence | assisted |
| fitness-02-c007 | abstained | scope abstention |
| nutrition-01-diagnostic-c001 | insufficient_evidence | assisted |

Insufficient evidence describes the supplied appraised evidence and does not establish that a claim is false. Abstention is separate from a medical verdict. Assisted acceptance is separate from independent expert validation. Source-video extraction is not revalidated for explicitly imported external assessments.

## Architecture

The research workflow is:

Video/audio/OCR -> atomic claims -> PubMed retrieval -> biomedical bi-encoder passage ranking -> cross-encoder reranking -> relevance and suitability screening -> scientific verification -> critical review -> adjudication -> report.

Retrieval, ranking, local verification and official-source tools exist as research components. The validated release entry point consumes existing reviewed assessments; it does not automatically execute every upstream stage.

LangGraph supplies persistent orchestration. Source hashes, exact claim identity, recorded review, bounded attempts and explicit failure states control progression. Runtime failures do not create medical verdicts. Official pages without relevant passages provide no claim-specific support.

## Scientific evaluation method

1. Freeze source snapshots, prompts, model versions and budgets. Keep current cases in the development set. Split unseen test videos by topic and duplicate claim to limit leakage.
2. Independently label claim spans, intervention, comparator, population, endpoint and timing. Keep body-fat loss distinct from body-weight change. Identify vague claims before retrieval.
3. Obtain independent reference judgements for relevance and evidence conclusions. Resolve reviewer disagreements through adjudication. Keep reference explanations and verdicts out of evaluated model inputs.
4. Compare the original small PubMed pool, a wider pool with biomedical bi-encoder ranking, and that same wider pool with cross-encoder reranking. Keep retrieval budgets and judged sources comparable.
5. Measure retrieval Recall@20 and nDCG@10, disclosing incomplete relevance judgements. Measure extraction precision/recall, endpoint alignment, quotation validity and citation entailment separately.
6. Report verdict macro-F1, confusion matrices and sample counts. Report verdict coverage, abstention rate and accuracy among decided cases together. Include technical failures in the overall accounting.
7. Report latency, attempts and generation cost separately. Verify dataset licences and label definitions before using external benchmarks.

No clinical precision, recall, F1 or accuracy score is assigned to this release. Available evaluation consists of descriptive workflow results and software consistency checks.

## Reproduction and preservation

See [QUICK_START.md](QUICK_START.md).

Preserve the complete reviewed-release folder together with its referenced assessments and code. Absolute paths bind this run to this checkout. Changed inputs or code require a new output directory.

## Remaining research work

- Reliable automatic claim extraction and source screening on unseen videos.
- Faithful scientific verification with endpoint and population alignment.
- Integrated recovery across upstream stages.
- Independent scientific evaluation.

This release is an assisted research prototype with a completed reviewed demonstration.
