# Agentic Health Claim Fact-Checking

A research prototype for checking individual health claims in short videos, with persistent state, bounded attempts, source provenance, critical review and rule-based adjudication.

The completed demonstration follows **reviewed evidence -> LangGraph adjudication -> per-claim report**. Assessments include assisted review. Automatic verification of unseen videos and independent scientific accuracy remain research work.

- [Results, architecture, evaluation and limitations](PROJECT_STATUS.md)
- [Reproduce the reviewed demonstration](QUICK_START.md)
- [Final claim report](data/processed/reviewed-release-v1/final-report.md)
- [Machine-readable results](data/processed/reviewed-release-v1/final-report.json)

Each claim receives its own result. Underspecified claims can abstain without a medical verdict. Insufficient evidence is scoped to the appraised sources and does not mean false. Software checks and review acceptance do not establish clinical correctness.
