# Run the reviewed demonstration

Run from the project root using the existing environment. This replay requires no API key, model generation or paid service.

## Reproduce

```bash
python3 scripts/project_claim_status.py --finish-reviewed --include-reviewed /Users/User/Desktop/agentic-health-fact-checking/data/experiments/fitness-02-c001/20261019T070530Z-785dd27d/data/processed/verification-local-v1/verification-assisted.json --include-reviewed /Users/User/Desktop/agentic-health-fact-checking/data/processed/claim-workflow-demo-v1/assisted-review-v1/fitness-02-c002.json --include-reviewed /Users/User/Desktop/agentic-health-fact-checking/data/processed/claim-workflow-demo-v1/assisted-review-v1/fitness-02-c003.json --include-reviewed /Users/User/Desktop/agentic-health-fact-checking/data/processed/claim-workflow-demo-v1/assisted-review-v1/fitness-02-c006.json --include-reviewed /Users/User/Desktop/agentic-health-fact-checking/data/diagnostics/nutrition-01-assessment-reviewed-v1/assessment-reviewed.json --finish-dir data/processed/reviewed-release-v1
```

The command reuses the existing bound inputs and checkpoints. If inputs or code have changed, preserve this release and choose a fresh finish directory. Do not reset hashes or attempt histories.

## Open the report on macOS

```bash
open data/processed/reviewed-release-v1/final-report.md
```

## Read the report in a terminal

```bash
cat data/processed/reviewed-release-v1/final-report.md
```

## Recorded result

8 claims, 5 adjudications and 3 abstentions.

The report consistency check passed on the user's computer. These development results include assisted review and have no independent accuracy estimate.

## Other reviewed claims

Supply the reviewed assessment with `--include-reviewed PATH` and choose a new finish directory. Existing claim-identity, source, citation and review gates still apply. Raw-video autonomous verification is outside this replay.
