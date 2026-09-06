# Dataset layout

The balanced starter set is five public YouTube Shorts per domain:

- nutrition/diet
- supplements
- vaccines
- cancer
- mental health

Put one URL in each manifest row in `manifest.csv`. The `status` column is
intentionally `pending` until the file has been downloaded and processed.
Add human-reviewed claim labels before using the records for evaluation or
fine-tuning. Do not treat the video's topic as evidence that its claims are
misinformation.
