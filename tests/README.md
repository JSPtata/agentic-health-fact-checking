# Tests

Tests are organized by responsibility:

- `test_collection.py`: manifest persistence
- `test_preprocessing.py`: ASR/OCR/audio adapter composition
- `test_process_videos.py`: ffmpeg discovery
- `test_claim_extraction.py`: structured LLM contract with a fake client

Run all tests with `pytest -q`.
