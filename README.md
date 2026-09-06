# Health Misinformation Verification Pipeline

This project analyzes short health videos and is being built to identify
individual health claims, compare them with reliable evidence, and produce a
clear verdict.

## Pipeline

```text
Video
  ↓
Audio
  ↓
Transcript
  ↓
Health Claims
  ↓
Evidence
  ↓
Verification
  ↓
Final Verdict
```

Current status:

- ✅ Video/audio extraction
- ✅ Speech-to-text transcription
- 🟡 Video collection
- 🟡 OCR/keyframe processing
- ❌ Claim extraction implementation
- ❌ Evidence retrieval and source validation
- ❌ Verification and final verdict generation
- 🟡 Evaluation (unit tests only)

## Project structure

- `data/raw/videos/`: downloaded videos and original yt-dlp metadata
- `data/processed/audio/`: extracted WAV audio
- `data/processed/transcripts/`: typed transcript JSON
- `data/claims/`: planned output location; no claim files yet
- `data/evidence/`: planned output location; no evidence files yet
- `data/results/`: planned output location; no verdict files yet
- `src/health_pipeline/`: reusable Python pipeline code
- `scripts/`: beginner-friendly commands for collection and preprocessing
- `tests/`: unit tests for implemented behavior
- `docs/`: architecture and development notes

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the verified status and
[QUICK_START.md](QUICK_START.md) for commands.

## Install

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,preprocessing]"
```

`ffmpeg` is required for audio extraction. Optional model downloads occur on
the first faster-whisper run.

## Run implemented stages

Search candidate videos:

```powershell
python scripts/collect_videos.py --search
```

Download reviewed URLs:

```powershell
python scripts/collect_videos.py --download
```

Process complete downloaded videos:

```powershell
python scripts/process_videos.py --model tiny
```

Run tests:

```powershell
pytest -q
```

## Current dataset

- 25 candidate records
- 4 downloaded videos
- 21 failed downloads
- 3 transcript JSON files
- 1 video skipped because it has no audio
- 0 claim files
- 0 evidence files
- 0 final verdict files

## Limitations

The claim-extraction package currently defines an injected LLM interface but
does not provide a concrete LLM client or claim output command. OCR is also an
interface only. Evidence retrieval, verification, adjudication, and HealthFC
evaluation are not implemented. Downloaded content must be reviewed for
licensing and health-claim relevance before research use.
