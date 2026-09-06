# Agentic Health Fact-Checking Pipeline

This project analyzes short health videos to identify individual health claims,
compare them with reliable evidence, and produce a clear verdict.

## Problem and goal

Given a short-form health or nutrition video, the eventual system should
extract spoken content, split it into atomic claims, retrieve reliable evidence,
verify each claim, and classify it as supported, refuted, misleading, or
insufficient evidence.

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

- ✅ Audio extraction and speech-to-text
- 🟡 Video collection
- 🟡 OCR/keyframe processing
- ❌ Claim extraction implementation
- ❌ Evidence retrieval/source validation
- ❌ Verification, adjudication, and final verdict generation
- 🟡 Evaluation (unit tests only)

## Structure

- `data/raw/videos/`: downloaded videos and original yt-dlp metadata
- `data/processed/audio/`: extracted WAV audio
- `data/processed/transcripts/`: typed transcript JSON
- `data/claims/`, `data/evidence/`, `data/results/`: planned output locations
- `src/health_pipeline/`: reusable typed pipeline code
- `scripts/`: collection and preprocessing commands
- `tests/`: unit tests
- `docs/`: architecture and development notes

See [PROJECT_STATUS.md](PROJECT_STATUS.md) and [QUICK_START.md](QUICK_START.md).

## Install and run

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,preprocessing]"
python scripts/collect_videos.py --search
python scripts/collect_videos.py --download
python scripts/process_videos.py --model tiny
pytest -q
```

The currently runnable stages are collection, audio extraction, and
transcription. Claim extraction, evidence retrieval, verification, and
evaluation commands do not exist yet.

## Dataset and limitations

The manifest contains 25 candidate records: 4 downloaded videos and 21 failed
downloads. Three transcripts have been generated; one video was skipped because
it has no audio. There are currently zero claim, evidence, and verdict files.
Downloaded content requires license and health-claim review before research use.

Team members and the original research objectives are documented in the
repository history and project documentation.
