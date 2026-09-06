# Project status

| Stage | Status | Current implementation | Input | Output | Next step |
|---|---|---|---|---|---|
| Video collection | 🟡 | yt-dlp search/download script with resumable statuses | YouTube search/URLs | `data/raw/videos/` | Replace failed candidates and review licenses |
| Audio extraction | ✅ | ffmpeg mono 16 kHz WAV extraction | video | `data/processed/audio/*.wav` | Keep validating media |
| Transcription | ✅ | faster-whisper `tiny` model | WAV | `data/processed/transcripts/*.transcript.json` | Review quality and configure model if needed |
| Claim extraction | ❌ | Injected `LLMClient` protocol and prompt only | Transcript | none | Add a concrete client and command |
| Evidence retrieval | ❌ | No module | claims | none | Implement after claim extraction |
| Source validation | ❌ | `Evidence` schema only | evidence | none | Add whitelist/provenance checks |
| Verification | ❌ | `Verdict` schema only | claim + evidence | none | Add verification agent |
| OCR | 🟡 | EasyOCR class exists, `recognize()` is unimplemented | video | none | Add keyframe sampling |
| Evaluation | 🟡 | Five unit tests | code contracts | test results | Add HealthFC and Recall@K evaluation |

## Counts

- 25 candidate videos
- 4 downloaded
- 21 failed downloads
- 3 transcripts generated
- 1 video skipped because it has no audio
- 0 extracted claim files
- 0 evidence files
- 0 final verdict files
- 5 tests passing

## Current work

Organization and preprocessing are complete. The next development task is a
small, executable claim-extraction path using the existing `Transcript` and
`AtomicClaim` models.
