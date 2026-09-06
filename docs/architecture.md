# Architecture

```text
collection
    ↓
preprocessing
    ↓
transcription
    ↓
claim extraction
    ↓
evidence retrieval
    ↓
verification
    ↓
verdict
```

| Stage | Purpose | Status | Input → output | Source |
|---|---|---|---|---|
| Collection | Find and download candidate videos | Partial | YouTube → raw videos/metadata | `scripts/collect_videos.py` |
| Preprocessing | Extract audio and detect audio streams | Complete | video → WAV | `src/health_pipeline/preprocessing/pipeline.py` |
| Transcription | Convert speech to timed text | Complete | WAV → `Transcript` JSON | `src/health_pipeline/preprocessing/pipeline.py`, `scripts/process_videos.py` |
| Claim extraction | Identify atomic factual claims | Interface only | `Transcript` → `AtomicClaim[]` | `src/health_pipeline/claim_extraction/agent.py` |
| Evidence retrieval | Find reliable supporting passages | Not implemented | claims → evidence | none |
| Verification | Compare claims with evidence | Not implemented | claim + evidence → verdict | none |
| Verdict | Store final classification | Schema only | verification → `Verdict` | `src/health_pipeline/models.py` |

The typed contracts are centralized in `models.py`. The top-level
`Pipeline` currently wires preprocessing and claim extraction dependencies,
but no concrete LLM client is supplied.
