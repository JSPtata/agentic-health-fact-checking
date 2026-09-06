# Pipeline guide

The currently runnable path is:

```text
data/raw/videos/<video>
  → ffprobe audio check
  → ffmpeg mono 16 kHz WAV
  → faster-whisper tiny
  → data/processed/transcripts/<video>.transcript.json
```

Each transcript contains `text`, timed `timestamps`, and `on_screen_text`.
OCR is currently empty because keyframe extraction and recognition are not
implemented.

The later stages are documented design targets, not runnable behavior yet.
