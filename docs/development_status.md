# Development status

## Built

- Pydantic contracts for transcripts, claims, evidence, and verdicts
- Resumable yt-dlp collection
- ffmpeg audio extraction
- Audio-stream detection
- faster-whisper transcription
- Typed transcript JSON storage
- Unit tests for current adapters and scripts

## Tested

The current suite has five passing tests. Three downloaded audio-bearing videos
produce transcripts; one video-only file is skipped.

## Blocked

Twenty-one candidate downloads failed because of unavailable videos or network
issues. The current implementation does not retry indefinitely. Claim files do
not exist because no concrete LLM client has been selected.

## Decisions

- Keep stage boundaries injectable.
- Store original media/metadata separately from processed audio/transcripts.
- Use the small `tiny` Whisper model for the initial local pass.
- Do not invent evidence or verdict outputs before claim extraction is working.

## Next

Implement the smallest concrete claim-extraction runner using the existing
structured schema, then test it against the three generated transcripts.
