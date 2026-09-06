"""Small orchestration surface for the currently implemented stages."""

from __future__ import annotations

from pathlib import Path

from .claim_extraction import ClaimExtractor
from .models import ClaimExtractionResult, Transcript
from .preprocessing import VideoPreprocessor


class Pipeline:
    """Runs enabled stages; future stages can be injected without rewiring callers."""

    def __init__(
        self,
        preprocessor: VideoPreprocessor | None = None,
        claim_extractor: ClaimExtractor | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.claim_extractor = claim_extractor

    def run(
        self,
        *,
        video_path: Path | None = None,
        transcript: Transcript | None = None,
        audio_path: Path | None = None,
    ) -> ClaimExtractionResult:
        if transcript is None:
            if self.preprocessor is None or video_path is None or audio_path is None:
                raise ValueError(
                    "Provide transcript, or configure preprocessor with video_path "
                    "and audio_path."
                )
            transcript = self.preprocessor.process(video_path, audio_path)
        if self.claim_extractor is None:
            raise ValueError("A claim_extractor is required to run claim extraction.")
        return self.claim_extractor.extract(transcript)
