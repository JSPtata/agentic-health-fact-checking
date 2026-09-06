"""Composable ffmpeg, ASR, and OCR preprocessing adapters."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol, Sequence

from ..models import Transcript, TranscriptSegment


class AudioExtractor(Protocol):
    def extract(self, video_path: Path, output_path: Path) -> Path: ...


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]: ...


class FrameTextRecognizer(Protocol):
    def recognize(self, video_path: Path) -> list[TranscriptSegment]: ...


class VideoPreprocessor(Protocol):
    def process(self, video_path: Path) -> Transcript: ...


class FFmpegAudioExtractor:
    """Extract mono 16 kHz PCM audio using the installed ffmpeg executable."""

    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        self.ffmpeg_binary = ffmpeg_binary

    def extract(self, video_path: Path, output_path: Path) -> Path:
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        return output_path


class FasterWhisperTranscriber:
    """Lazy faster-whisper adapter; model loading is deferred until construction."""

    def __init__(self, model_name: str = "small", **model_kwargs: object) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'preprocessing' extra to use faster-whisper."
            ) from exc
        self._model = WhisperModel(model_name, **model_kwargs)

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        segments, _ = self._model.transcribe(str(audio_path))
        return [
            TranscriptSegment(
                text=segment.text.strip(),
                start=float(segment.start),
                end=float(segment.end),
            )
            for segment in segments
            if segment.text.strip()
        ]


class EasyOCRFrameTextRecognizer:
    """OCR adapter. Frame extraction is intentionally injected for testability."""

    def __init__(self, reader: object | None = None) -> None:
        if reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "Install the 'preprocessing' extra to use EasyOCR."
                ) from exc
            reader = easyocr.Reader(["en"])
        self._reader = reader

    def recognize(self, video_path: Path) -> list[TranscriptSegment]:
        raise NotImplementedError(
            "Provide keyframes to the OCR reader or implement frame extraction "
            "for the target video backend."
        )


class Preprocessor:
    """Coordinates independent audio, ASR, and OCR implementations."""

    def __init__(
        self,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        ocr: FrameTextRecognizer,
    ) -> None:
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.ocr = ocr

    def process(self, video_path: Path, audio_path: Path) -> Transcript:
        extracted_audio = self.audio_extractor.extract(video_path, audio_path)
        spoken_segments = self.transcriber.transcribe(extracted_audio)
        on_screen_segments = self.ocr.recognize(video_path)
        return Transcript(
            text=" ".join(segment.text for segment in spoken_segments),
            timestamps=spoken_segments,
            on_screen_text=on_screen_segments,
        )
