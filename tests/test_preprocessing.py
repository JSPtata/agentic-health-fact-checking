from pathlib import Path

from health_pipeline.models import TranscriptSegment
from health_pipeline.preprocessing.pipeline import Preprocessor


class FakeExtractor:
    def extract(self, video_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return output_path


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        return [TranscriptSegment(text="Drink water to stay hydrated.", start=0, end=2)]


class FakeOCR:
    def recognize(self, video_path: Path) -> list[TranscriptSegment]:
        return [TranscriptSegment(text="Hydration matters", start=1, end=2)]


def test_preprocessor_unifies_asr_and_ocr(tmp_path: Path) -> None:
    transcript = Preprocessor(
        FakeExtractor(), FakeTranscriber(), FakeOCR()
    ).process(tmp_path / "video.mp4", tmp_path / "audio.wav")

    assert transcript.text == "Drink water to stay hydrated."
    assert transcript.timestamps[0].start == 0
    assert transcript.on_screen_text[0].text == "Hydration matters"
