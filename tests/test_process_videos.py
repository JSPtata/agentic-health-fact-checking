from pathlib import Path

from scripts.process_videos import find_ffmpeg


def test_find_ffmpeg_returns_existing_path(monkeypatch, tmp_path: Path) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg.exe"
    fake_ffmpeg.write_bytes(b"")
    monkeypatch.setattr("scripts.process_videos.shutil.which", lambda _: str(fake_ffmpeg))
    assert find_ffmpeg() == str(fake_ffmpeg)
