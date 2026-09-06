"""Process complete downloaded videos into inspectable transcript JSON files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from health_pipeline.preprocessing.pipeline import (
    FasterWhisperTranscriber,
    FFmpegAudioExtractor,
)
from health_pipeline.models import Transcript


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    matches = list(
        (
            Path.home()
            / "AppData/Local/Microsoft/WinGet/Packages"
        ).glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe")
    )
    if matches:
        return str(matches[0])
    raise RuntimeError("ffmpeg was not found on PATH or in the WinGet package directory.")


def has_audio_stream(video_path: Path, ffmpeg_path: str) -> bool:
    probe = Path(ffmpeg_path).with_name("ffprobe.exe")
    if not probe.is_file():
        raise RuntimeError(f"ffprobe was not found next to ffmpeg: {probe}")
    result = subprocess.run(
        [str(probe), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path.name}: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def process_video(
    video_path: Path,
    output_dir: Path,
    audio_dir: Path,
    transcriber: FasterWhisperTranscriber,
    ffmpeg_path: str,
) -> Path:
    audio_path = audio_dir / f"{video_path.stem}.wav"
    transcript_path = output_dir / f"{video_path.stem}.transcript.json"
    audio = FFmpegAudioExtractor(ffmpeg_path).extract(video_path, audio_path)
    segments = transcriber.transcribe(audio)
    transcript = Transcript(
        text=" ".join(segment.text for segment in segments),
        timestamps=segments,
        on_screen_text=[],
    )
    transcript_path.write_text(
        json.dumps(transcript.model_dump(), indent=2),
        encoding="utf-8",
    )
    return transcript_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=Path("data/raw/videos"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/transcripts"))
    parser.add_argument("--audio-output", type=Path, default=Path("data/processed/audio"))
    parser.add_argument("--model", default="tiny")
    args = parser.parse_args()

    videos = sorted(
        path
        for path in args.videos.iterdir()
        if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
        and not path.name.endswith(".part")
    )
    if not videos:
        raise SystemExit(f"No complete video files found in {args.videos}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.audio_output.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = find_ffmpeg()
    try:
        transcriber = FasterWhisperTranscriber(args.model)
    except RuntimeError as exc:
        raise SystemExit(
            f"{exc}\nInstall with: python -m pip install -e \".[preprocessing]\""
        ) from exc
    for video in videos:
        print(f"Processing {video.name}")
        if not has_audio_stream(video, ffmpeg_path):
            print(f"SKIPPED {video.name}: no audio stream")
            continue
        try:
            print(
                process_video(
                    video, args.output, args.audio_output, transcriber, ffmpeg_path
                )
            )
        except (OSError, RuntimeError) as exc:
            print(f"FAILED {video.name}: {exc}")


if __name__ == "__main__":
    main()
