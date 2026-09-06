"""Collect metadata or downloads for the balanced 25-video manifest.

Requires yt-dlp for network collection:
    python -m pip install yt-dlp
    python scripts/collect_videos.py --search
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
VIDEO_DIR = ROOT / "data" / "raw" / "videos"
FIELDS = ["video_id", "domain", "url", "status", "notes"]


def ffmpeg_location() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    winget_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    matches = list(winget_root.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
    return str(matches[0]) if matches else None


def load_rows() -> list[dict[str, str]]:
    if not MANIFEST.is_file():
        raise SystemExit(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Manifest is empty: {MANIFEST}")
    missing = set(FIELDS) - set(rows[0])
    if missing:
        raise SystemExit(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    return rows


def save_rows(rows: list[dict[str, str]]) -> None:
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def search_manifest() -> None:
    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit("Install yt-dlp first: python -m pip install yt-dlp") from exc

    rows = load_rows()
    seen_urls: set[str] = set()
    for row in rows:
        if row["url"] and row["url"] in seen_urls:
            row["url"] = ""
            row["status"] = "pending"
            row["notes"] = "Duplicate candidate cleared; searching for a replacement."
        elif row["url"]:
            seen_urls.add(row["url"])
    options: dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "ignoreerrors": True,
    }
    for row in rows:
        if row["url"]:
            continue
        query = f"ytsearch10:{row['domain']} health misinformation short"
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                result = ydl.extract_info(query, download=False)
        except yt_dlp.utils.DownloadError as exc:
            row["status"] = "search_error"
            row["notes"] = f"Search failed: {exc}".replace("\n", " ")[:500]
            print(f"{row['video_id']}: search failed; continuing")
            continue
        entries = (result or {}).get("entries") or []
        if entries:
            entry = None
            for candidate in entries:
                if not candidate:
                    continue
                candidate_url = candidate.get("webpage_url") or candidate.get("url", "")
                candidate_id = candidate.get("id", "")
                canonical_url = candidate_url or (
                    f"https://www.youtube.com/shorts/{candidate_id}"
                    if candidate_id
                    else ""
                )
                if canonical_url and canonical_url not in seen_urls:
                    entry = candidate
                    row["url"] = canonical_url
                    seen_urls.add(canonical_url)
                    break
            row["notes"] = "Candidate from yt-dlp search; human review required."
            row["status"] = "candidate" if row["url"] else "no_result"
        else:
            row["status"] = "no_result"
            row["notes"] = "No result returned; add a reviewed URL manually."
        save_rows(rows)
    save_rows(rows)


def download_manifest() -> None:
    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit("Install yt-dlp first: python -m pip install yt-dlp") from exc

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    for row in rows:
        if not row["url"]:
            print(f"Skipping {row['video_id']}: URL is empty")
            continue
        existing = list(VIDEO_DIR.glob(f"{row['video_id']}.*"))
        media = [
            path for path in existing
            if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
            and not path.name.endswith(".part")
        ]
        if media:
            row["status"] = "downloaded"
            row["notes"] = f"Already present: {media[0].name}"
            print(f"Skipping {row['video_id']}: already downloaded")
            continue
        output = VIDEO_DIR / f"{row['video_id']}.%(ext)s"
        try:
            with yt_dlp.YoutubeDL(
                {
                    "noplaylist": True,
                    "outtmpl": str(output),
                    "format": "bv*[height<=1080]+ba/b[height<=1080]",
                    "ffmpeg_location": ffmpeg_location(),
                    "writeinfojson": True,
                    "ignoreerrors": False,
                }
            ) as ydl:
                ydl.download([row["url"]])
        except yt_dlp.utils.DownloadError as exc:
            row["status"] = "download_error"
            row["notes"] = f"Download failed: {exc}".replace("\n", " ")[:500]
            print(f"{row['video_id']}: download failed; continuing")
            save_rows(rows)
            continue
        row["status"] = "downloaded"
        row["notes"] = "Downloaded; verify license and claim manually."
        save_rows(rows)
    save_rows(rows)


def main() -> None:
    global MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    MANIFEST = args.manifest.resolve()
    if not args.search and not args.download:
        parser.error("Choose --search, --download, or both.")
    if args.search:
        search_manifest()
    if args.download:
        download_manifest()


if __name__ == "__main__":
    main()
