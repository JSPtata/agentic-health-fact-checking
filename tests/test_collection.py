import csv
from pathlib import Path

from scripts.collect_videos import load_rows, save_rows


def test_manifest_round_trip(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,domain,url,status,notes\n"
        "x,nutrition/diet,,pending,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.collect_videos.MANIFEST", manifest)
    rows = load_rows()
    rows[0]["status"] = "search_error"
    save_rows(rows)
    with manifest.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["status"] == "search_error"
