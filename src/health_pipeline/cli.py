"""Command-line entry point for inspecting the initial pipeline stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import Transcript


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--transcript", type=Path)
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("Only --dry-run is available in this initial scaffold.")
    transcript = (
        Transcript.model_validate_json(args.transcript.read_text(encoding="utf-8"))
        if args.transcript
        else Transcript(
            text="Vitamin C prevents every cold.",
            timestamps=[],
            on_screen_text=[],
        )
    )
    print(json.dumps({"stage": "preprocessing", "output": transcript.model_dump()}, indent=2))
    print(json.dumps({"stage": "claim_extraction", "output": {"claims": []}}, indent=2))


if __name__ == "__main__":
    main()
