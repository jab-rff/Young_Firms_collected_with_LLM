"""Prepare Model-2-ready candidate JSONL from the separate Diffbot founder list."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.diffbot_founder_pipeline import (
    DEFAULT_DIFFBOT_INPUT_PATH,
    DEFAULT_PIPELINE_SLUG,
    build_diffbot_candidate,
    build_diffbot_paths,
    load_diffbot_rows,
)
from src.io import save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare separate Diffbot founder candidates for Model 2.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DIFFBOT_INPUT_PATH,
        help="Path to diffbot_dk_founder_after_1999.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=build_diffbot_paths(DEFAULT_PIPELINE_SLUG).candidates,
        help="Path to write candidate JSONL output.",
    )
    args = parser.parse_args()

    rows = load_diffbot_rows(args.input)
    candidates = [build_diffbot_candidate(row) for row in rows if str(row.get("name") or "").strip()]
    save_jsonl(candidates, args.output)

    print(f"candidate_rows={len(candidates)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
