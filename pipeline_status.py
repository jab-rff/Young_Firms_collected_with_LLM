"""Report output-file status for a specific pipeline round."""

from __future__ import annotations

import argparse
from typing import Any

from src.pipeline_rounds import ORIGIN_TRACK_CHOICES, build_round_paths, count_rows


def build_status_rows(round_number: int, origin_track: str = "in_denmark") -> list[dict[str, Any]]:
    paths = build_round_paths(round_number, origin_track=origin_track)
    rows: list[dict[str, Any]] = []
    for slug, label, path, counts_rows in (
        ("discovery", "Snowball discovery", paths.discovery, True),
        ("dedup", "Deduplicate", paths.dedup, True),
        ("model1", "Model 1 candidate extraction", paths.model1, True),
        ("model2", "Model 2 enrichment", paths.model2, True),
        ("model3", "Model 3 validation", paths.model3, True),
        ("export", "Export final review", paths.review, True),
        ("manifest", "Run manifest", paths.manifest, False),
    ):
        exists = path.exists()
        rows.append(
            {
                "stage": slug,
                "label": label,
                "path": str(path),
                "exists": exists,
                "row_count": count_rows(path) if exists and counts_rows else None,
            }
        )
    return rows


def print_status(round_number: int, origin_track: str = "in_denmark") -> None:
    print(f"round={round_number}")
    print(f"origin_track={origin_track}")
    for row in build_status_rows(round_number, origin_track=origin_track):
        if row["exists"]:
            if row["row_count"] is None:
                print(f"OK {row['stage']}: exists path={row['path']}")
            else:
                print(f"OK {row['stage']}: {row['row_count']} rows path={row['path']}")
        else:
            print(f"MISSING {row['stage']}: path={row['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report round-file status for the staged snowball pipeline.")
    parser.add_argument("--round", required=True, type=int, dest="round_number", help="Pipeline round number.")
    parser.add_argument(
        "--origin-track",
        choices=ORIGIN_TRACK_CHOICES,
        default="in_denmark",
        help="Origin track whose round files should be inspected.",
    )
    args = parser.parse_args()
    print_status(args.round_number, origin_track=args.origin_track)


if __name__ == "__main__":
    main()
