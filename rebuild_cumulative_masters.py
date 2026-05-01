"""Rebuild cumulative validated/review masters from saved round-level Model 3 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_final_review import export_final_review
from model_3_validation import (
    DEFAULT_MASTER_REVIEW_PATH,
    DEFAULT_MASTER_VALIDATED_PATH,
    DEFAULT_TRACK_MASTER_REVIEW_PATHS,
    DEFAULT_TRACK_MASTER_VALIDATED_PATHS,
)
from src.io import save_jsonl
from src.normalization import normalize_company_name


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def find_round_validated_files(model3_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(model3_dir.glob("snowball_round_*_validated.jsonl")):
        if path.name.startswith("model3_validated_master"):
            continue
        files.append(path)
    return files


def rebuild_cumulative_masters(
    *,
    model3_dir: Path = Path("data/model3"),
    all_tracks_validated_path: Path = DEFAULT_MASTER_VALIDATED_PATH,
    all_tracks_review_path: Path = DEFAULT_MASTER_REVIEW_PATH,
    track_validated_paths: dict[str, Path] | None = None,
    track_review_paths: dict[str, Path] | None = None,
) -> dict[str, int]:
    track_validated_paths = track_validated_paths or DEFAULT_TRACK_MASTER_VALIDATED_PATHS
    track_review_paths = track_review_paths or DEFAULT_TRACK_MASTER_REVIEW_PATHS

    merged: dict[str, dict[str, Any]] = {}
    source_files = find_round_validated_files(model3_dir)
    for path in source_files:
        for record in load_jsonl(path):
            normalized = normalize_company_name(record.get("firm_name"))
            if not normalized:
                continue
            merged[normalized] = record

    merged_records = sorted(merged.values(), key=lambda row: str(row.get("firm_name") or "").casefold())
    save_jsonl(merged_records, all_tracks_validated_path)
    export_final_review(merged_records, all_tracks_review_path)

    counts: dict[str, int] = {"all_tracks": len(merged_records)}
    for origin_track, validated_path in track_validated_paths.items():
        filtered = [
            record
            for record in merged_records
            if str(record.get("origin_track") or "in_denmark") == origin_track
        ]
        save_jsonl(filtered, validated_path)
        review_path = track_review_paths.get(origin_track)
        if review_path is not None:
            export_final_review(filtered, review_path)
        counts[origin_track] = len(filtered)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild cumulative master files from saved Model 3 round outputs.")
    parser.add_argument("--model3-dir", type=Path, default=Path("data/model3"), help="Directory containing round-level validated JSONL files.")
    args = parser.parse_args()

    counts = rebuild_cumulative_masters(model3_dir=args.model3_dir)
    print(f"all_tracks_rows={counts['all_tracks']}")
    print(f"in_denmark_rows={counts.get('in_denmark', 0)}")
    print(f"abroad_danish_founders_rows={counts.get('abroad_danish_founders', 0)}")
    print(f"all_tracks_validated_path={DEFAULT_MASTER_VALIDATED_PATH}")
    print(f"all_tracks_review_path={DEFAULT_MASTER_REVIEW_PATH}")
    print(f"in_denmark_validated_path={DEFAULT_TRACK_MASTER_VALIDATED_PATHS['in_denmark']}")
    print(f"in_denmark_review_path={DEFAULT_TRACK_MASTER_REVIEW_PATHS['in_denmark']}")
    print(f"abroad_validated_path={DEFAULT_TRACK_MASTER_VALIDATED_PATHS['abroad_danish_founders']}")
    print(f"abroad_review_path={DEFAULT_TRACK_MASTER_REVIEW_PATHS['abroad_danish_founders']}")


if __name__ == "__main__":
    main()
