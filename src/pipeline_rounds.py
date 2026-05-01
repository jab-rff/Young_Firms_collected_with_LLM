"""Shared helpers for round-based pipeline orchestration and status."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STOP_AFTER_CHOICES = ("discovery", "dedup", "model1", "model2", "model3", "export")
ORIGIN_TRACK_CHOICES = ("in_denmark", "abroad_danish_founders")


@dataclass(frozen=True)
class RoundPaths:
    round_number: int
    origin_track: str
    discovery: Path
    dedup: Path
    model1: Path
    model2: Path
    model3: Path
    review: Path
    manifest: Path


def build_round_paths(round_number: int, origin_track: str = "in_denmark") -> RoundPaths:
    if origin_track not in ORIGIN_TRACK_CHOICES:
        raise ValueError(f"Unsupported origin track: {origin_track}")
    round_token = f"snowball_round_{round_number:03d}"
    if origin_track != "in_denmark":
        round_token = f"{round_token}_{origin_track}"
    return RoundPaths(
        round_number=round_number,
        origin_track=origin_track,
        discovery=Path("data/discovery") / f"{round_token}.jsonl",
        dedup=Path("data/discovery") / f"{round_token}_deduped.jsonl",
        model1=Path("data/model1") / f"{round_token}_candidates.jsonl",
        model2=Path("data/model2") / f"{round_token}_enriched.jsonl",
        model3=Path("data/model3") / f"{round_token}_validated.jsonl",
        review=Path("data/review") / f"{round_token}_review.csv",
        manifest=Path("data/runs") / f"{round_token}_manifest.json",
    )


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _count_jsonl_rows(path)
    if suffix == ".csv":
        return _count_csv_rows(path)
    raise ValueError(f"Unsupported row-count format: {path}")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for row in reader if any(cell.strip() for cell in row))
