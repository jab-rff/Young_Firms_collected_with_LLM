"""Deduplicate snowball discovery candidates and remove already-known firms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.io import save_jsonl
from src.normalization import normalize_company_name
from src.seed_list import build_exclusion_list, load_seed_firms


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _merge_text_values(left: Any, right: Any) -> str | None:
    values: list[str] = []
    for value in (left, right):
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)
    if not values:
        return None
    return " || ".join(values)


def _merge_urls(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*left, *right]:
        url = str(value).strip()
        if url and url not in merged:
            merged.append(url)
    return merged


def deduplicate_candidates(records: list[dict[str, Any]], known_names: list[str]) -> list[dict[str, Any]]:
    known_normalized = {normalize_company_name(name) for name in known_names if normalize_company_name(name)}
    deduped: dict[str, dict[str, Any]] = {}

    for record in records:
        firm_name = str(record.get("firm_name") or "").strip()
        normalized_name = normalize_company_name(firm_name)
        if not normalized_name or normalized_name in known_normalized:
            continue

        if normalized_name not in deduped:
            merged = dict(record)
            merged["source_urls"] = _merge_urls([], list(record.get("source_urls") or []))
            merged["discovery_buckets"] = [record.get("discovery_bucket")] if record.get("discovery_bucket") else []
            merged["merged_record_count"] = 1
            deduped[normalized_name] = merged
            continue

        current = deduped[normalized_name]
        current["source_urls"] = _merge_urls(
            list(current.get("source_urls") or []),
            list(record.get("source_urls") or []),
        )
        for field in ("possible_abroad_location", "signal_type", "signal_strength", "short_reason", "origin_track"):
            current[field] = _merge_text_values(current.get(field), record.get(field))
        bucket = record.get("discovery_bucket")
        if bucket and bucket not in current["discovery_buckets"]:
            current["discovery_buckets"].append(bucket)
        current["merged_record_count"] = int(current.get("merged_record_count") or 1) + 1

    return list(deduped.values())


def run_deduplication(input_path: Path, known_path: Path, output_path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(input_path)
    known_names = build_exclusion_list(load_seed_firms(known_path))
    deduped = deduplicate_candidates(records, known_names)
    save_jsonl(deduped, output_path)
    save_deduplicated_csv(deduped, output_path.with_suffix(".csv"))
    return deduped


def save_deduplicated_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "firm_name",
        "origin_track",
        "discovery_bucket",
        "discovery_buckets",
        "sector_if_known",
        "possible_founding_location",
        "possible_founding_year",
        "possible_abroad_location",
        "possible_move_year",
        "signal_type",
        "signal_strength",
        "short_reason",
        "source_urls",
        "merged_record_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["discovery_buckets"] = " | ".join(str(value) for value in row.get("discovery_buckets") or [])
            row["source_urls"] = " | ".join(str(value) for value in row.get("source_urls") or [])
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deduplicate snowball discovery candidates.")
    parser.add_argument("--input", type=Path, required=True, help="Snowball discovery JSONL path.")
    parser.add_argument("--known", type=Path, required=True, help="Known seed CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Deduplicated JSONL path.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_deduplication(input_path=args.input, known_path=args.known, output_path=args.output)


if __name__ == "__main__":
    main()
