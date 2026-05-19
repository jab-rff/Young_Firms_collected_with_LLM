"""Merge the reviewed LLM and Diffbot master CSVs into one row-level CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_LLM_INPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_unique_firms.csv")
DEFAULT_DIFFBOT_INPUT = Path("data/diffbot/diffbot_dk_founder_after_1999_master_review.csv")
DEFAULT_OUTPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_unique_firms_with_diffbot_review.csv")

PREFERRED_COLUMN_ORDER = [
    "source",
    "firm_name",
    "founding_year",
    "first_legal_entity_name",
    "origin_track",
    "validation_label",
    "needs_human_review",
    "founded_in_denmark",
    "danish_founders_abroad",
    "founding_city",
    "founding_country_iso",
    "moved_hq_abroad",
    "move_year",
    "moved_to_city",
    "moved_to_country_iso",
    "hq_today_city",
    "hq_today_country_iso",
    "status_today",
    "confidence",
    "unique_firm_key",
    "source_row_count",
    "source_firm_names_merged",
    "found_in_diffbot",
    "validation_reason",
    "exclusion_reason",
    "evidence_summary",
    "founding_evidence",
    "founder_danish_context",
    "relocation_evidence",
    "ma_evidence",
    "relocation_context",
    "ma_context",
    "uncertainty_note",
    "sources_founding",
    "sources_founder_identity",
    "sources_relocation",
    "sources_ma",
    "sources_status_today",
    "dk_founder_1",
    "dk_founder_1_linkedin",
    "dk_founder_2",
    "dk_founder_2_linkedin",
    "dk_founder_3",
    "dk_founder_3_linkedin",
    "dk_founder_4",
    "dk_founder_4_linkedin",
    "dk_founder_5",
    "dk_founder_5_linkedin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed LLM and Diffbot master CSVs.")
    parser.add_argument("--llm-input", type=Path, default=DEFAULT_LLM_INPUT)
    parser.add_argument("--diffbot-input", type=Path, default=DEFAULT_DIFFBOT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_rows = pd.read_csv(args.llm_input).where(pd.notna, "").to_dict(orient="records")
    diffbot_rows = pd.read_csv(args.diffbot_input).where(pd.notna, "").to_dict(orient="records")

    llm_rows = [with_source(row, "LLM prompting") for row in llm_rows]
    diffbot_rows = [with_source(row, "Diffbot") for row in diffbot_rows]

    fieldnames = build_fieldnames(llm_rows, diffbot_rows)
    output_rows = [fill_missing_columns(row, fieldnames) for row in [*llm_rows, *diffbot_rows]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_rows, fieldnames, args.output)

    print(f"llm_rows={len(llm_rows)}")
    print(f"diffbot_rows={len(diffbot_rows)}")
    print(f"merged_rows={len(output_rows)}")
    print(f"output_path={args.output}")


def with_source(row: dict[str, Any], source: str) -> dict[str, Any]:
    updated = dict(row)
    updated["source"] = source
    return updated


def build_fieldnames(llm_rows: list[dict[str, Any]], diffbot_rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for key in PREFERRED_COLUMN_ORDER:
        if key not in ordered:
            ordered.append(key)
    for rows in (llm_rows, diffbot_rows):
        for row in rows:
            for key in row.keys():
                if key not in ordered:
                    ordered.append(key)
    return ordered


def fill_missing_columns(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in fieldnames}


def write_csv(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
