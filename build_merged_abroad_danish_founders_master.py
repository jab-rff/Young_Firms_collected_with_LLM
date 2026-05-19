"""Merge the LLM-reviewed and Diffbot founder lists into one unioned CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from build_unique_abroad_danish_founders import extract_name_aliases, normalize_firm_key


DEFAULT_LLM_INPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_unique_firms.csv")
DEFAULT_DIFFBOT_INPUT = Path("diffbot_dk_founder_after_1999.csv")
DEFAULT_PRELIMINARY_INPUT = Path("preliminary_data_29_04.csv")
DEFAULT_OUTPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_merged_master.csv")
BORSEN_FLAG_COLUMN = "found_w_b\u00f8rsen"

ESSENTIAL_COLUMNS = [
    "name",
    "first_legal_name",
    "founding_year",
    "founding_city",
    "founding_country_iso",
    "founded_in_denmark",
    "danish_founders_abroad",
    "moved_hq_abroad",
    "move_year",
    "moved_to_city",
    "moved_to_country_iso",
    "hq_today_city",
    "hq_today_country_iso",
    "status_today",
    "validation_label",
    "confidence",
    "found_in_diffbot",
    BORSEN_FLAG_COLUMN,
    "source_list",
    "needs_human_review",
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

HELPER_COLUMNS = [
    "origin_track",
    "unique_firm_key",
    "source_row_count",
    "source_firm_names_merged",
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
    "diffbot_name",
    "diffbot_founders_name",
    "diffbot_company_linkedin",
    "diffbot_homepage",
    "diffbot_location_city",
    "diffbot_location_region",
    "diffbot_location_country",
    "diffbot_summary",
    "diffbot_categories",
    "diffbot_ceo_name",
    "diffbot_nb_employees_max",
    "diffbot_founders_target_diffbot_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the unique LLM and Diffbot founder lists into one CSV.")
    parser.add_argument("--llm-input", type=Path, default=DEFAULT_LLM_INPUT)
    parser.add_argument("--diffbot-input", type=Path, default=DEFAULT_DIFFBOT_INPUT)
    parser.add_argument("--preliminary-input", type=Path, default=DEFAULT_PRELIMINARY_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_rows = pd.read_csv(args.llm_input).where(pd.notna, "").to_dict(orient="records")
    diffbot_rows = pd.read_csv(args.diffbot_input).where(pd.notna, "").to_dict(orient="records")
    preliminary_keys = load_preliminary_keys(args.preliminary_input)
    merged = merge_lists(llm_rows, diffbot_rows, preliminary_keys=preliminary_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(merged, args.output)
    print(f"llm_rows={len(llm_rows)}")
    print(f"diffbot_rows={len(diffbot_rows)}")
    print(f"merged_rows={len(merged)}")
    print(f"output_path={args.output}")


def merge_lists(
    llm_rows: list[dict[str, Any]],
    diffbot_rows: list[dict[str, Any]],
    preliminary_keys: set[str],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    llm_alias_map: dict[str, str] = {}

    for row in llm_rows:
        canonical_key = normalize_firm_key(str(row.get("unique_firm_key") or row.get("firm_name") or ""))
        output_row = build_llm_output_row(row)
        output_row[BORSEN_FLAG_COLUMN] = "true" if row_matches_preliminary(output_row, preliminary_keys) else "false"
        merged[canonical_key] = output_row
        for alias in row_aliases(row):
            llm_alias_map[alias] = canonical_key

    for diffbot_row in diffbot_rows:
        diffbot_name = str(diffbot_row.get("name") or "").strip()
        if not diffbot_name:
            continue
        diffbot_key = normalize_firm_key(diffbot_name)
        canonical_key = llm_alias_map.get(diffbot_key, diffbot_key)
        if canonical_key in merged:
            enrich_with_diffbot(merged[canonical_key], diffbot_row)
        else:
            output_row = build_diffbot_only_output_row(diffbot_row)
            output_row[BORSEN_FLAG_COLUMN] = "true" if row_matches_preliminary(output_row, preliminary_keys) else "false"
            merged[canonical_key] = output_row

    rows = sorted(merged.values(), key=lambda row: str(row.get("name") or "").casefold())
    return rows


def row_aliases(row: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for field in ("unique_firm_key", "firm_name", "source_firm_names_merged", "first_legal_entity_name"):
        aliases.update(extract_name_aliases(str(row.get(field) or "")))
    return {alias for alias in aliases if alias}


def build_llm_output_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        "name": first_non_empty(row.get("firm_name"), row.get("source_firm_names_merged"), row.get("unique_firm_key")),
        "first_legal_name": str(row.get("first_legal_entity_name") or ""),
        "founding_year": row.get("founding_year", ""),
        "founding_city": str(row.get("founding_city") or ""),
        "founding_country_iso": str(row.get("founding_country_iso") or ""),
        "founded_in_denmark": str(row.get("founded_in_denmark") or ""),
        "danish_founders_abroad": str(row.get("danish_founders_abroad") or ""),
        "moved_hq_abroad": str(row.get("moved_hq_abroad") or ""),
        "move_year": row.get("move_year", ""),
        "moved_to_city": str(row.get("moved_to_city") or ""),
        "moved_to_country_iso": str(row.get("moved_to_country_iso") or ""),
        "hq_today_city": str(row.get("hq_today_city") or ""),
        "hq_today_country_iso": str(row.get("hq_today_country_iso") or ""),
        "status_today": str(row.get("status_today") or ""),
        "validation_label": str(row.get("validation_label") or ""),
        "confidence": str(row.get("confidence") or ""),
        "found_in_diffbot": str(row.get("found_in_diffbot") or "false"),
        BORSEN_FLAG_COLUMN: "",
        "source_list": "llm",
        "needs_human_review": str(row.get("needs_human_review") or ""),
        "dk_founder_1": str(row.get("dk_founder_1") or ""),
        "dk_founder_1_linkedin": str(row.get("dk_founder_1_linkedin") or ""),
        "dk_founder_2": str(row.get("dk_founder_2") or ""),
        "dk_founder_2_linkedin": str(row.get("dk_founder_2_linkedin") or ""),
        "dk_founder_3": str(row.get("dk_founder_3") or ""),
        "dk_founder_3_linkedin": str(row.get("dk_founder_3_linkedin") or ""),
        "dk_founder_4": str(row.get("dk_founder_4") or ""),
        "dk_founder_4_linkedin": str(row.get("dk_founder_4_linkedin") or ""),
        "dk_founder_5": str(row.get("dk_founder_5") or ""),
        "dk_founder_5_linkedin": str(row.get("dk_founder_5_linkedin") or ""),
        "origin_track": str(row.get("origin_track") or ""),
        "unique_firm_key": str(row.get("unique_firm_key") or ""),
        "source_row_count": row.get("source_row_count", ""),
        "source_firm_names_merged": str(row.get("source_firm_names_merged") or ""),
        "validation_reason": str(row.get("validation_reason") or ""),
        "exclusion_reason": str(row.get("exclusion_reason") or ""),
        "evidence_summary": str(row.get("evidence_summary") or ""),
        "founding_evidence": str(row.get("founding_evidence") or ""),
        "founder_danish_context": str(row.get("founder_danish_context") or ""),
        "relocation_evidence": str(row.get("relocation_evidence") or ""),
        "ma_evidence": str(row.get("ma_evidence") or ""),
        "relocation_context": str(row.get("relocation_context") or ""),
        "ma_context": str(row.get("ma_context") or ""),
        "uncertainty_note": str(row.get("uncertainty_note") or ""),
        "sources_founding": str(row.get("sources_founding") or ""),
        "sources_founder_identity": str(row.get("sources_founder_identity") or ""),
        "sources_relocation": str(row.get("sources_relocation") or ""),
        "sources_ma": str(row.get("sources_ma") or ""),
        "sources_status_today": str(row.get("sources_status_today") or ""),
        "diffbot_name": "",
        "diffbot_founders_name": "",
        "diffbot_company_linkedin": "",
        "diffbot_homepage": "",
        "diffbot_location_city": "",
        "diffbot_location_region": "",
        "diffbot_location_country": "",
        "diffbot_summary": "",
        "diffbot_categories": "",
        "diffbot_ceo_name": "",
        "diffbot_nb_employees_max": "",
        "diffbot_founders_target_diffbot_id": "",
    }
    return output


def build_diffbot_only_output_row(row: dict[str, Any]) -> dict[str, Any]:
    founder_names = split_csv_like(str(row.get("founders_name") or ""))
    output = {
        "name": str(row.get("name") or ""),
        "first_legal_name": "",
        "founding_year": "",
        "founding_city": "",
        "founding_country_iso": "",
        "founded_in_denmark": "",
        "danish_founders_abroad": "",
        "moved_hq_abroad": "",
        "move_year": "",
        "moved_to_city": "",
        "moved_to_country_iso": "",
        "hq_today_city": str(row.get("location_city_name") or ""),
        "hq_today_country_iso": "",
        "status_today": "",
        "validation_label": "",
        "confidence": "",
        "found_in_diffbot": "true",
        BORSEN_FLAG_COLUMN: "",
        "source_list": "diffbot",
        "needs_human_review": "",
        "dk_founder_1": founder_names[0] if len(founder_names) > 0 else "",
        "dk_founder_1_linkedin": "",
        "dk_founder_2": founder_names[1] if len(founder_names) > 1 else "",
        "dk_founder_2_linkedin": "",
        "dk_founder_3": founder_names[2] if len(founder_names) > 2 else "",
        "dk_founder_3_linkedin": "",
        "dk_founder_4": founder_names[3] if len(founder_names) > 3 else "",
        "dk_founder_4_linkedin": "",
        "dk_founder_5": founder_names[4] if len(founder_names) > 4 else "",
        "dk_founder_5_linkedin": "",
        "origin_track": "abroad_danish_founders",
        "unique_firm_key": str(row.get("name") or ""),
        "source_row_count": 1,
        "source_firm_names_merged": str(row.get("name") or ""),
        "validation_reason": "",
        "exclusion_reason": "",
        "evidence_summary": "",
        "founding_evidence": "",
        "founder_danish_context": "",
        "relocation_evidence": "",
        "ma_evidence": "",
        "relocation_context": "",
        "ma_context": "",
        "uncertainty_note": "",
        "sources_founding": "",
        "sources_founder_identity": "",
        "sources_relocation": "",
        "sources_ma": "",
        "sources_status_today": "",
        "diffbot_name": str(row.get("name") or ""),
        "diffbot_founders_name": str(row.get("founders_name") or ""),
        "diffbot_company_linkedin": str(row.get("linkedInUri") or ""),
        "diffbot_homepage": str(row.get("homepageUri") or ""),
        "diffbot_location_city": str(row.get("location_city_name") or ""),
        "diffbot_location_region": str(row.get("location_region_name") or ""),
        "diffbot_location_country": str(row.get("location_country_name") or ""),
        "diffbot_summary": str(row.get("summary") or ""),
        "diffbot_categories": str(row.get("categories_name") or ""),
        "diffbot_ceo_name": str(row.get("ceo_name") or ""),
        "diffbot_nb_employees_max": row.get("nbEmployeesMax", ""),
        "diffbot_founders_target_diffbot_id": str(row.get("founders_targetDiffbotId") or ""),
    }
    return output


def enrich_with_diffbot(row: dict[str, Any], diffbot_row: dict[str, Any]) -> None:
    row["found_in_diffbot"] = "true"
    row["source_list"] = "both" if row.get("source_list") == "llm" else row.get("source_list") or "diffbot"
    row["diffbot_name"] = first_non_empty(row.get("diffbot_name"), diffbot_row.get("name"))
    row["diffbot_founders_name"] = first_non_empty(row.get("diffbot_founders_name"), diffbot_row.get("founders_name"))
    row["diffbot_company_linkedin"] = first_non_empty(row.get("diffbot_company_linkedin"), diffbot_row.get("linkedInUri"))
    row["diffbot_homepage"] = first_non_empty(row.get("diffbot_homepage"), diffbot_row.get("homepageUri"))
    row["diffbot_location_city"] = first_non_empty(row.get("diffbot_location_city"), diffbot_row.get("location_city_name"))
    row["diffbot_location_region"] = first_non_empty(row.get("diffbot_location_region"), diffbot_row.get("location_region_name"))
    row["diffbot_location_country"] = first_non_empty(row.get("diffbot_location_country"), diffbot_row.get("location_country_name"))
    row["diffbot_summary"] = first_non_empty(row.get("diffbot_summary"), diffbot_row.get("summary"))
    row["diffbot_categories"] = first_non_empty(row.get("diffbot_categories"), diffbot_row.get("categories_name"))
    row["diffbot_ceo_name"] = first_non_empty(row.get("diffbot_ceo_name"), diffbot_row.get("ceo_name"))
    row["diffbot_nb_employees_max"] = first_non_empty(row.get("diffbot_nb_employees_max"), diffbot_row.get("nbEmployeesMax"))
    row["diffbot_founders_target_diffbot_id"] = first_non_empty(
        row.get("diffbot_founders_target_diffbot_id"),
        diffbot_row.get("founders_targetDiffbotId"),
    )


def load_preliminary_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path).where(pd.notna, "")
    keys: set[str] = set()
    for name in df.get("name", []):
        normalized = normalize_firm_key(str(name or ""))
        if normalized:
            keys.add(normalized)
    return keys


def row_matches_preliminary(row: dict[str, Any], preliminary_keys: set[str]) -> bool:
    aliases: set[str] = set()
    for field in ("name", "first_legal_name", "unique_firm_key", "source_firm_names_merged"):
        aliases.update(extract_name_aliases(str(row.get(field) or "")))
    return any(alias in preliminary_keys for alias in aliases if alias)


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def split_csv_like(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [*ESSENTIAL_COLUMNS, *HELPER_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
