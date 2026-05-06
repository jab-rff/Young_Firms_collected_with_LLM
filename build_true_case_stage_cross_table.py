"""Build stage-by-stage mention counts for manual-true firms.

This script focuses on the firms in `results_llm_prompting.xlsx` with
`human_validation == True` and counts how often they appear across the
available old Borsen article-pipeline datasets on the X: drive.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

from src.normalization import normalize_company_name

DEFAULT_RESULTS = Path("results_llm_prompting.xlsx")
DEFAULT_SHEET = "close_reading_cases"
DEFAULT_OUTPUT = Path("analysis/true_case_stage_cross_table.csv")
DEFAULT_SUMMARY = Path("analysis/true_case_stage_cross_table.md")

X_ROOT = Path(r"X:\Produktivitet\_1_Mapping_successful_firms - 7002\3_step2_data_scraping\borsen_articles")


@dataclass(frozen=True)
class StageSpec:
    slug: str
    label: str
    path: Path
    kind: str  # article_scan | row_match
    text_columns: tuple[str, ...] = ()
    match_columns: tuple[str, ...] = ()


STAGES = [
    StageSpec(
        slug="all_articles_817081",
        label="All scraped articles (817,081)",
        path=X_ROOT / "3_articles_scraped_full_text" / "all_articles_817081_ready_for_extraction.parquet",
        kind="article_scan",
        text_columns=("title", "text_into_model"),
    ),
    StageSpec(
        slug="schema2_articles_24401",
        label="Schema 2 filtered articles (24,401)",
        path=X_ROOT / "4_articles_information_extracted_full_text" / "1_df_moved" / "schema_2_moved_abroad_broad" / "2_filtered" / "df_moved_s2_24401_unique.parquet",
        kind="article_scan",
        text_columns=("title", "text_into_model"),
    ),
    StageSpec(
        slug="schema3_rows_27747",
        label="Schema 3 filtered firm rows (27,747 rows; 8,720 firms)",
        path=X_ROOT / "4_articles_information_extracted_full_text" / "1_df_moved" / "schema_3_moved_abroad_detail" / "2_filtered" / "df_moved_s3_8720_unique_firms.parquet",
        kind="row_match",
        match_columns=("Firm", "Firm_3_new"),
    ),
    StageSpec(
        slug="final_rows_18719",
        label="Final pre-GPT firm rows (18,719 rows; 6,022 firms)",
        path=X_ROOT / "4_articles_information_extracted_full_text" / "1_df_moved" / "df_moved_final_data" / "df_moved_s3_18719_rows_13879_links_6022_firms.parquet",
        kind="row_match",
        match_columns=("Firm", "Firm_3_new"),
    ),
    StageSpec(
        slug="gpt_rows_2842",
        label="GPT-classified rows (2,842 rows; 1,279 firms incl. pop cases)",
        path=X_ROOT / "5_classification_w_GPT" / "2_filtered_on_class_scores" / "df_moved_classified_filtered_2185_articles_1279_firms_incl_pop_cases.parquet",
        kind="row_match",
        match_columns=("Firm", "Firm_3_new"),
    ),
    StageSpec(
        slug="close_reading_rows_2842",
        label="Close-reading working table (2,842 rows)",
        path=X_ROOT / "11_close_reading" / "df_main.parquet",
        kind="row_match",
        match_columns=("firm", "firm_clean_standardized"),
    ),
    StageSpec(
        slug="close_reading_final_109",
        label="Close-reading final output (109 rows)",
        path=X_ROOT / "11_close_reading" / "df_close_reading_output.csv",
        kind="row_match",
        match_columns=("firm_annotation",),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage counts for human-validation true firms.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df = pd.read_excel(args.input, sheet_name=args.sheet)
    true_df = results_df[results_df["human_validation"].astype(str) == "True"].copy()

    firm_specs = [build_firm_spec(row) for row in true_df.to_dict(orient="records")]
    stage_counts = {stage.slug: count_stage(stage, firm_specs) for stage in STAGES}

    output_rows = []
    for spec in firm_specs:
        row = {
            "firm": spec["firm"],
            "aliases": " | ".join(spec["aliases_display"]),
        }
        for stage in STAGES:
            row[stage.slug] = stage_counts[stage.slug].get(spec["firm"], 0)
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_rows, args.output)
    write_summary(output_rows, args.summary_output)

    print(f"true_firms={len(output_rows)}")
    print(f"output_path={args.output}")
    print(f"summary_output_path={args.summary_output}")


def build_firm_spec(row: dict[str, Any]) -> dict[str, Any]:
    firm = str(row.get("firm") or "").strip()
    aliases = extract_aliases(firm)
    for extra_name in (row.get("name_first"), row.get("name_today")):
        extra_text = str(extra_name or "").strip()
        if extra_text:
            aliases.update(extract_aliases(extra_text))
    patterns = build_text_patterns(aliases)
    normalized_aliases = {normalize_company_name(alias) for alias in aliases}
    normalized_aliases.discard("")
    return {
        "firm": firm,
        "aliases_display": sorted(aliases),
        "text_patterns": patterns,
        "normalized_aliases": normalized_aliases,
    }


def extract_aliases(name: str) -> set[str]:
    raw = str(name or "").strip()
    if not raw:
        return set()
    candidates = {raw}
    candidates.update(split_parts(raw))
    for paren_text in re.findall(r"\(([^()]*)\)", raw):
        candidates.add(paren_text.strip())
        candidates.update(split_parts(paren_text))

    aliases: set[str] = set()
    for candidate in candidates:
        cleaned = re.sub(r"\bformerly\b", "", candidate, flags=re.IGNORECASE).strip(" -,:")
        if cleaned:
            aliases.add(cleaned)
            normalized = normalize_company_name(cleaned)
            if normalized and normalized != cleaned.casefold():
                aliases.add(normalized)
    return {alias for alias in aliases if is_reasonable_alias(alias)}


def split_parts(text: str) -> list[str]:
    chunks = [text]
    for separator in [" → ", " -> ", " / ", ";", " | "]:
        next_chunks: list[str] = []
        for chunk in chunks:
            next_chunks.extend(chunk.split(separator))
        chunks = next_chunks
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def is_reasonable_alias(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) < 4:
        return False
    return any(char.isalpha() for char in text)


def build_text_patterns(aliases: Iterable[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for alias in sorted(set(aliases), key=lambda value: (-len(value), value)):
        escaped = re.escape(alias)
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
        patterns.append(pattern)
    return patterns


def count_stage(stage: StageSpec, firm_specs: list[dict[str, Any]]) -> dict[str, int]:
    if stage.kind == "article_scan":
        return count_article_stage(stage, firm_specs)
    return count_row_stage(stage, firm_specs)


def count_article_stage(stage: StageSpec, firm_specs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {spec["firm"]: 0 for spec in firm_specs}
    parquet = pq.ParquetFile(stage.path)
    columns = list(stage.text_columns)
    for batch in parquet.iter_batches(columns=columns):
        frame = batch.to_pandas()
        texts = (
            frame[columns[0]].fillna("").astype(str)
            if len(columns) == 1
            else frame[list(columns)].fillna("").astype(str).agg(" ".join, axis=1)
        )
        for spec in firm_specs:
            matched = texts.map(lambda text: any(pattern.search(text) for pattern in spec["text_patterns"]))
            counts[spec["firm"]] += int(matched.sum())
    return counts


def count_row_stage(stage: StageSpec, firm_specs: list[dict[str, Any]]) -> dict[str, int]:
    try:
        df = pd.read_parquet(stage.path, columns=list(stage.match_columns))
    except Exception:
        df = pd.read_csv(stage.path, usecols=list(stage.match_columns))

    normalized_rows: set[str] = set()
    for column in stage.match_columns:
        if column not in df.columns:
            continue
        series = df[column].fillna("").astype(str).map(normalize_company_name)
        normalized_rows.add(column)
        df[f"__norm_{column}"] = series

    norm_columns = [f"__norm_{column}" for column in stage.match_columns if f"__norm_{column}" in df.columns]
    counts = {spec["firm"]: 0 for spec in firm_specs}
    for spec in firm_specs:
        mask = False
        for column in norm_columns:
            column_mask = df[column].isin(spec["normalized_aliases"])
            mask = column_mask if isinstance(mask, bool) else (mask | column_mask)
        counts[spec["firm"]] = int(mask.sum()) if not isinstance(mask, bool) else 0
    return counts


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_columns = [stage.slug for stage in STAGES]
    lines = [
        "# True Case Stage Cross Table",
        "",
        f"- Firms covered: {len(rows)}",
        "",
        "## Stage Columns",
    ]
    lines.extend(f"- `{stage.slug}`: {stage.label}" for stage in STAGES)
    lines.extend(["", "## Firms With Zero Counts In All Old Article Stages"])
    zero_rows = [
        row["firm"]
        for row in rows
        if sum(int(row[column]) for column in stage_columns[:-1]) == 0
    ]
    if zero_rows:
        lines.extend(f"- {firm}" for firm in zero_rows)
    else:
        lines.append("- None")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
