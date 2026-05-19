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

from src.true_case_article_search import build_alias_search_index, match_firms_in_text
from src.normalization import normalize_company_name
from src.true_case_firm_list import get_firm_name, load_firm_rows

DEFAULT_RESULTS = Path("results_llm_prompting.xlsx")
DEFAULT_SHEET = "close_reading_cases"
DEFAULT_OUTPUT = Path("analysis/true_case_stage_cross_table.csv")
DEFAULT_SUMMARY = Path("analysis/true_case_stage_cross_table.md")

X_ROOT = Path(r"X:\Produktivitet\_1_Mapping_successful_firms - 7002\3_step2_data_scraping\borsen_articles")
BEFORE_STEP_ROOT = X_ROOT / "0_datasets_before_each_step"


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
        slug="df1_816000_articles_scraped",
        label="df1_816000_articles_scraped.parquet",
        path=BEFORE_STEP_ROOT / "df1_816000_articles_scraped.parquet",
        kind="article_scan",
        text_columns=("title", "text_into_model"),
    ),
    StageSpec(
        slug="df2_60000_move_schema_1",
        label="df2_60000_move_schema_1.parquet",
        path=BEFORE_STEP_ROOT / "df2_60000_move_schema_1.parquet",
        kind="row_match",
        match_columns=("Firm", "Firm_3_new"),
    ),
    StageSpec(
        slug="df3_10000_move_schema_2",
        label="df3_10000_move_schema_2.parquet",
        path=BEFORE_STEP_ROOT / "df3_10000_move_schema_2.parquet",
        kind="row_match",
        match_columns=("Firm", "Firm_3_new"),
    ),
    StageSpec(
        slug="df4_8700_move_schema_3",
        label="df4_8700_move_schema_3.parquet",
        path=BEFORE_STEP_ROOT / "df4_8700_move_schema_3.parquet",
        kind="row_match",
        match_columns=("Firm", "Firm_3_new"),
    ),
    StageSpec(
        slug="df5_6000_w_move_score_dk_moves",
        label="df5_6000_w_move_score_dk_moves.parquet",
        path=BEFORE_STEP_ROOT / "df5_6000_w_move_score_dk_moves.parquet",
        kind="row_match",
        match_columns=("Firm", "Firm_3_new"),
    ),
    StageSpec(
        slug="df6_1200_data_triangulation",
        label="df6_1200_data_triangulation.parquet",
        path=BEFORE_STEP_ROOT / "df6_1200_data_triangulation.parquet",
        kind="row_match",
        match_columns=("Firm", "Firm_3_new", "firm", "firm_clean_standardized"),
    ),
    StageSpec(
        slug="df7_close_reading",
        label="df7_close_reading.csv",
        path=BEFORE_STEP_ROOT / "df7_close_reading.csv",
        kind="row_match",
        match_columns=("firm_annotation", "firm", "firm_clean_standardized"),
    ),
]
STAGE_NUMBER = {stage.slug: index for index, stage in enumerate(STAGES, start=1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage counts for human-validation true firms.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--article-stage-input", type=Path, default=None)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument(
        "--require-human-validation-true",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When set, keep only rows with human_validation=True.",
    )
    parser.add_argument(
        "--filter-founding-origin",
        default="",
        help="Optional exact founding_origin filter, case-insensitive after stripping.",
    )
    parser.add_argument(
        "--exclude-method",
        default="",
        help="Optional exact method exclusion, case-insensitive after stripping.",
    )
    parser.add_argument("--filter-validation-label", default="", help="Optional exact validation_label filter.")
    parser.add_argument("--filter-origin-track", default="", help="Optional exact origin_track filter.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    firm_rows = load_firm_rows(
        input_path=args.input,
        sheet=args.sheet,
        require_human_validation_true=args.require_human_validation_true,
        filter_founding_origin=args.filter_founding_origin,
        exclude_method=args.exclude_method,
        include_column_filters={
            "validation_label": args.filter_validation_label,
            "origin_track": args.filter_origin_track,
        },
    )
    firm_specs = [build_firm_spec(row) for row in firm_rows]
    if args.article_stage_input is not None:
        stage_counts = count_stages_from_article_dataset(args.article_stage_input, firm_specs)
    else:
        stage_counts = {stage.slug: count_stage(stage, firm_specs) for stage in STAGES}

    output_rows = []
    for spec in firm_specs:
        deepest_stage = ""
        deepest_stage_nr = ""
        row = {
            "firm": spec["firm"],
            "aliases": " | ".join(spec["aliases_display"]),
            "deepest_stage": deepest_stage,
            "deepest_stage_nr": deepest_stage_nr,
        }
        for stage in STAGES:
            count = stage_counts[stage.slug].get(spec["firm"], 0)
            row[stage.slug] = count
            if count > 0:
                row["deepest_stage"] = stage.slug
                row["deepest_stage_nr"] = STAGE_NUMBER[stage.slug]
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_rows, args.output)
    write_summary(output_rows, args.summary_output)

    print(f"true_firms={len(output_rows)}")
    print(f"output_path={args.output}")
    print(f"summary_output_path={args.summary_output}")


def build_firm_spec(row: dict[str, Any]) -> dict[str, Any]:
    firm = get_firm_name(row)
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
    if not stage.path.exists():
        return {spec["firm"]: 0 for spec in firm_specs}
    if stage.kind == "article_scan":
        return count_article_stage(stage, firm_specs)
    return count_row_stage(stage, firm_specs)


def count_article_stage(stage: StageSpec, firm_specs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {spec["firm"]: 0 for spec in firm_specs}
    search_index = build_alias_search_index(firm_specs)
    parquet = pq.ParquetFile(stage.path)
    columns = list(stage.text_columns)
    for batch in parquet.iter_batches(columns=columns):
        frame = batch.to_pandas()
        texts = (
            frame[columns[0]].fillna("").astype(str)
            if len(columns) == 1
            else frame[list(columns)].fillna("").astype(str).agg(" ".join, axis=1)
        )
        for text in texts:
            for firm in match_firms_in_text(text, search_index):
                counts[firm] += 1
    return counts


def count_row_stage(stage: StageSpec, firm_specs: list[dict[str, Any]]) -> dict[str, int]:
    available_columns = choose_available_match_columns(stage)
    if stage.path.suffix.lower() == ".csv":
        df = read_csv_with_fallback(stage.path, usecols=available_columns)
    else:
        df = pd.read_parquet(stage.path, columns=list(available_columns))

    alias_to_firms: dict[str, set[str]] = {}
    for spec in firm_specs:
        for alias in spec["normalized_aliases"]:
            alias_to_firms.setdefault(alias, set()).add(spec["firm"])

    counts = {spec["firm"]: 0 for spec in firm_specs}
    for row in df.to_dict(orient="records"):
        matched_firms: set[str] = set()
        for column in available_columns:
            normalized = normalize_company_name(str(row.get(column) or ""))
            if not normalized:
                continue
            matched_firms.update(alias_to_firms.get(normalized, set()))
        for firm in matched_firms:
            counts[firm] += 1
    return counts


def choose_available_match_columns(stage: StageSpec) -> tuple[str, ...]:
    if stage.path.suffix.lower() == ".csv":
        columns = read_csv_columns(stage.path)
    else:
        columns = tuple(pq.ParquetFile(stage.path).schema.names)
    available = tuple(column for column in stage.match_columns if column in columns)
    if available:
        return available
    raise ValueError(f"No configured match columns found for {stage.path} in columns: {list(columns)}")


def read_csv_columns(path: Path) -> tuple[str, ...]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return tuple(pd.read_csv(path, nrows=0, encoding=encoding).columns)
        except UnicodeDecodeError:
            continue
    return tuple(pd.read_csv(path, nrows=0).columns)


def read_csv_with_fallback(path: Path, usecols: tuple[str, ...]) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, usecols=list(usecols), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, usecols=list(usecols))


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
        "## Stage Numbering",
    ]
    lines.extend(f"- `{stage.slug}` = {STAGE_NUMBER[stage.slug]}" for stage in STAGES)
    lines.extend([
        "",
        "## Stage Columns",
    ])
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


def count_stages_from_article_dataset(
    article_stage_input: Path,
    firm_specs: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    df = pd.read_csv(article_stage_input).fillna("")
    counts = {
        stage.slug: {spec["firm"]: 0 for spec in firm_specs}
        for stage in STAGES
    }
    if df.empty:
        return counts

    firm_column = "firm_name_from_list"
    for _, row in df.iterrows():
        firm = str(row.get(firm_column) or "").strip()
        if not firm:
            continue
        for stage in STAGES:
            value = str(row.get(stage.slug) or "").strip().casefold()
            if value == "true":
                counts[stage.slug][firm] += 1
    return counts


if __name__ == "__main__":
    main()
