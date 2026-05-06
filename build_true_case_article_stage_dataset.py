"""Build a row-level article dataset for `human_validation=True` firms.

Output shape:
- one row per firm x article mention in the old Borsen article pipeline
- article metadata (date, title, link, text_into_model)
- the list firm name from `results_llm_prompting.xlsx`
- stage booleans from top to bottom of the old pipeline
- a `deepest_stage` label showing how far the article-firm pair traveled
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

from src.normalization import normalize_company_name

DEFAULT_RESULTS = Path("results_llm_prompting.xlsx")
DEFAULT_SHEET = "close_reading_cases"
DEFAULT_OUTPUT = Path("analysis/true_case_article_stage_dataset.csv")
DEFAULT_SUMMARY = Path("analysis/true_case_article_stage_dataset.md")

X_ROOT = Path(r"X:\Produktivitet\_1_Mapping_successful_firms - 7002\3_step2_data_scraping\borsen_articles")
BEFORE_STEP_ROOT = X_ROOT / "0_datasets_before_each_step"

STAGE_FILES = {
    "all_articles_pair": BEFORE_STEP_ROOT / "df1_816000_articles_scraped.parquet",
    "schema1_pair": BEFORE_STEP_ROOT / "df2_60000_move_schema_1.parquet",
    "schema2_pair": BEFORE_STEP_ROOT / "df3_10000_move_schema_2.parquet",
    "schema3_pair": BEFORE_STEP_ROOT / "df4_8700_move_schema_3.parquet",
    "final_6000_pair": BEFORE_STEP_ROOT / "df5_6000_w_move_score_dk_moves.parquet",
    "triangulation_1200_pair": BEFORE_STEP_ROOT / "df6_1200_data_triangulation.parquet",
    "close_reading_pair": BEFORE_STEP_ROOT / "df7_close_reading.csv",
}

STAGE_ORDER = [
    "all_articles_pair",
    "schema1_pair",
    "schema2_pair",
    "schema3_pair",
    "final_6000_pair",
    "triangulation_1200_pair",
    "close_reading_pair",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build row-level article-stage dataset for manual-true firms.")
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

    pair_indexes = {
        "schema1_pair": load_pair_index(STAGE_FILES["schema1_pair"], "schema1_pair"),
        "schema2_pair": load_pair_index(STAGE_FILES["schema2_pair"], "schema2_pair"),
        "schema3_pair": load_pair_index(STAGE_FILES["schema3_pair"], "schema3_pair"),
        "final_6000_pair": load_pair_index(STAGE_FILES["final_6000_pair"], "final_6000_pair"),
        "triangulation_1200_pair": load_pair_index(STAGE_FILES["triangulation_1200_pair"], "triangulation_1200_pair"),
        "close_reading_pair": load_pair_index(STAGE_FILES["close_reading_pair"], "close_reading_pair"),
    }

    rows = build_article_rows(firm_specs=firm_specs, pair_indexes=pair_indexes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output)
    write_summary(rows, args.summary_output)

    print(f"article_rows={len(rows)}")
    print(f"output_path={args.output}")
    print(f"summary_output_path={args.summary_output}")


def build_firm_spec(row: dict[str, Any]) -> dict[str, Any]:
    firm = str(row.get("firm") or "").strip()
    aliases = extract_aliases(firm)
    for extra_name in (row.get("name_first"), row.get("name_today")):
        extra_text = str(extra_name or "").strip()
        if extra_text:
            aliases.update(extract_aliases(extra_text))
    normalized_aliases = {normalize_company_name(alias) for alias in aliases}
    normalized_aliases.discard("")
    patterns = build_text_patterns(aliases)
    return {
        "firm": firm,
        "aliases_display": sorted(aliases),
        "normalized_aliases": normalized_aliases,
        "patterns": patterns,
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
        patterns.append(re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE))
    return patterns


def load_pair_index(path: Path, stage: str) -> dict[tuple[str, str], dict[str, str]]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    link_column = choose_link_column(df.columns)
    firm_columns = choose_firm_columns(df.columns)
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in df.to_dict(orient="records"):
        link = clean_link(row.get(link_column))
        normalized_firms = set()
        matched_values = []
        for column in firm_columns:
            value = str(row.get(column) or "").strip()
            if not value:
                continue
            matched_values.append(value)
            normalized = normalize_company_name(value)
            if normalized:
                normalized_firms.add(normalized)
        if not normalized_firms:
            continue
        for normalized_firm in normalized_firms:
            index.setdefault(
                (link, normalized_firm),
                {
                    "stage": stage,
                    "matched_firm_variant": " | ".join(dict.fromkeys(matched_values)),
                },
            )
    return index


def build_article_rows(
    firm_specs: list[dict[str, Any]],
    pair_indexes: dict[str, dict[tuple[str, str], dict[str, str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(STAGE_FILES["all_articles_pair"])
    for batch in parquet.iter_batches(columns=["date", "title", "link", "text_into_model"]):
        frame = batch.to_pandas()
        texts = frame[["title", "text_into_model"]].fillna("").astype(str).agg(" ".join, axis=1)
        for spec in firm_specs:
            matched_mask = texts.map(lambda text: any(pattern.search(text) for pattern in spec["patterns"]))
            if not matched_mask.any():
                continue
            matched_frame = frame.loc[matched_mask].copy()
            for _, article_row in matched_frame.iterrows():
                rows.append(
                    make_output_row(
                        spec=spec,
                        article_row=article_row,
                        pair_indexes=pair_indexes,
                    )
                )
    rows.sort(key=lambda row: (row["firm_name_from_list"].casefold(), row["date"], row["title"].casefold()))
    return rows


def make_output_row(
    spec: dict[str, Any],
    article_row: pd.Series,
    pair_indexes: dict[str, dict[tuple[str, str], dict[str, str]]],
) -> dict[str, Any]:
    link = clean_link(article_row.get("link"))
    matched_variants: list[str] = []

    stage_flags = {stage: False for stage in STAGE_ORDER}
    stage_flags["all_articles_pair"] = True

    for stage_name, index in pair_indexes.items():
        matched = False
        for alias in spec["normalized_aliases"]:
            key = (link, alias)
            value = index.get(key)
            if value is not None:
                matched = True
                matched_variants.append(value["matched_firm_variant"])
                break
        stage_flags[stage_name] = matched

    deepest_stage = "all_articles_pair"
    for stage in STAGE_ORDER:
        if stage_flags[stage]:
            deepest_stage = stage

    return {
        "firm_name_from_list": spec["firm"],
        "firm_aliases_used": " | ".join(spec["aliases_display"]),
        "date": stringify(article_row.get("date")),
        "title": stringify(article_row.get("title")),
        "link": link,
        "text_into_model": stringify(article_row.get("text_into_model")),
        "matched_firm_variant_in_pipeline": " | ".join(dict.fromkeys(value for value in matched_variants if value)),
        "all_articles_pair": bool_text(stage_flags["all_articles_pair"]),
        "schema1_pair": bool_text(stage_flags["schema1_pair"]),
        "schema2_pair": bool_text(stage_flags["schema2_pair"]),
        "schema3_pair": bool_text(stage_flags["schema3_pair"]),
        "final_6000_pair": bool_text(stage_flags["final_6000_pair"]),
        "triangulation_1200_pair": bool_text(stage_flags["triangulation_1200_pair"]),
        "close_reading_pair": bool_text(stage_flags["close_reading_pair"]),
        "deepest_stage": deepest_stage,
    }


def choose_link_column(columns: Iterable[str]) -> str:
    for name in ("link", "url", "article_link"):
        if name in columns:
            return name
    raise ValueError(f"No link-like column found in columns: {list(columns)}")


def choose_firm_columns(columns: Iterable[str]) -> tuple[str, ...]:
    preferred = [
        "Firm",
        "Firm_3_new",
        "firm",
        "firm_clean_standardized",
        "firm_annotation",
        "firm_final",
    ]
    chosen = tuple(name for name in preferred if name in columns)
    if chosen:
        return chosen
    fallback = tuple(name for name in columns if "firm" in name.lower())
    if fallback:
        return fallback
    raise ValueError(f"No firm-like columns found in columns: {list(columns)}")


def clean_link(value: Any) -> str:
    text = str(value or "").strip()
    return text


def stringify(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deepest_counts = pd.Series([row["deepest_stage"] for row in rows]).value_counts()
    lines = [
        "# True Case Article Stage Dataset",
        "",
        f"- Rows: {len(rows)}",
        "",
        "## Deepest Stage Counts",
    ]
    lines.extend(f"- `{stage}`: {count}" for stage, count in deepest_counts.items())
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
