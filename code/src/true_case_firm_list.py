"""Helpers for loading firm lists for true-case stage analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import unicodedata

import pandas as pd

ASCII_FOLD_MAP = str.maketrans({
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "å": "aa",
    "Å": "AA",
})


def load_firm_rows(
    input_path: Path,
    sheet: str,
    require_human_validation_true: bool,
    filter_founding_origin: str,
    exclude_method: str,
    include_column_filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load firm rows from either the legacy workbook or a plain CSV file."""
    df = load_input_frame(input_path, sheet)

    if require_human_validation_true:
        if "human_validation" not in df.columns:
            raise ValueError("`human_validation` column is required when filtering workbook true cases.")
        df = df[df["human_validation"].astype(str).str.strip().str.casefold() == "true"].copy()

    if filter_founding_origin:
        if "founding_origin" not in df.columns:
            raise ValueError("`founding_origin` column is required when using --filter-founding-origin.")
        target = normalize_filter_value(filter_founding_origin)
        df = df[
            df["founding_origin"].fillna("").astype(str).map(normalize_filter_value) == target
        ].copy()

    if exclude_method:
        if "method" not in df.columns:
            raise ValueError("`method` column is required when using --exclude-method.")
        excluded = normalize_filter_value(exclude_method)
        df = df[df["method"].fillna("").astype(str).map(normalize_filter_value) != excluded].copy()

    for column, expected_value in (include_column_filters or {}).items():
        if not expected_value:
            continue
        if column not in df.columns:
            raise ValueError(f"`{column}` column is required when filtering on {column!r}.")
        target = normalize_filter_value(expected_value)
        df = df[df[column].fillna("").astype(str).map(normalize_filter_value) == target].copy()

    return df.to_dict(orient="records")


def load_input_frame(input_path: Path, sheet: str) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(input_path, sheet_name=sheet)
    return pd.read_csv(input_path)


def get_firm_name(row: dict[str, Any]) -> str:
    for key in ("firm", "name", "firm_name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def normalize_filter_value(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text.translate(ASCII_FOLD_MAP))
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return without_marks.casefold()
