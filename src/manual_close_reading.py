"""Helpers for manual close-reading datasets and review UI."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from src.normalization import normalize_company_name

FINAL_COLUMNS = [
    "index",
    "firm",
    "group",
    "name_first",
    "name_today",
    "name_change_source",
    "move_to",
    "founded_dk",
    "relocation",
    "ma_co_occ",
    "ma_type",
    "relocation_OR_dk_founder_young",
    "entre_intra",
    "Duplicate",
    "comment_final",
    "dk_emp_2024",
    "total_emp_2024",
    "linkedin_cat",
    "source_total",
    "dk_gr_profit_2024",
    "foreign_entity_gr_profit_2024",
    "total_gr_profit_2024",
    "emp_comment",
    "annotator",
    "founding_year",
    "young",
    "founding_source",
    "founding_unsure",
    "real_move_to_country",
    "real_move_to_city",
    "move_year",
    "guesstimate",
    "guesstimate_comment",
    "relocation_source",
    "relocation_unsure",
    "status_today_manual",
    "location_today_country",
    "location_today_city",
    "industry",
    "today_source",
    "today_unsure",
    "additional_comment",
    "cvr",
    "1_comment",
    "2_comment",
    "1_founded_dk",
    "1_relocation",
    "1_ma_co_occ",
    "1_annotator",
    "approve_1st_ann",
    "2_founded_dk",
    "2_relocation",
    "2_ma_co_occ",
    "2_annotator",
    "date",
    "link",
    "acq_type",
    "acquiror_real_name",
    "acq_iso",
    "acquiror_type",
    "deal value in th USD",
    "deal_year",
    "deal_source",
    "deal_number",
    "acquiror",
    "acquiror_country",
    "target",
    "target_country",
    "deal_type",
    "deal_status",
    "deal_value_th_usd",
    "deal_date",
    "method",
    "Column1",
]

REASONING_COLUMNS = [
    "origin_track",
    "validation_label",
    "confidence",
    "validation_reason",
    "evidence_summary",
    "founder_danish_context",
    "relocation_context",
    "ma_context",
    "uncertainty_note",
    "sources_founding",
    "sources_founder_identity",
    "sources_relocation",
    "sources_ma",
    "sources_status_today",
]

DISPLAY_COLUMNS = FINAL_COLUMNS

EDITABLE_COLUMNS = [
    *FINAL_COLUMNS,
]

DEFAULT_VALIDATED_MASTER_PATH = Path("data/cumulative/model3_validated_master_all_tracks.jsonl")
DEFAULT_MANUAL_REVIEW_PATH = Path("data/manual_review/close_reading_cases.csv")
DEFAULT_FINAL_DATASET_PATH = Path("data/manual_review/final_dataset.csv")

ISO3_TO_ISO2 = {
    "ARG": "AR",
    "ARE": "AE",
    "AUS": "AU",
    "AUT": "AT",
    "BEL": "BE",
    "BRA": "BR",
    "CAN": "CA",
    "CHE": "CH",
    "CHN": "CN",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "HKG": "HK",
    "IND": "IN",
    "IRL": "IE",
    "ITA": "IT",
    "JPN": "JP",
    "LUX": "LU",
    "MEX": "MX",
    "NLD": "NL",
    "NOR": "NO",
    "NZL": "NZ",
    "PRT": "PT",
    "SGP": "SG",
    "SWE": "SE",
    "USA": "US",
    "ZAF": "ZA",
}

_LEGAL_SUFFIX_TOKENS = {
    "a",
    "s",
    "a s",
    "aps",
    "a/s",
    "ab",
    "as",
    "oy",
    "oyj",
    "ag",
    "gmbh",
    "ltd",
    "limited",
    "llc",
    "inc",
    "corp",
    "corporation",
    "plc",
    "bv",
    "nv",
    "sa",
    "sas",
    "spa",
    "sarl",
    "pte",
    "pty",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_manual_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(_sanitized_csv_lines(handle))
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            key = _row_key(row.get("firm") or row.get("name") or row.get("name_today"))
            if key:
                rows[key] = {str(k): str(v or "") for k, v in row.items()}
        return rows


def build_manual_close_reading_rows(
    validated_records: list[dict[str, Any]],
    existing_rows: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    existing_rows = existing_rows or {}
    industry_index = build_industry_index()
    rows: list[dict[str, str]] = []
    for record in validated_records:
        founding_country = normalize_country_iso2(record.get("founding_country_iso"))
        if founding_country != "DK":
            continue
        hq_today_country = normalize_country_iso2(record.get("hq_today_country_iso"))
        moved_to_country = normalize_country_iso2(record.get("moved_to_country_iso"))
        if not _has_non_danish_foreign_hq_signal(hq_today_country, moved_to_country):
            continue
        base_row = build_manual_close_reading_row(record, industry_index)
        existing = existing_rows.get(_row_key(base_row.get("firm")))
        if existing:
            merged = dict(base_row)
            for column in DISPLAY_COLUMNS:
                if column in existing and str(existing.get(column) or "").strip():
                    merged[column] = str(existing.get(column) or "")
            rows.append(merged)
        else:
            rows.append(base_row)
    return sorted(rows, key=lambda row: row["firm"].casefold())


def build_manual_close_reading_row(
    record: dict[str, Any],
    industry_index: dict[str, str] | None = None,
) -> dict[str, str]:
    industry_index = industry_index or {}
    normalized_name = normalize_company_name(record.get("firm_name"))
    firm_name = str(record.get("firm_name") or "").strip()
    first_legal_name = str(record.get("first_legal_entity_name") or "").strip()
    moved_to_first_legal_name = str(record.get("moved_to_first_legal_entity_name") or "").strip()
    first_cvr = str(record.get("first_cvr") or "").strip()
    latest_hq_city = str(record.get("moved_to_city") or "").strip()
    latest_hq_country = normalize_country_iso2(record.get("moved_to_country_iso"))
    today_hq_city = str(record.get("hq_today_city") or "").strip()
    today_hq_country = normalize_country_iso2(record.get("hq_today_country_iso"))
    if latest_hq_city and latest_hq_country and today_hq_city == latest_hq_city and today_hq_country == latest_hq_country:
        today_hq_city = ""
        today_hq_country = ""
    sources = {
        "sources_founding": _join_sources(record.get("sources_founding")),
        "sources_founder_identity": _join_sources(record.get("sources_founder_identity")),
        "sources_relocation": _join_sources(record.get("sources_relocation")),
        "sources_ma": _join_sources(record.get("sources_ma")),
        "sources_status_today": _join_sources(record.get("sources_status_today")),
    }
    source_total = _join_sources(
        [
            *list(record.get("sources_founding") or []),
            *list(record.get("sources_founder_identity") or []),
            *list(record.get("sources_relocation") or []),
            *list(record.get("sources_ma") or []),
            *list(record.get("sources_status_today") or []),
        ]
    )
    acq_year = _extract_year(record.get("acq_date"))
    industry_detailed = industry_index.get(normalized_name, "")
    founded_dk = _tri_state_text(record.get("founded_in_denmark"))
    relocation = _tri_state_text(record.get("moved_hq_abroad"))
    ma_co_occ = _tri_state_text(record.get("ma_after_or_during_move"))
    has_deal_evidence = _has_deal_evidence(record)
    founding_year = _stringify_year(record.get("founding_year"))
    young = _young_flag(record.get("founding_year"))
    founding_unsure = "true" if founded_dk == "unclear" or not founding_year else "false"
    relocation_unsure = "true" if relocation == "unclear" else "false"
    today_unsure = "true" if not today_hq_city or not today_hq_country else "false"
    move_to = latest_hq_country
    has_substantive_name_change = _is_substantive_name_change(first_legal_name, moved_to_first_legal_name)
    name_change_source = (
        sources["sources_status_today"]
        if has_substantive_name_change
        else ""
    )
    row = {
        "index": "",
        "firm": firm_name,
        "group": "",
        "name_first": first_legal_name,
        "name_today": moved_to_first_legal_name,
        "name_change_source": name_change_source,
        "move_to": move_to,
        "founded_dk": founded_dk,
        "relocation": relocation,
        "ma_co_occ": ma_co_occ,
        "ma_type": str(record.get("ma_type") or "").strip(),
        "relocation_OR_dk_founder_young": "",
        "entre_intra": "",
        "Duplicate": "",
        "comment_final": "",
        "dk_emp_2024": "",
        "total_emp_2024": "",
        "linkedin_cat": "",
        "source_total": "",
        "dk_gr_profit_2024": "",
        "foreign_entity_gr_profit_2024": "",
        "total_gr_profit_2024": "",
        "emp_comment": "",
        "annotator": "",
        "founding_year": founding_year,
        "young": young,
        "founding_source": sources["sources_founding"],
        "founding_unsure": founding_unsure,
        "real_move_to_country": latest_hq_country,
        "real_move_to_city": latest_hq_city,
        "move_year": _stringify_year(record.get("move_year")),
        "guesstimate": "",
        "guesstimate_comment": "",
        "relocation_source": sources["sources_relocation"],
        "relocation_unsure": relocation_unsure,
        "status_today_manual": str(record.get("status_today") or "").strip(),
        "location_today_country": today_hq_country,
        "location_today_city": today_hq_city,
        "industry": _coarse_industry(industry_detailed),
        "today_source": sources["sources_status_today"],
        "today_unsure": today_unsure,
        "additional_comment": "",
        "cvr": first_cvr,
        "1_comment": "",
        "2_comment": "",
        "1_founded_dk": "",
        "1_relocation": "",
        "1_ma_co_occ": "",
        "1_annotator": "",
        "approve_1st_ann": "",
        "2_founded_dk": "",
        "2_relocation": "",
        "2_ma_co_occ": "",
        "2_annotator": "",
        "date": "",
        "link": "",
        "acq_type": str(record.get("ma_type") or "").strip() if has_deal_evidence else "",
        "acquiror_real_name": str(record.get("acquirer") or "").strip() if has_deal_evidence else "",
        "acq_iso": "",
        "acquiror_type": "",
        "deal value in th USD": _safe_text(record.get("deal_value_th_usd") or record.get("deal_value_usd_thousands")) if has_deal_evidence else "",
        "deal_year": acq_year if has_deal_evidence else "",
        "deal_source": sources["sources_ma"] if has_deal_evidence else "",
        "deal_number": "",
        "acquiror": str(record.get("acquirer") or "").strip() if has_deal_evidence else "",
        "acquiror_country": "",
        "target": firm_name if has_deal_evidence else "",
        "target_country": "",
        "deal_type": str(record.get("ma_type") or "").strip() if has_deal_evidence else "",
        "deal_status": "",
        "deal_value_th_usd": _safe_text(record.get("deal_value_th_usd") or record.get("deal_value_usd_thousands")) if has_deal_evidence else "",
        "deal_date": _safe_text(record.get("acq_date")) if has_deal_evidence else "",
        "method": "LLM pipeline",
        "Column1": "unclear",
        "origin_track": str(record.get("origin_track") or "").strip(),
        "validation_label": str(record.get("validation_label") or "").strip(),
        "confidence": str(record.get("confidence") or "").strip(),
        "validation_reason": _clean_text(record.get("validation_reason")),
        "evidence_summary": _clean_text(record.get("evidence_summary")),
        "founder_danish_context": _clean_text(record.get("founder_danish_context")),
        "relocation_context": _clean_text(record.get("relocation_context")),
        "ma_context": _clean_text(record.get("ma_context")),
        "uncertainty_note": _clean_text(record.get("uncertainty_note")),
        **sources,
    }
    return row


def build_industry_index(discovery_dir: Path = Path("data/discovery")) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in sorted(discovery_dir.glob("snowball_round_*.jsonl")):
        if path.name.endswith("_deduped.jsonl") or path.name.endswith("_bucket_runs.jsonl") or path.name.endswith("_api_costs.jsonl"):
            continue
        for row in load_jsonl(path):
            normalized_name = normalize_company_name(row.get("firm_name"))
            if not normalized_name or normalized_name in index:
                continue
            sector_if_known = str(row.get("sector_if_known") or "").strip()
            if sector_if_known:
                index[normalized_name] = sector_if_known
                continue
            discovery_bucket = str(row.get("discovery_bucket") or "").strip()
            if discovery_bucket.startswith("sector:"):
                index[normalized_name] = discovery_bucket.split(":", 1)[1].strip()
    return index


def save_manual_close_reading_rows(
    rows: list[dict[str, Any]],
    manual_review_path: Path = DEFAULT_MANUAL_REVIEW_PATH,
    final_dataset_path: Path = DEFAULT_FINAL_DATASET_PATH,
) -> None:
    manual_review_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, DISPLAY_COLUMNS, manual_review_path)
    final_only_rows = [{column: _safe_text(row.get(column)) for column in FINAL_COLUMNS} for row in rows]
    write_csv(final_only_rows, FINAL_COLUMNS, final_dataset_path)


def write_csv(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _safe_text(row.get(field)) for field in fieldnames})


def sanitize_manual_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for row in rows:
        clean_row = {column: _safe_text(row.get(column)) for column in FINAL_COLUMNS}
        for column in ("real_move_to_country", "location_today_country", "acq_iso"):
            clean_row[column] = normalize_country_iso2(clean_row.get(column))
        human_validation = clean_row.get("Column1", "").strip().lower()
        if human_validation not in {"true", "false", "unclear"}:
            clean_row["Column1"] = "unclear"
        else:
            clean_row["Column1"] = human_validation
        sanitized.append(clean_row)
    return sanitized


def find_invalid_iso2_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    invalid: list[tuple[str, str, str]] = []
    for row in rows:
        name = _safe_text(row.get("firm"))
        for column in ("real_move_to_country", "location_today_country", "acq_iso"):
            value = _safe_text(row.get(column))
            if value and not is_valid_iso2(value):
                invalid.append((name, column, value))
    return invalid


def normalize_country_iso2(value: Any) -> str:
    text = _safe_text(value).upper()
    if not text:
        return ""
    if len(text) == 2 and text.isalpha():
        return text
    if len(text) == 3 and text in ISO3_TO_ISO2:
        return ISO3_TO_ISO2[text]
    return text


def is_valid_iso2(value: Any) -> bool:
    text = _safe_text(value).upper()
    return len(text) == 2 and text.isalpha()


def map_origin_track_to_founding_origin(value: Any) -> str:
    if str(value or "").strip() == "abroad_danish_founders":
        return "abroad (Danish founders)"
    return "in Denmark"


def _join_sources(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, list):
        return " | ".join(_safe_text(value) for value in values if _safe_text(value))
    return _safe_text(values)


def _first_source_url(value: str) -> str:
    text = _safe_text(value).strip()
    if not text:
        return ""
    return text.split(" | ", 1)[0].strip()


def _extract_year(value: Any) -> str:
    text = _safe_text(value)
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else ""


def _stringify_year(value: Any) -> str:
    text = _safe_text(value)
    return text


def _clean_text(value: Any) -> str:
    return _safe_text(value).replace("\x00", "")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _tri_state_text(value: Any) -> str:
    text = _safe_text(value).strip().lower()
    if text in {"true", "false", "unclear"}:
        return text
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unclear"


def _young_flag(value: Any) -> str:
    year_text = _safe_text(value).strip()
    if not year_text:
        return "unclear"
    try:
        year = int(float(year_text))
    except ValueError:
        return "unclear"
    return "true" if year >= 1999 else "false"


def _coarse_industry(value: str) -> str:
    text = _safe_text(value).strip().casefold()
    if not text:
        return ""
    biotech_markers = [
        "biotech",
        "bio ",
        "bioscience",
        "pharma",
        "therapeutic",
        "vaccine",
        "oncology",
        "genomic",
    ]
    if any(marker in text for marker in biotech_markers):
        return "biotech"
    tech_markers = [
        "tech",
        "software",
        "saas",
        "fintech",
        "gaming",
        "medtech",
        "ai",
        "analytics",
        "cloud",
        "digital",
        "platform",
        "internet",
        "app",
        "data",
    ]
    if any(marker in text for marker in tech_markers):
        return "tech"
    return "others"


def _is_substantive_name_change(first_name: Any, today_name: Any) -> bool:
    first = _normalize_name_for_change_check(first_name)
    today = _normalize_name_for_change_check(today_name)
    return bool(first and today and first != today)


def _normalize_name_for_change_check(value: Any) -> str:
    text = _safe_text(value).casefold().strip()
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [token for token in text.split() if token and token not in _LEGAL_SUFFIX_TOKENS]
    return " ".join(tokens)


def _has_deal_evidence(record: dict[str, Any]) -> bool:
    direct_fields = [
        record.get("acquirer"),
        record.get("ma_type"),
        record.get("acq_date"),
        record.get("deal_value_th_usd"),
        record.get("deal_value_usd_thousands"),
    ]
    if any(_safe_text(value).strip() for value in direct_fields):
        return True
    return _tri_state_text(record.get("ma_after_or_during_move")) == "true"


def _row_key(name: Any) -> str:
    if name is None:
        return ""
    normalized_name = normalize_company_name(name)
    if not normalized_name:
        return ""
    return normalized_name


def _sanitized_csv_lines(handle: Any) -> Any:
    for line in handle:
        yield str(line).replace("\x00", "")


def _has_non_danish_foreign_hq_signal(hq_today_country: str, moved_to_country: str) -> bool:
    disallowed = {"", "DK"}
    return hq_today_country not in disallowed or moved_to_country not in disallowed
