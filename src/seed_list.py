"""Validated seed-list loading and helper views for staged discovery."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.normalization import normalize_company_name

ORIGIN_TRACK_CHOICES = ("in_denmark", "abroad_danish_founders")


@dataclass(frozen=True)
class SeedFirm:
    name: str
    founding_origin: str
    industry: str
    founded: str
    moved: str
    latest_hq_city: str
    latest_hq_country: str
    today_hq_city: str
    today_hq_country: str
    status_today: str
    employment_total: str
    employment_dk: str
    acquiror: str
    acquiror_country: str
    deal_value_th_usd: str
    deal_year: str
    method: str


def _read_text_with_supported_encodings(path: Path) -> str:
    last_error: UnicodeError | None = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return path.read_text(encoding=encoding).lstrip("\ufeff")
        except UnicodeError as exc:
            last_error = exc
    raise RuntimeError(
        f"Could not decode file {path} with supported encodings: utf-8, utf-8-sig, utf-16."
    ) from last_error


def load_seed_firms(path: Path) -> list[SeedFirm]:
    text = _read_text_with_supported_encodings(path)
    rows = csv.DictReader(text.splitlines())
    firms: list[SeedFirm] = []
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        firms.append(
            SeedFirm(
                name=name,
                founding_origin=str(row.get("founding_origin") or "").strip(),
                industry=str(row.get("industry") or "").strip(),
                founded=str(row.get("founded") or "").strip(),
                moved=str(row.get("moved") or "").strip(),
                latest_hq_city=str(row.get("latest_hq_city") or "").strip(),
                latest_hq_country=str(row.get("latest_hq_country") or "").strip(),
                today_hq_city=str(row.get("today_hq_city") or "").strip(),
                today_hq_country=str(row.get("today_hq_country") or "").strip(),
                status_today=str(row.get("status_today") or "").strip(),
                employment_total=str(row.get("employment_total") or "").strip(),
                employment_dk=str(row.get("employment_dk") or "").strip(),
                acquiror=str(row.get("acquiror") or "").strip(),
                acquiror_country=str(row.get("acquiror_country") or "").strip(),
                deal_value_th_usd=str(row.get("deal_value_th_usd") or "").strip(),
                deal_year=str(row.get("deal_year") or "").strip(),
                method=str(row.get("method") or "").strip(),
            )
        )
    return firms


def select_discovery_prompt_firms(path: Path, firms: list[SeedFirm], origin_track: str | None = None) -> list[SeedFirm]:
    stem = path.stem.lower()
    filtered = list(firms)
    if "29_04" in stem:
        filtered = [firm for firm in filtered if _is_borsen_method(firm.method)]
    if origin_track is None:
        return filtered
    return filter_seed_firms_for_origin_track(filtered, origin_track)


def filter_seed_firms_for_origin_track(firms: list[SeedFirm], origin_track: str) -> list[SeedFirm]:
    if origin_track == "in_denmark":
        return [firm for firm in firms if firm.founding_origin == "in Denmark"]
    if origin_track == "abroad_danish_founders":
        return [firm for firm in firms if firm.founding_origin == "abroad (Danish founders)"]
    raise ValueError(f"Unsupported origin track: {origin_track}")


def build_exclusion_list(firms: list[SeedFirm]) -> list[str]:
    deduped: dict[str, str] = {}
    for firm in firms:
        normalized = normalize_company_name(firm.name)
        if normalized and normalized not in deduped:
            deduped[normalized] = firm.name
    return sorted(deduped.values(), key=str.lower)


def build_origin_track_names(firms: list[SeedFirm], origin_track: str) -> list[str]:
    return build_exclusion_list(filter_seed_firms_for_origin_track(firms, origin_track))


def build_core_relocation_names(firms: list[SeedFirm]) -> list[str]:
    deduped: dict[str, str] = {}
    for firm in firms:
        if firm.founding_origin != "in Denmark":
            continue
        normalized = normalize_company_name(firm.name)
        if normalized and normalized not in deduped:
            deduped[normalized] = firm.name
    return sorted(deduped.values(), key=str.lower)


def _is_borsen_method(value: str) -> bool:
    text = str(value or "").strip()
    normalized = (
        text.replace("Ã¸", "ø")
        .replace("Ø", "ø")
        .replace("ö", "ø")
        .replace("Ö", "ø")
        .casefold()
    )
    return normalized == "børsen".casefold()
