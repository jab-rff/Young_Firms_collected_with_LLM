"""Helpers for the separate Diffbot-founder pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DIFFBOT_INPUT_PATH = Path(
    r"X:\Produktivitet\_1_Mapping_successful_firms - 7002\3_step2_data_scraping\diffbot_dk_founder_after_1999.csv"
)
DEFAULT_PIPELINE_SLUG = "diffbot_dk_founder_after_1999"


@dataclass(frozen=True)
class DiffbotPipelinePaths:
    slug: str
    candidates: Path
    model2: Path
    model3: Path
    review: Path
    master_validated: Path
    master_review: Path


def build_diffbot_paths(slug: str = DEFAULT_PIPELINE_SLUG) -> DiffbotPipelinePaths:
    return DiffbotPipelinePaths(
        slug=slug,
        candidates=Path("data/diffbot") / f"{slug}_candidates.jsonl",
        model2=Path("data/diffbot") / f"{slug}_enriched.jsonl",
        model3=Path("data/diffbot") / f"{slug}_validated.jsonl",
        review=Path("data/diffbot") / f"{slug}_review.csv",
        master_validated=Path("data/diffbot") / f"{slug}_master_validated.jsonl",
        master_review=Path("data/diffbot") / f"{slug}_master_review.csv",
    )


def load_diffbot_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    return df.where(pd.notna(df), None).to_dict(orient="records")


def build_diffbot_candidate(row: dict[str, Any]) -> dict[str, Any]:
    firm_name = _safe_text(row.get("name"))
    homepage = _safe_text(row.get("homepageUri"))
    linkedin = _safe_text(row.get("linkedInUri"))
    founder_names = _safe_text(row.get("founders_name"))
    summary = _safe_text(row.get("summary"))
    location_bits = [
        _safe_text(row.get("location_city_name")),
        _safe_text(row.get("location_region_name")),
        _safe_text(row.get("location_country_name")),
    ]
    location_hint = ", ".join(part for part in location_bits if part)
    sources = [value for value in [homepage, linkedin] if value]

    founder_hint = (
        f"Third-party Diffbot founder hint: {founder_names}."
        if founder_names
        else "Third-party Diffbot founder hint exists but founder names are missing."
    )
    location_hint_text = (
        f"Diffbot current location hint: {location_hint}."
        if location_hint
        else "Diffbot current location hint is missing."
    )
    summary_hint = f"Diffbot summary: {summary}." if summary else "Diffbot summary is missing."

    return {
        "firm_name": firm_name,
        "origin_track": "abroad_danish_founders",
        "founded_in_denmark": "uncertain",
        "danish_founders_abroad": "uncertain",
        "founding_year": None,
        "founding_city": None,
        "founding_country_iso": None,
        "moved_hq_abroad": "uncertain",
        "move_year": None,
        "moved_to_city": None,
        "moved_to_country_iso": None,
        "ma_co_occurred": "uncertain",
        "ma_type": "unknown",
        "founding_evidence": None,
        "founder_danish_evidence": founder_hint,
        "relocation_evidence": location_hint_text,
        "ma_evidence": None,
        "reasoning": (
            "Third-party founder-abroad seed. Do not treat Diffbot fields as confirmed facts; "
            "verify founders, founding location, founding year, and current HQ with real sources."
        ),
        "sources": sources,
        "confidence_note": (
            "Seed list claims Danish-founder / after-1999 relevance, but these are unverified hints only."
        ),
        "source_record": {
            "seed_source": "diffbot_dk_founder_after_1999",
            "firm_name_diffbot": firm_name,
            "founders_name": founder_names,
            "homepageUri": homepage,
            "linkedInUri": linkedin,
            "location_city_name": _safe_text(row.get("location_city_name")),
            "location_region_name": _safe_text(row.get("location_region_name")),
            "location_country_name": _safe_text(row.get("location_country_name")),
            "summary": summary,
            "categories_name": _safe_text(row.get("categories_name")),
            "ceo_name": _safe_text(row.get("ceo_name")),
            "nbEmployeesMax": row.get("nbEmployeesMax"),
            "founders_targetDiffbotId": _safe_text(row.get("founders_targetDiffbotId")),
        },
        "third_party_seed": {
            "source": "diffbot_dk_founder_after_1999",
            "name": firm_name,
            "founders_name": founder_names,
            "homepageUri": homepage,
            "linkedInUri": linkedin,
            "location_city_name": _safe_text(row.get("location_city_name")),
            "location_country_name": _safe_text(row.get("location_country_name")),
            "location_region_name": _safe_text(row.get("location_region_name")),
            "summary": summary,
        },
        "seed_summary": " ".join(part for part in [founder_hint, location_hint_text, summary_hint] if part),
        "prompt_version_seed": "2026-05-06-diffbot-seed-v1",
    }


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
