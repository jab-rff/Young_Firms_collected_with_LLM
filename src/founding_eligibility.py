"""Helpers for enforcing the research founding-year eligibility rule."""

from __future__ import annotations

import re
from typing import Any

MIN_ELIGIBLE_FOUNDING_YEAR = 2000

POST_1999_YEAR_PATTERN = re.compile(
    r"\b(?:founded|founded in|launched|started|established|incorporated|formed)\b[^.:\n]{0,40}\b(20\d{2})\b",
    re.IGNORECASE,
)

POST_1999_HINT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bstartup\b",
        r"\bstart-up\b",
        r"\bscaleup\b",
        r"\byoung firm\b",
        r"\byoung company\b",
        r"\bearly-stage\b",
        r"\bseed-stage\b",
        r"\bventure-backed\b",
        r"\bfounded in the 20\d0s\b",
    )
]


def normalize_year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def is_eligible_founding_year(value: Any) -> bool:
    year = normalize_year(value)
    return year is not None and year >= MIN_ELIGIBLE_FOUNDING_YEAR


def has_strong_post_1999_evidence(*texts: Any) -> bool:
    combined = "\n".join(str(text or "") for text in texts if text)
    if not combined:
        return False

    if POST_1999_YEAR_PATTERN.search(combined):
        return True

    return any(pattern.search(combined) for pattern in POST_1999_HINT_PATTERNS)


def is_post_1999_eligible_or_supported(found_year: Any, *texts: Any) -> bool:
    year = normalize_year(found_year)
    if year is not None:
        return year >= MIN_ELIGIBLE_FOUNDING_YEAR
    return has_strong_post_1999_evidence(*texts)
