"""Conservative name normalization for deterministic grouping."""

from __future__ import annotations

import re
from hashlib import sha1

_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_LEGAL_SUFFIX_RE = re.compile(
    r"(?:\s|,)+(?:a/s|aps|i/s|p/s|inc\.?|ltd\.?|limited|llc|corp\.?|corporation|gmbh)$",
    re.IGNORECASE,
)


def strip_trailing_legal_suffixes(name: str) -> str:
    """Strip trailing legal entity suffixes while preserving the core display name."""
    cleaned = name.strip(" \t\r\n.,;:")
    cleaned = cleaned.replace("\u2019", "'").replace("`", "'")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    if not cleaned:
        return ""

    stripped = cleaned
    while True:
        next_value = _TRAILING_LEGAL_SUFFIX_RE.sub("", stripped).strip(" \t\r\n.,;:")
        if next_value == stripped:
            break
        stripped = next_value
    return stripped


def normalize_company_name(name: str) -> str:
    """Normalize casing, spacing, edge punctuation, and trailing legal suffixes."""
    if "\n" in name or "\r" in name:
        return ""
    return strip_trailing_legal_suffixes(name).lower()


def make_candidate_id(normalized_name: str) -> str:
    digest = sha1(normalized_name.encode("utf-8")).hexdigest()[:12]
    return f"cand_{digest}"


def make_mention_id(retrieved_item_id: str, firm_name_raw: str, offset: int) -> str:
    value = f"{retrieved_item_id}|{firm_name_raw}|{offset}"
    digest = sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"ment_{digest}"


def make_retrieved_item_id(row_number: int, query_id: str, url: str, title: str) -> str:
    value = f"{row_number}|{query_id}|{url}|{title}"
    digest = sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"ret_{digest}"
