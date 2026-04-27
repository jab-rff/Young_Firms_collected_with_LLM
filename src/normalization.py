"""Conservative name normalization for deterministic grouping."""

from __future__ import annotations

import re
from hashlib import sha1

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_company_name(name: str) -> str:
    """Normalize only casing, spacing, and edge punctuation."""
    cleaned = name.strip(" \t\r\n.,;:")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.lower()


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
