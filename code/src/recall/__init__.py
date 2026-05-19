"""Recall pipeline package."""

from recall.models import CandidateFirm, CandidateMention, SearchQuery, SearchResult
from recall.storage import (
    aggregate_candidate_mentions,
    load_candidate_mentions_jsonl,
    normalize_firm_name,
    save_candidate_mentions_jsonl,
)

__all__ = [
    "SearchQuery",
    "SearchResult",
    "CandidateMention",
    "CandidateFirm",
    "normalize_firm_name",
    "save_candidate_mentions_jsonl",
    "load_candidate_mentions_jsonl",
    "aggregate_candidate_mentions",
]
