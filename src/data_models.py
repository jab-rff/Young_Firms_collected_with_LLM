"""Typed records for the recall-first research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


UncertainBool = Literal["true", "false", "uncertain"]
MAType = Optional[Literal["acquisition", "merger", "unknown"]]


@dataclass(frozen=True)
class Query:
    query_id: str
    query_text: str
    language: str
    family: str
    created_at: str


@dataclass(frozen=True)
class RetrievedItem:
    retrieved_item_id: str
    query_id: str
    query_text: str
    source_name: str
    title: str
    snippet: str
    url: str
    language: str
    retrieved_at: str
    raw_text: str


@dataclass(frozen=True)
class Mention:
    mention_id: str
    retrieved_item_id: str
    query_id: str
    query_text: str
    source_name: str
    title: str
    url: str
    language: str
    retrieved_at: str
    firm_name_raw: str
    normalized_name: str
    evidence_text: str


@dataclass(frozen=True)
class CandidateFirm:
    candidate_id: str
    firm_name: str
    normalized_name: str
    raw_name_variants: List[str] = field(default_factory=list)
    mention_ids: List[str] = field(default_factory=list)
    source_names: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    evidence_count: int = 0


@dataclass(frozen=True)
class EnrichmentResult:
    candidate_id: str
    firm_name: str
    founded_in_denmark: UncertainBool
    founding_year: Optional[int]
    founding_city: Optional[str]
    founding_country_iso: Optional[str]
    moved_hq_abroad: UncertainBool
    move_year: Optional[int]
    moved_to_city: Optional[str]
    moved_to_country_iso: Optional[str]
    relocation_context: Optional[str]
    co_occured_with_ma: UncertainBool
    ma_type: MAType
    reasoning: str
    sources: List[str]
    model_name: str
    prompt_version: str
    raw_response: str
    created_at: str
