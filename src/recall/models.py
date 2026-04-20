"""Minimal recall-stage data model with provenance-preserving records."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchQuery:
    query_text: str
    language: str
    family: str
    priority: int


@dataclass(frozen=True)
class SearchResult:
    query_text: str
    source_url: str
    source_title: str
    snippet: str
    retrieved_at: str


@dataclass(frozen=True)
class CandidateMention:
    firm_name_raw: str
    normalized_name: str
    source_url: str
    source_title: str
    snippet: str
    query_used: str
    retrieval_timestamp: str


@dataclass
class CandidateFirm:
    normalized_name: str
    raw_name_variants: set[str] = field(default_factory=set)
    mentions: list[CandidateMention] = field(default_factory=list)

    def mention_count(self) -> int:
        return len(self.mentions)
