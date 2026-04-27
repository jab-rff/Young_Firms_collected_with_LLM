"""High-recall, heuristic company mention extraction."""

from __future__ import annotations

import re

from src.data_models import Mention, RetrievedItem
from src.normalization import make_mention_id, normalize_company_name

_LEGAL_SUFFIXES = r"(?:A/S|ApS|I/S|P/S|Inc\.?|Ltd\.?|Limited|GmbH|AG|AB|BV|LLC|Corp\.?|Corporation)"
_SUFFIX_PATTERN = re.compile(
    rf"\b([A-ZÆØÅ][\w&'.\-]*(?:\s+[A-ZÆØÅ][\w&'.\-]*){{0,5}}\s+{_LEGAL_SUFFIXES})\b"
)
_CAPITALIZED_SEQUENCE_PATTERN = re.compile(
    r"\b([A-ZÆØÅ][A-Za-zÆØÅæøå0-9&'.\-]+(?:\s+[A-ZÆØÅ][A-Za-zÆØÅæøå0-9&'.\-]+){1,4})\b"
)
_STOP_PHRASES = {
    "Denmark",
    "Danish",
    "United States",
    "United Kingdom",
    "New York",
    "San Francisco",
    "Copenhagen",
    "København",
    "European Union",
}


def extract_mentions(items: list[RetrievedItem]) -> list[Mention]:
    mentions: list[Mention] = []
    seen: set[str] = set()

    for item in items:
        text = _combined_text(item)
        for offset, raw_name in _iter_company_like_names(text):
            normalized = normalize_company_name(raw_name)
            if not normalized:
                continue
            mention_id = make_mention_id(item.retrieved_item_id, raw_name, offset)
            if mention_id in seen:
                continue
            seen.add(mention_id)
            mentions.append(
                Mention(
                    mention_id=mention_id,
                    retrieved_item_id=item.retrieved_item_id,
                    query_id=item.query_id,
                    query_text=item.query_text,
                    source_name=item.source_name,
                    title=item.title,
                    url=item.url,
                    language=item.language,
                    retrieved_at=item.retrieved_at,
                    firm_name_raw=raw_name,
                    normalized_name=normalized,
                    evidence_text=_context_window(text, offset),
                )
            )

    return mentions


def _combined_text(item: RetrievedItem) -> str:
    return "\n".join(part for part in [item.title, item.snippet, item.raw_text] if part)


def _iter_company_like_names(text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for pattern in [_SUFFIX_PATTERN, _CAPITALIZED_SEQUENCE_PATTERN]:
        for match in pattern.finditer(text):
            name = _clean_name(match.group(1))
            if _keep_name(name):
                matches.append((match.start(1), name))
    return matches


def _clean_name(name: str) -> str:
    return name.strip(" \t\r\n.,;:()[]")


def _keep_name(name: str) -> bool:
    if name in _STOP_PHRASES:
        return False
    if len(name) < 3:
        return False
    words = name.split()
    if len(words) > 6:
        return False
    return True


def _context_window(text: str, offset: int, radius: int = 240) -> str:
    start = max(0, offset - radius)
    end = min(len(text), offset + radius)
    return " ".join(text[start:end].split())
