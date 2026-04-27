"""High-recall, heuristic company mention extraction."""

from __future__ import annotations

import re

from src.data_models import Mention, RetrievedItem
from src.normalization import make_mention_id, normalize_company_name

_LEGAL_SUFFIXES = r"(?:A/S|ApS|I/S|P/S|Inc\.?|Ltd\.?|Limited|GmbH|AG|AB|BV|LLC|Corp\.?|Corporation)"
_CONTEXT_TERMS = (
    "founded",
    "founded in",
    "headquartered",
    "headquarters",
    "hq",
    "moved headquarters",
    "moved its headquarters",
    "relocated",
    "startup",
    "company",
    "firm",
    "acquired",
    "merger",
    "launched",
    "based in",
    "stiftet",
    "grundlagt",
    "hovedkontor",
    "flyttede",
    "virksomhed",
    "selskab",
    "opkøbt",
)
_CAP_WORD = r"[A-ZÆØÅ][A-Za-zÆØÅæøå0-9&'.\-]*"
_SHORT_NAME_PATTERN = rf"{_CAP_WORD}(?:\s+{_CAP_WORD}){{0,2}}"
_STRONG_COMPANY_PATTERN = re.compile(
    rf"\b({_CAP_WORD}(?:\s+{_CAP_WORD}){{0,4}}\s+{_LEGAL_SUFFIXES})\b"
)
_CONTEXT_BEFORE_PATTERN = re.compile(
    rf"\b({_SHORT_NAME_PATTERN})\b"
    rf"(?=[^\n.]{{0,80}}\b(?i:{'|'.join(re.escape(term) for term in _CONTEXT_TERMS)})\b)",
)
_CONTEXT_AFTER_PATTERN = re.compile(
    rf"\b(?i:founded|acquired|launched|company|firm|startup|virksomhed|selskab|opkøbt)\b"
    rf"(?:\s+(?:the|a|an))?\s+({_SHORT_NAME_PATTERN})\b",
)
_STOP_PHRASES = {
    "California",
    "Copenhagen",
    "Denmark",
    "European Union",
    "København",
    "London",
    "New York",
    "Palo Alto",
    "San Francisco",
    "United Kingdom",
    "United States",
}
_GENERIC_SINGLE_WORDS = {
    "Article",
    "Artiklen",
    "Company",
    "Firm",
    "Historien",
    "HQ",
    "Presence",
    "Profilen",
    "Software",
    "Startup",
    "Technologies",
    "Virksomhed",
}
_TITLE_PREFIXES = {"About", "How", "What", "When", "Why"}
_HEADLINE_VERBS = {
    "Acquires",
    "Buys",
    "Expands",
    "Hires",
    "Launches",
    "Moves",
    "Opens",
    "Raises",
}
_GEOGRAPHY_WORDS = {
    "alto",
    "california",
    "copenhagen",
    "denmark",
    "francisco",
    "københavn",
    "london",
    "new",
    "palo",
    "san",
    "states",
    "united",
    "york",
}
_PERSON_NAME_DENYLIST = {
    "Alexander Aghassipour",
    "Anders Pollas",
    "Andreas Haugstrup Pedersen",
    "Christian Bach",
    "Christian Lanng",
    "Jon Froda",
    "Mathias Biilmann",
    "Michael Hansen",
    "Mikkel Hippe Brun",
    "Mikkel Jensen",
    "Mikkel Svane",
    "Ruben Bjerg Hansen",
}
_KNOWN_FIRST_NAMES = {
    "Alexander",
    "Anders",
    "Andreas",
    "Christian",
    "Jon",
    "Mathias",
    "Michael",
    "Mikkel",
    "Ruben",
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
    for pattern in [_STRONG_COMPANY_PATTERN, _CONTEXT_BEFORE_PATTERN, _CONTEXT_AFTER_PATTERN]:
        for match in pattern.finditer(text):
            raw_match = match.group(1)
            if "\n" in raw_match or "\r" in raw_match:
                continue
            name = _clean_name(raw_match)
            if _keep_name(name):
                matches.append((match.start(1), name))
    return matches


def _clean_name(name: str) -> str:
    return name.strip(" \t\r\n.,;:()[]")


def _keep_name(name: str) -> bool:
    if "\n" in name or "\r" in name:
        return False
    if re.search(r"[.!?]\s+[A-ZÆØÅ]", name):
        return False
    if name in _STOP_PHRASES:
        return False
    if name in _GENERIC_SINGLE_WORDS:
        return False
    words = name.split()
    if len(words) < 1 or len(words) > 6:
        return False
    if words[0] in _TITLE_PREFIXES:
        return False
    if len(words) >= 2 and words[1] in _HEADLINE_VERBS:
        return False
    if len(words) >= 2 and words[0] == words[1]:
        return False

    cleaned_words = [word.strip(".,;:()[]").lower() for word in words]
    if cleaned_words and all(word in _GEOGRAPHY_WORDS for word in cleaned_words):
        return False

    last_word = cleaned_words[-1]
    if "-" in last_word:
        stem, suffix = last_word.rsplit("-", 1)
        if suffix in {"based", "headquartered"} and stem in _GEOGRAPHY_WORDS and all(
            word in _GEOGRAPHY_WORDS for word in cleaned_words[:-1]
        ):
            return False

    if last_word in {"based", "headquartered"} and all(
        word in _GEOGRAPHY_WORDS for word in cleaned_words[:-1]
    ):
        return False

    if _looks_like_person_name(name, words):
        return False

    return True


def _looks_like_person_name(name: str, words: list[str]) -> bool:
    if re.search(rf"(?:^|\s){_LEGAL_SUFFIXES}$", name):
        return False
    if name in _PERSON_NAME_DENYLIST:
        return True
    if not 2 <= len(words) <= 4:
        return False
    if words[0] not in _KNOWN_FIRST_NAMES:
        return False
    return all(re.fullmatch(r"[A-ZÆØÅ][A-Za-zÆØÅæøå'.\-]+", word) for word in words)


def _context_window(text: str, offset: int, radius: int = 240) -> str:
    start = max(0, offset - radius)
    end = min(len(text), offset + radius)
    return " ".join(text[start:end].split())
