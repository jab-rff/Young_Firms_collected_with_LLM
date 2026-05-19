"""Collapse the abroad-Danish-founders review file to one row per firm.

Rules:
- Prefer `first_legal_entity_name` as the firm key when present.
- Fall back to `firm_name` when legal name is missing.
- Merge duplicate rows conservatively by keeping the first non-empty scalar value
  and unioning selected text/url fields.
- Extract up to five likely Danish founder names from `founder_danish_context`.
- Extract up to five personal LinkedIn profile URLs from `sources_founder_identity`.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/cumulative/final_review_master_abroad_danish_founders.csv")
DEFAULT_OUTPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_unique_firms.csv")
DEFAULT_DIFFBOT_INPUT = Path("diffbot_dk_founder_after_1999.csv")

FOUNDER_NAME_COLUMNS = [f"dk_founder_{index}" for index in range(1, 6)]
FOUNDER_LINKEDIN_COLUMNS = [f"dk_founder_{index}_linkedin" for index in range(1, 6)]

URL_PATTERN = re.compile(r"https?://[^\s|)]+", flags=re.IGNORECASE)
LINKEDIN_PERSON_PATTERN = re.compile(r"https?://[a-z]{0,3}\.?linkedin\.com/in/[^\s|)]+", flags=re.IGNORECASE)
LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(llc|inc|corp|corporation|ltd|limited|aps|a/s|a\.s\.|ab|ag|gmbh|s\.l\.|s\.l|srl|s\.r\.l\.|sa|bv|plc|oy|as)\b",
    flags=re.IGNORECASE,
)
GENERIC_DESCRIPTOR_PATTERN = re.compile(
    r"\b(group|holding|holdings|studio|design|architecture|architects)\b",
    flags=re.IGNORECASE,
)
NAME_PATTERN = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){0,4})\b"
)
EXPLICIT_NAME_PATTERNS = [
    re.compile(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){1,4})\s+is\s+(?:documented\s+as\s+)?Danish\b"),
    re.compile(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){1,4})\s+is\s+described.*?\s+as\s+Danish\b", flags=re.IGNORECASE),
    re.compile(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){1,4})\s+is\s+identified.*?\s+as\s+Danish\b", flags=re.IGNORECASE),
    re.compile(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){1,4})\s+.*?nationality\s+'Danish'\b"),
    re.compile(r"\bDanish-born\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){1,4})\b"),
    re.compile(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){1,4}),\s+a\s+Danish\b", flags=re.IGNORECASE),
    re.compile(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){1,4})\s+\(born.*?Denmark\)", flags=re.IGNORECASE),
    re.compile(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'._-]+){1,4})\s+.*?\bfrom Denmark\b", flags=re.IGNORECASE),
]
STOP_NAMES = {
    "Companies House",
    "Royal Society",
    "University Of Copenhagen",
    "Copenhagen University",
    "Novo Nordisk Foundation",
    "Protein Research",
    "Danish Chamber",
    "Danish Natural Research Council",
    "Danish Business Registry",
    "Clue",
    "Berlin",
    "Denmark",
    "Copenhagen",
    "Sweden",
    "London",
    "New York",
    "Danish",
    "Danish-born",
    "Founder",
    "Co Founder",
    "Co-Founder",
    "Our Team",
    "Multiple",
    "October",
    "Company",
    "Commerce",
    "Fantastic Services",
    "The Wise Wolf",
    "Abzu",
    "Clue",
}
STOP_TOKENS = {
    "about",
    "abzu",
    "clue",
    "commerce",
    "company",
    "copenhagen",
    "danish",
    "denmark",
    "fantastic",
    "medical",
    "managing",
    "october",
    "our",
    "director",
    "fellow",
    "school",
    "senior",
    "services",
    "team",
    "university",
    "vice",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one-row-per-firm abroad-Danish-founders review CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diffbot-input", type=Path, default=DEFAULT_DIFFBOT_INPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input).where(pd.notna, "")
    diffbot_keys = load_diffbot_keys(args.diffbot_input)
    rows = df.to_dict(orient="records")
    deduped = dedupe_rows(rows, diffbot_keys=diffbot_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(deduped, args.output)
    print(f"input_rows={len(rows)}")
    print(f"unique_firms={len(deduped)}")
    print(f"output_path={args.output}")


def dedupe_rows(rows: list[dict[str, Any]], diffbot_keys: set[str]) -> list[dict[str, Any]]:
    merged_by_key: dict[str, dict[str, Any]] = {}
    source_names_by_key: dict[str, list[str]] = {}

    for row in rows:
        canonical_key, display_key = build_firm_key(row)
        if canonical_key not in merged_by_key:
            merged_by_key[canonical_key] = dict(row)
            source_names_by_key[canonical_key] = [clean_text(row.get("firm_name"))] if clean_text(row.get("firm_name")) else []
            merged_by_key[canonical_key]["unique_firm_key"] = display_key
            merged_by_key[canonical_key]["source_row_count"] = 1
            continue

        merged = merged_by_key[canonical_key]
        merged["source_row_count"] = int(merged.get("source_row_count") or 1) + 1
        firm_name = clean_text(row.get("firm_name"))
        if firm_name and firm_name not in source_names_by_key[canonical_key]:
            source_names_by_key[canonical_key].append(firm_name)

        for field, value in row.items():
            existing = merged.get(field)
            if field in {
                "validation_reason",
                "evidence_summary",
                "founding_evidence",
                "founder_danish_context",
                "relocation_evidence",
                "ma_evidence",
                "relocation_context",
                "ma_context",
                "uncertainty_note",
                "sources_founding",
                "sources_founder_identity",
                "sources_relocation",
                "sources_ma",
                "sources_status_today",
            }:
                merged[field] = merge_text(existing, value)
            elif not clean_text(existing):
                merged[field] = value

    output_rows: list[dict[str, Any]] = []
    for canonical_key, row in merged_by_key.items():
        row["source_firm_names_merged"] = " | ".join(source_names_by_key[canonical_key])
        row["found_in_diffbot"] = "true" if has_diffbot_match(row, diffbot_keys) else "false"
        founders = extract_likely_danish_founders(clean_text(row.get("founder_danish_context")))
        linkedin_urls = extract_person_linkedin_urls(clean_text(row.get("sources_founder_identity")))
        for index, column in enumerate(FOUNDER_NAME_COLUMNS):
            row[column] = founders[index] if index < len(founders) else ""
        for index, column in enumerate(FOUNDER_LINKEDIN_COLUMNS):
            row[column] = linkedin_urls[index] if index < len(linkedin_urls) else ""
        output_rows.append(row)

    output_rows.sort(key=lambda row: normalize_firm_key(clean_text(row.get("unique_firm_key"))))
    return output_rows


def build_firm_key(row: dict[str, Any]) -> tuple[str, str]:
    legal_name = clean_text(row.get("first_legal_entity_name"))
    if legal_name:
        return normalize_firm_key(legal_name), legal_name
    firm_name = clean_text(row.get("firm_name"))
    normalized = normalize_firm_key(firm_name)
    return normalized, firm_name


def load_diffbot_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path).where(pd.notna, "")
    keys: set[str] = set()
    for row in df.to_dict(orient="records"):
        for candidate in (
            clean_text(row.get("name")),
            clean_text(row.get("homepageUri")),
            clean_text(row.get("linkedInUri")),
        ):
            if not candidate:
                continue
            normalized = normalize_firm_key(candidate)
            if normalized:
                keys.add(normalized)
    return keys


def has_diffbot_match(row: dict[str, Any], diffbot_keys: set[str]) -> bool:
    aliases: set[str] = set()
    for raw in (
        clean_text(row.get("unique_firm_key")),
        clean_text(row.get("firm_name")),
        clean_text(row.get("source_firm_names_merged")),
        clean_text(row.get("first_legal_entity_name")),
    ):
        aliases.update(extract_name_aliases(raw))
    return any(alias in diffbot_keys for alias in aliases if alias)


def extract_name_aliases(text: str) -> set[str]:
    raw = clean_text(text)
    if not raw:
        return set()
    aliases = {normalize_firm_key(raw)}
    for part in raw.split(" | "):
        part = clean_text(part)
        if not part:
            continue
        aliases.add(normalize_firm_key(part))
        for subpart in re.split(r"[()/]", part):
            subpart = clean_text(subpart)
            if subpart:
                aliases.add(normalize_firm_key(subpart))
    aliases.discard("")
    return aliases


def normalize_firm_key(text: str) -> str:
    normalized = clean_text(text).casefold()
    normalized = transliterate_scandinavian(normalized)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"\s*\([^)]*\)", "", normalized)
    normalized = LEGAL_SUFFIX_PATTERN.sub(" ", normalized)
    normalized = GENERIC_DESCRIPTOR_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"[&/,+.'-]", " ", normalized)
    normalized = re.sub(r"\bpre google maps\b", " ", normalized)
    normalized = re.sub(r"\bformerly\b", " ", normalized)
    normalized = re.sub(r"\bthe\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def transliterate_scandinavian(text: str) -> str:
    replacements = {
        "æ": "ae",
        "ø": "o",
        "å": "aa",
        "ä": "a",
        "ö": "o",
        "ü": "u",
        "é": "e",
        "è": "e",
    }
    return "".join(replacements.get(char, char) for char in text)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def merge_text(left: Any, right: Any) -> str:
    parts: list[str] = []
    for raw in (left, right):
        text = clean_text(raw)
        if not text:
            continue
        for piece in text.split(" | "):
            piece = piece.strip()
            if piece and piece not in parts:
                parts.append(piece)
    return " | ".join(parts)


def extract_person_linkedin_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in LINKEDIN_PERSON_PATTERN.findall(text):
        url = match.rstrip(".,);")
        if url not in urls:
            urls.append(url)
        if len(urls) == 5:
            break
    return urls


def extract_likely_danish_founders(text: str) -> list[str]:
    names: list[str] = []
    if not text:
        return names

    for pattern in EXPLICIT_NAME_PATTERNS:
        for match in pattern.findall(text):
            cleaned = normalize_name(match)
            if not cleaned or cleaned in STOP_NAMES or cleaned in names:
                continue
            names.append(cleaned)
            if len(names) == 5:
                return names
    if names:
        return names

    candidate_sentences = split_candidate_sentences(text)
    for sentence in candidate_sentences:
        lowered = sentence.casefold()
        if not any(
            marker in lowered
            for marker in (
                "danish",
                "denmark",
                "copenhagen",
                "nationality",
                "born in",
                "from denmark",
            )
        ):
            continue
        for name in NAME_PATTERN.findall(sentence):
            cleaned = normalize_name(name)
            if not cleaned or cleaned in STOP_NAMES:
                continue
            if token_count(cleaned) < 2:
                continue
            if contains_stop_token(cleaned):
                continue
            if cleaned not in names:
                names.append(cleaned)
            if len(names) == 5:
                return names
    return names


def split_candidate_sentences(text: str) -> list[str]:
    normalized = text.replace("—", " ").replace("–", " ")
    return [piece.strip() for piece in re.split(r"(?<=[.!?;])\s+", normalized) if piece.strip()]


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip(" -,:;()[]{}")
    cleaned = re.sub(
        r"^(Founder|Co Founder|Co-Founder|Founder/co-founder context|Dr|Mr|Mrs|Ms|Professor)\.?\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(Ph\.?D\.?|M\.?D\.?)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,:;()[]{}")
    if len(cleaned) < 4:
        return ""
    if not any(char.isalpha() for char in cleaned):
        return ""
    return cleaned


def token_count(text: str) -> int:
    return len([part for part in text.split() if part])


def contains_stop_token(text: str) -> bool:
    return any(token.casefold() in STOP_TOKENS for token in text.split())


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["unique_firm_key"])
        return
    fieldnames = list(rows[0].keys())
    ordered = [
        "unique_firm_key",
        "source_row_count",
        "source_firm_names_merged",
        "found_in_diffbot",
        *[
            name
            for name in fieldnames
            if name not in {"unique_firm_key", "source_row_count", "source_firm_names_merged", "found_in_diffbot"}
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
