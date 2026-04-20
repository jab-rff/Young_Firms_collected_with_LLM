"""Local storage helpers for candidate mentions and firm aggregation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from recall.models import CandidateFirm, CandidateMention

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_firm_name(name: str) -> str:
    """Normalize a firm name conservatively for exact-key grouping."""
    stripped = name.strip().lower()
    return _WHITESPACE_RE.sub(" ", stripped)


def save_candidate_mentions_jsonl(mentions: list[CandidateMention], output_path: Path) -> None:
    """Write candidate mentions to JSONL, one record per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for mention in mentions:
            handle.write(json.dumps(asdict(mention), ensure_ascii=False) + "\n")


def load_candidate_mentions_jsonl(input_path: Path) -> list[CandidateMention]:
    """Load candidate mentions from a JSONL file."""
    mentions: list[CandidateMention] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            mentions.append(CandidateMention(**json.loads(line)))
    return mentions


def aggregate_candidate_mentions(mentions: list[CandidateMention]) -> list[CandidateFirm]:
    """Group mentions by normalized name while preserving provenance."""
    firms_by_name: dict[str, CandidateFirm] = {}
    for mention in mentions:
        firm = firms_by_name.get(mention.normalized_name)
        if firm is None:
            firm = CandidateFirm(normalized_name=mention.normalized_name)
            firms_by_name[mention.normalized_name] = firm
        firm.raw_name_variants.add(mention.firm_name_raw)
        firm.mentions.append(mention)
    return list(firms_by_name.values())
