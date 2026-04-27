"""Exact-key aggregation of mentions into candidate firms."""

from __future__ import annotations

from collections import defaultdict

from src.data_models import CandidateFirm, Mention
from src.normalization import make_candidate_id


def aggregate_mentions(mentions: list[Mention]) -> list[CandidateFirm]:
    grouped: dict[str, list[Mention]] = defaultdict(list)
    for mention in mentions:
        grouped[mention.normalized_name].append(mention)

    candidates: list[CandidateFirm] = []
    for normalized_name, group in sorted(grouped.items()):
        raw_variants = sorted({mention.firm_name_raw for mention in group})
        source_names = sorted({mention.source_name for mention in group if mention.source_name})
        source_urls = sorted({mention.url for mention in group if mention.url})
        candidates.append(
            CandidateFirm(
                candidate_id=make_candidate_id(normalized_name),
                firm_name=raw_variants[0],
                normalized_name=normalized_name,
                raw_name_variants=raw_variants,
                mention_ids=[mention.mention_id for mention in group],
                source_names=source_names,
                source_urls=source_urls,
                evidence_count=len(group),
            )
        )
    return candidates
