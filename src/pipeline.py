"""Orchestration for the local recall-first pipeline."""

from __future__ import annotations

from pathlib import Path

from src.aggregation import aggregate_mentions
from src.data_models import CandidateFirm, EnrichmentResult, Mention
from src.enrichment_openai import build_enrichment_input_record, enrich_candidate
from src.io import load_retrieved_items, save_jsonl, save_parquet
from src.mention_extraction import extract_mentions


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    run_enrichment: bool = False,
) -> tuple[list[Mention], list[CandidateFirm], list[EnrichmentResult]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieved_items = load_retrieved_items(input_path)
    mentions = extract_mentions(retrieved_items)
    candidates = aggregate_mentions(mentions)

    save_parquet(mentions, output_dir / "mentions.parquet")
    save_parquet(candidates, output_dir / "candidates.parquet")
    save_jsonl(mentions, output_dir / "mentions.jsonl")
    save_jsonl(candidates, output_dir / "candidates.jsonl")

    enrichment_results: list[EnrichmentResult] = []
    if run_enrichment:
        mentions_by_id = {mention.mention_id: mention for mention in mentions}
        enrichment_inputs = []
        for candidate in candidates:
            candidate_mentions = [
                mentions_by_id[mention_id]
                for mention_id in candidate.mention_ids
                if mention_id in mentions_by_id
            ]
            enrichment_inputs.append(build_enrichment_input_record(candidate, candidate_mentions))
            enrichment_results.append(enrich_candidate(candidate, candidate_mentions))
        save_jsonl(enrichment_inputs, output_dir / "enrichment_openai_inputs.jsonl")
        save_jsonl(enrichment_results, output_dir / "enrichment_openai.jsonl")

    return mentions, candidates, enrichment_results
