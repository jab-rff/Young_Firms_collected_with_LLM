from pathlib import Path

from recall.models import CandidateMention
from recall.storage import (
    aggregate_candidate_mentions,
    load_candidate_mentions_jsonl,
    normalize_firm_name,
    save_candidate_mentions_jsonl,
)


def test_normalize_firm_name_is_conservative_and_consistent() -> None:
    assert normalize_firm_name("  Acme   Labs  ") == "acme labs"
    assert normalize_firm_name("Acme, Inc.") == "acme, inc."
    assert normalize_firm_name("\tACME\nLabs\t") == "acme labs"


def test_aggregate_candidate_mentions_merges_by_normalized_name() -> None:
    mentions = [
        CandidateMention(
            firm_name_raw="Acme Labs",
            normalized_name="acme labs",
            source_url="https://example.com/1",
            source_title="Result 1",
            snippet="Acme Labs builds tools.",
            query_used="ai startups berlin",
            retrieval_timestamp="2026-04-20T12:00:00Z",
        ),
        CandidateMention(
            firm_name_raw="ACME Labs",
            normalized_name="acme labs",
            source_url="https://example.com/2",
            source_title="Result 2",
            snippet="ACME Labs raised funding.",
            query_used="berlin startups",
            retrieval_timestamp="2026-04-20T12:05:00Z",
        ),
    ]

    firms = aggregate_candidate_mentions(mentions)

    assert len(firms) == 1
    assert firms[0].normalized_name == "acme labs"
    assert firms[0].raw_name_variants == {"Acme Labs", "ACME Labs"}
    assert firms[0].mention_count() == 2


def test_candidate_mentions_jsonl_roundtrip(tmp_path: Path) -> None:
    mentions = [
        CandidateMention(
            firm_name_raw="Beta Works",
            normalized_name="beta works",
            source_url="https://example.com/beta",
            source_title="Beta Result",
            snippet="Beta Works launched a product.",
            query_used="new companies europe",
            retrieval_timestamp="2026-04-20T13:00:00Z",
        )
    ]

    output_path = tmp_path / "mentions.jsonl"
    save_candidate_mentions_jsonl(mentions, output_path)
    loaded = load_candidate_mentions_jsonl(output_path)

    assert loaded == mentions
