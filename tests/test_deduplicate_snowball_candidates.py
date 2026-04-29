import csv
from pathlib import Path

from deduplicate_snowball_candidates import deduplicate_candidates
from src.seed_list import build_exclusion_list, load_seed_firms


def write_known_csv(path: Path) -> None:
    rows = [
        {"name": "Zendesk", "founding_origin": "in Denmark", "industry": "software"},
        {"name": "Unity", "founding_origin": "abroad (Danish founders)", "industry": "gaming"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "founding_origin", "industry"])
        writer.writeheader()
        writer.writerows(rows)


def test_dedup_removes_known_firms(tmp_path: Path) -> None:
    known_path = tmp_path / "known.csv"
    write_known_csv(known_path)
    known_names = build_exclusion_list(load_seed_firms(known_path))

    records = [
        {
            "firm_name": "Zendesk",
            "discovery_bucket": "sector:software",
            "source_urls": ["https://example.com/zendesk"],
            "possible_abroad_location": "San Francisco",
            "signal_type": "explicit_hq_move",
            "signal_strength": "strong",
            "short_reason": "Founded in Copenhagen and later headquartered in San Francisco.",
        },
        {
            "firm_name": "LessKnownCo",
            "discovery_bucket": "sector:software",
            "source_urls": ["https://example.com/lessknownco"],
            "possible_abroad_location": "London",
            "signal_type": "leadership_abroad",
            "signal_strength": "weak",
            "short_reason": "Danish base appears in one source and leadership later appears in London.",
        },
    ]

    deduped = deduplicate_candidates(records, known_names)

    assert len(deduped) == 1
    assert deduped[0]["firm_name"] == "LessKnownCo"


def test_dedup_merges_repeated_candidates() -> None:
    known_names: list[str] = []
    records = [
        {
            "firm_name": "LessKnownCo",
            "discovery_bucket": "sector:software",
            "source_urls": ["https://example.com/1"],
            "possible_abroad_location": "London",
            "signal_type": "leadership_abroad",
            "signal_strength": "weak",
            "short_reason": "Possible move to London from Aarhus.",
        },
        {
            "firm_name": "LessKnownCo ApS",
            "discovery_bucket": "destination:UK",
            "source_urls": ["https://example.com/2"],
            "possible_abroad_location": "London",
            "signal_type": "foreign_principal_office",
            "signal_strength": "medium",
            "short_reason": "Another source points to London as the main base.",
        },
    ]

    deduped = deduplicate_candidates(records, known_names)

    assert len(deduped) == 1
    record = deduped[0]
    assert record["source_urls"] == ["https://example.com/1", "https://example.com/2"]
    assert record["merged_record_count"] == 2
    assert record["discovery_buckets"] == ["sector:software", "destination:UK"]
    assert "Possible move to London from Aarhus." in record["short_reason"]
    assert "Another source points to London as the main base." in record["short_reason"]
