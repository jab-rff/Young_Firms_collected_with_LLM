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
            "why_candidate": "Known already.",
            "founding_denmark_evidence": "Founded in Copenhagen.",
            "abroad_hq_evidence": "HQ in San Francisco.",
            "uncertainty_note": "",
            "ma_context": None,
        },
        {
            "firm_name": "LessKnownCo",
            "discovery_bucket": "sector:software",
            "source_urls": ["https://example.com/lessknownco"],
            "why_candidate": "Possible move.",
            "founding_denmark_evidence": "Danish base appears in source.",
            "abroad_hq_evidence": "Leadership later appears in London.",
            "uncertainty_note": "Still uncertain.",
            "ma_context": None,
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
            "why_candidate": "Possible move to London.",
            "founding_denmark_evidence": "Founded in Aarhus.",
            "abroad_hq_evidence": "Leadership later in London.",
            "uncertainty_note": "Needs confirmation.",
            "ma_context": None,
        },
        {
            "firm_name": "LessKnownCo ApS",
            "discovery_bucket": "destination:UK",
            "source_urls": ["https://example.com/2"],
            "why_candidate": "Another source points to London HQ.",
            "founding_denmark_evidence": "Danish registration mentioned.",
            "abroad_hq_evidence": "Company page shows London office as main base.",
            "uncertainty_note": "May be operations rather than formal HQ.",
            "ma_context": "No obvious M&A context.",
        },
    ]

    deduped = deduplicate_candidates(records, known_names)

    assert len(deduped) == 1
    record = deduped[0]
    assert record["source_urls"] == ["https://example.com/1", "https://example.com/2"]
    assert record["merged_record_count"] == 2
    assert record["discovery_buckets"] == ["sector:software", "destination:UK"]
    assert "Possible move to London." in record["why_candidate"]
    assert "Another source points to London HQ." in record["why_candidate"]
