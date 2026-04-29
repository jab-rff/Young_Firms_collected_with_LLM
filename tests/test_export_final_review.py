import csv
import json
from pathlib import Path

from export_final_review import export_final_review, load_jsonl, prepare_review_row


def test_prepare_review_row_contains_validation_fields() -> None:
    record = {
        "firm_name": "Zendesk",
        "first_legal_entity_name": "Zendesk ApS",
        "validation_label": "true",
        "needs_human_review": True,
        "founded_in_denmark": "true",
        "founding_year": 2007,
        "founding_city": "Copenhagen",
        "founding_country_iso": "DK",
        "moved_hq_abroad": "true",
        "move_year": None,
        "moved_to_city": "San Francisco",
        "moved_to_country_iso": "US",
        "hq_today_city": "San Francisco",
        "hq_today_country_iso": "US",
        "status_today": "active",
        "confidence": "high",
        "validation_reason": "Strong evidence.",
        "exclusion_reason": None,
        "evidence_summary": "Summary.",
        "founding_evidence": "Founded in Copenhagen.",
        "relocation_evidence": "Later headquartered in San Francisco.",
        "ma_evidence": "",
        "relocation_context": "Operational shift abroad.",
        "ma_context": "",
        "uncertainty_note": "",
        "sources_founding": ["https://example.com/founding"],
        "sources_relocation": ["https://example.com/relocation"],
        "sources_ma": [],
        "sources_status_today": ["https://example.com/status"],
    }

    row = prepare_review_row(record)

    assert row["firm_name"] == "Zendesk"
    assert row["validation_label"] == "true"
    assert row["needs_human_review"] == "true"
    assert row["sources_founding"] == "https://example.com/founding"


def test_export_final_review_writes_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "review" / "final.csv"
    records = [
        {
            "firm_name": "Zendesk",
            "first_legal_entity_name": "Zendesk ApS",
            "validation_label": "true",
            "needs_human_review": True,
            "founded_in_denmark": "true",
            "founding_year": 2007,
            "founding_city": "Copenhagen",
            "founding_country_iso": "DK",
            "moved_hq_abroad": "true",
            "move_year": None,
            "moved_to_city": "San Francisco",
            "moved_to_country_iso": "US",
            "hq_today_city": "San Francisco",
            "hq_today_country_iso": "US",
            "status_today": "active",
            "confidence": "high",
            "validation_reason": "Strong evidence.",
            "exclusion_reason": None,
            "evidence_summary": "Summary.",
            "founding_evidence": "Founded in Copenhagen.",
            "relocation_evidence": "Later headquartered in San Francisco.",
            "ma_evidence": "",
            "relocation_context": "Operational shift abroad.",
            "ma_context": "",
            "uncertainty_note": "",
            "sources_founding": ["https://example.com/founding"],
            "sources_relocation": ["https://example.com/relocation"],
            "sources_ma": [],
            "sources_status_today": ["https://example.com/status"],
        }
    ]

    export_final_review(records, output_path)

    with output_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["validation_label"] == "true"
    assert rows[0]["firm_name"] == "Zendesk"


def test_load_jsonl_reads_records(tmp_path: Path) -> None:
    path = tmp_path / "validated.jsonl"
    rows = [{"firm_name": "Zendesk", "validation_label": "true"}]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded = load_jsonl(path)

    assert loaded == rows
