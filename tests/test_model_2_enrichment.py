import json
from pathlib import Path

import pytest

from model_2_enrichment import parse_enriched_record, save_enriched_records


@pytest.fixture
def model1_candidates() -> list[dict[str, object]]:
    return [
        {
            "firm_name": "Zendesk",
            "founded_in_denmark": "true",
            "founding_year": 2007,
            "founding_city": "Copenhagen",
            "founding_country_iso": "DK",
            "moved_hq_abroad": "true",
            "move_year": None,
            "moved_to_city": "San Francisco",
            "moved_to_country_iso": "US",
            "ma_co_occurred": "false",
            "ma_type": "unknown",
            "founding_evidence": "Founded in Copenhagen.",
            "relocation_evidence": "Now headquartered in San Francisco.",
            "ma_evidence": None,
            "reasoning": "Zendesk is a focal firm candidate.",
            "sources": ["https://example.com/zendesk"],
            "confidence_note": "Relocation is plausible.",
        },
        {
            "firm_name": "Just Eat",
            "founded_in_denmark": "uncertain",
            "founding_year": None,
            "founding_city": None,
            "founding_country_iso": None,
            "moved_hq_abroad": "uncertain",
            "move_year": None,
            "moved_to_city": None,
            "moved_to_country_iso": None,
            "ma_co_occurred": "true",
            "ma_type": "merger",
            "founding_evidence": None,
            "relocation_evidence": None,
            "ma_evidence": "Merger context appears in sources.",
            "reasoning": "Requires stricter reconciliation.",
            "sources": ["https://example.com/just-eat"],
            "confidence_note": "",
        },
        {
            "firm_name": "Endomondo",
            "founded_in_denmark": "true",
            "founding_year": None,
            "founding_city": "Copenhagen",
            "founding_country_iso": "DK",
            "moved_hq_abroad": "false",
            "move_year": None,
            "moved_to_city": None,
            "moved_to_country_iso": None,
            "ma_co_occurred": "true",
            "ma_type": "acquisition",
            "founding_evidence": "Danish origin is explicit.",
            "relocation_evidence": None,
            "ma_evidence": "Acquisition context is strong.",
            "reasoning": "Acquisition-only evidence should stay strict.",
            "sources": ["https://example.com/endomondo"],
            "confidence_note": "",
        },
    ]


def test_parse_enriched_record_preserves_schema_and_prompt_version(model1_candidates: list[dict[str, object]]) -> None:
    candidate = model1_candidates[0]
    payload = {
        "record": {
            "firm_name": "Zendesk",
            "first_legal_entity_name": "Zendesk ApS",
            "founding_date": None,
            "founding_year": 2007,
            "founding_city": "Copenhagen",
            "founding_country_iso": "DK",
            "founded_in_denmark": "true",
            "moved_hq_abroad": "true",
            "move_date": None,
            "move_year": None,
            "moved_to_city": "San Francisco",
            "moved_to_country_iso": "US",
            "relocation_context": "The company later operated from San Francisco.",
            "ma_after_or_during_move": "false",
            "ma_type": "unknown",
            "acquirer": None,
            "acq_date": None,
            "ma_context": None,
            "hq_today_city": "San Francisco",
            "hq_today_country_iso": "US",
            "status_today": "active",
            "status_today_context": "The firm still operates independently.",
            "sources_founding": ["https://example.com/zendesk/founding"],
            "sources_relocation": ["https://example.com/zendesk/hq"],
            "sources_ma": [],
            "sources_status_today": ["https://example.com/zendesk/status"],
            "confidence": "high",
            "uncertainty_note": "",
        }
    }

    record = parse_enriched_record(candidate, payload)

    assert record["firm_name"] == "Zendesk"
    assert record["first_legal_entity_name"] == "Zendesk ApS"
    assert record["founding_year"] == 2007
    assert record["founded_in_denmark"] == "true"
    assert record["moved_hq_abroad"] == "true"
    assert record["status_today"] == "active"
    assert record["sources_founding"] == ["https://example.com/zendesk/founding"]
    assert record["sources_relocation"] == ["https://example.com/zendesk/hq"]
    assert record["prompt_version"] == "2026-04-28-model2-v2"


def test_parse_enriched_record_keeps_nulls_and_fallback_sources(model1_candidates: list[dict[str, object]]) -> None:
    candidate = model1_candidates[1]
    payload = {
        "record": {
            "firm_name": "Just Eat",
            "first_legal_entity_name": None,
            "founding_date": None,
            "founding_year": None,
            "founding_city": None,
            "founding_country_iso": None,
            "founded_in_denmark": "uncertain",
            "moved_hq_abroad": "uncertain",
            "move_date": None,
            "move_year": None,
            "moved_to_city": None,
            "moved_to_country_iso": None,
            "relocation_context": None,
            "ma_after_or_during_move": "true",
            "ma_type": None,
            "acquirer": None,
            "acq_date": None,
            "ma_context": None,
            "hq_today_city": None,
            "hq_today_country_iso": None,
            "status_today": "uncertain",
            "status_today_context": None,
            "sources_founding": [],
            "sources_relocation": [],
            "sources_ma": [],
            "sources_status_today": [],
            "confidence": "medium",
            "uncertainty_note": "Evidence is mixed across sources.",
        }
    }

    record = parse_enriched_record(candidate, payload)

    assert record["first_legal_entity_name"] is None
    assert record["founding_date"] is None
    assert record["move_date"] is None
    assert record["sources_founding"] == ["https://example.com/just-eat"]
    assert record["sources_relocation"] == ["https://example.com/just-eat"]
    assert record["sources_ma"] == ["https://example.com/just-eat"]
    assert record["sources_status_today"] == ["https://example.com/just-eat"]
    assert record["confidence"] == "medium"


def test_parse_enriched_record_normalizes_status_today_and_ma_type(model1_candidates: list[dict[str, object]]) -> None:
    candidate = model1_candidates[2]
    payload = {
        "record": {
            "firm_name": "Endomondo",
            "first_legal_entity_name": None,
            "founding_date": None,
            "founding_year": None,
            "founding_city": "Copenhagen",
            "founding_country_iso": "DK",
            "founded_in_denmark": True,
            "moved_hq_abroad": False,
            "move_date": None,
            "move_year": None,
            "moved_to_city": None,
            "moved_to_country_iso": None,
            "relocation_context": None,
            "ma_after_or_during_move": "true",
            "ma_type": "takeover",
            "acquirer": "Under Armour",
            "acq_date": None,
            "ma_context": "Acquisition context is explicit.",
            "hq_today_city": None,
            "hq_today_country_iso": None,
            "status_today": "defunct",
            "status_today_context": "The product appears discontinued.",
            "sources_founding": ["https://example.com/endomondo/founding"],
            "sources_relocation": [],
            "sources_ma": ["https://example.com/endomondo/ma"],
            "sources_status_today": ["https://example.com/endomondo/status"],
            "confidence": "invalid",
            "uncertainty_note": "Status naming required normalization.",
        }
    }

    record = parse_enriched_record(candidate, payload)

    assert record["founded_in_denmark"] == "true"
    assert record["moved_hq_abroad"] == "false"
    assert record["ma_type"] == "unknown"
    assert record["status_today"] == "uncertain"
    assert record["confidence"] == "low"


def test_save_enriched_records_writes_jsonl_with_nulls(tmp_path: Path) -> None:
    output_path = tmp_path / "model2" / "enriched.jsonl"
    records = [
        {
            "firm_name": "Zendesk",
            "first_legal_entity_name": None,
            "founding_date": None,
            "founding_year": 2007,
            "founding_city": "Copenhagen",
            "founding_country_iso": "DK",
            "founded_in_denmark": "true",
            "moved_hq_abroad": "true",
            "move_date": None,
            "move_year": None,
            "moved_to_city": "San Francisco",
            "moved_to_country_iso": "US",
            "relocation_context": None,
            "ma_after_or_during_move": "false",
            "ma_type": "unknown",
            "acquirer": None,
            "acq_date": None,
            "ma_context": None,
            "hq_today_city": "San Francisco",
            "hq_today_country_iso": "US",
            "status_today": "active",
            "status_today_context": None,
            "sources_founding": ["https://example.com/zendesk/founding"],
            "sources_relocation": ["https://example.com/zendesk/hq"],
            "sources_ma": [],
            "sources_status_today": ["https://example.com/zendesk/status"],
            "confidence": "high",
            "uncertainty_note": "",
            "prompt_version": "2026-04-27-model2-v1",
        }
    ]

    save_enriched_records(records, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == records[0]
