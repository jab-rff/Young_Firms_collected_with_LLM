import json
from pathlib import Path

from model_1_candidate_extraction import parse_candidate_cases, save_candidate_cases


def test_parse_candidate_cases_preserves_schema_and_defaults() -> None:
    source_record = {
        "firm_name": "Issuu",
        "discovery_bucket": "sector:SaaS/software",
        "discovery_buckets": ["sector:SaaS/software", "destination:US"],
        "source_urls": ["https://example.com/issuu"],
        "why_candidate": "Possible Copenhagen origin and later US base.",
    }
    payload = {
        "candidates": [
            {
                "firm_name": "Issuu",
                "founded_in_denmark": "true",
                "founding_year": 2006,
                "founding_city": "Copenhagen",
                "founding_country_iso": "DK",
                "moved_hq_abroad": "uncertain",
                "move_year": None,
                "moved_to_city": None,
                "moved_to_country_iso": None,
                "ma_co_occurred": "false",
                "ma_type": "unknown",
                "founding_evidence": "The discovery evidence points to Copenhagen founding.",
                "relocation_evidence": "The evidence points to later US leadership and operations.",
                "ma_evidence": None,
                "reasoning": "Issuu remains plausible at recall stage.",
                "sources": [],
                "confidence_note": "Relocation evidence is weaker than founding evidence.",
            }
        ]
    }

    records = parse_candidate_cases(source_record, payload)

    assert records == [
        {
            "firm_name": "Issuu",
            "founded_in_denmark": "true",
            "founding_year": 2006,
            "founding_city": "Copenhagen",
            "founding_country_iso": "DK",
            "moved_hq_abroad": "uncertain",
            "move_year": None,
            "moved_to_city": None,
            "moved_to_country_iso": None,
            "ma_co_occurred": "false",
            "ma_type": "unknown",
            "founding_evidence": "The discovery evidence points to Copenhagen founding.",
            "relocation_evidence": "The evidence points to later US leadership and operations.",
            "ma_evidence": None,
            "reasoning": "Issuu remains plausible at recall stage.",
            "sources": ["https://example.com/issuu"],
            "confidence_note": "Relocation evidence is weaker than founding evidence.",
            "discovery_bucket": "sector:SaaS/software",
            "discovery_buckets": ["sector:SaaS/software", "destination:US"],
            "source_record": source_record,
            "prompt_version": "2026-04-28-model1-v3",
        }
    ]


def test_save_candidate_cases_writes_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "model1" / "candidates.jsonl"
    records = [
        {
            "firm_name": "FJ Industries",
            "founded_in_denmark": "true",
            "founding_year": None,
            "founding_city": None,
            "founding_country_iso": "DK",
            "moved_hq_abroad": "uncertain",
            "move_year": None,
            "moved_to_city": None,
            "moved_to_country_iso": None,
            "ma_co_occurred": "uncertain",
            "ma_type": "unknown",
            "founding_evidence": "The source ties FJ Industries to Denmark.",
            "relocation_evidence": None,
            "ma_evidence": None,
            "reasoning": "FJ Industries is the focal firm candidate.",
            "sources": ["https://example.com/fj-industries"],
            "confidence_note": "Relocation evidence is incomplete.",
            "discovery_bucket": "sector:design",
            "discovery_buckets": ["sector:design"],
            "source_record": {"firm_name": "FJ Industries"},
            "prompt_version": "2026-04-28-model1-v3",
        }
    ]

    save_candidate_cases(records, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == records[0]


def test_parse_candidate_cases_excludes_founding_year_1999() -> None:
    source_record = {
        "firm_name": "OldCo",
        "discovery_bucket": "sector:design",
        "discovery_buckets": ["sector:design"],
        "source_urls": ["https://example.com/oldco"],
        "why_candidate": "Possible relocation story.",
    }
    payload = {
        "candidates": [
            {
                "firm_name": "OldCo",
                "founded_in_denmark": "true",
                "founding_year": 1999,
                "founding_city": "Copenhagen",
                "founding_country_iso": "DK",
                "moved_hq_abroad": "true",
                "move_year": 2010,
                "moved_to_city": "London",
                "moved_to_country_iso": "UK",
                "ma_co_occurred": "false",
                "ma_type": "unknown",
                "founding_evidence": "Founded in Copenhagen in 1999.",
                "relocation_evidence": "Later operated from London.",
                "ma_evidence": None,
                "reasoning": "Possible fit except for date.",
                "sources": [],
                "confidence_note": "",
            }
        ]
    }

    assert parse_candidate_cases(source_record, payload) == []


def test_parse_candidate_cases_keeps_founding_year_2000() -> None:
    source_record = {
        "firm_name": "NewCo",
        "discovery_bucket": "sector:design",
        "discovery_buckets": ["sector:design"],
        "source_urls": ["https://example.com/newco"],
        "why_candidate": "Possible relocation story.",
    }
    payload = {
        "candidates": [
            {
                "firm_name": "NewCo",
                "founded_in_denmark": "true",
                "founding_year": 2000,
                "founding_city": "Copenhagen",
                "founding_country_iso": "DK",
                "moved_hq_abroad": "true",
                "move_year": 2010,
                "moved_to_city": "London",
                "moved_to_country_iso": "UK",
                "ma_co_occurred": "false",
                "ma_type": "unknown",
                "founding_evidence": "Founded in Copenhagen in 2000.",
                "relocation_evidence": "Later operated from London.",
                "ma_evidence": None,
                "reasoning": "Plausible fit.",
                "sources": [],
                "confidence_note": "",
            }
        ]
    }

    records = parse_candidate_cases(source_record, payload)
    assert len(records) == 1
    assert records[0]["firm_name"] == "NewCo"


def test_parse_candidate_cases_excludes_unknown_year_without_post_1999_support() -> None:
    source_record = {
        "firm_name": "UnknownCo",
        "discovery_bucket": "sector:design",
        "discovery_buckets": ["sector:design"],
        "source_urls": ["https://example.com/unknownco"],
        "why_candidate": "Possible relocation story.",
        "founding_denmark_evidence": "A Danish company with later London operations.",
        "uncertainty_note": "",
    }
    payload = {
        "candidates": [
            {
                "firm_name": "UnknownCo",
                "founded_in_denmark": "uncertain",
                "founding_year": None,
                "founding_city": None,
                "founding_country_iso": "DK",
                "moved_hq_abroad": "uncertain",
                "move_year": None,
                "moved_to_city": None,
                "moved_to_country_iso": None,
                "ma_co_occurred": "false",
                "ma_type": "unknown",
                "founding_evidence": "The source ties the firm to Denmark.",
                "relocation_evidence": "The source ties the firm to London.",
                "ma_evidence": None,
                "reasoning": "Plausible but sparse.",
                "sources": [],
                "confidence_note": "",
            }
        ]
    }

    assert parse_candidate_cases(source_record, payload) == []
