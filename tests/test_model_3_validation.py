import json
from pathlib import Path

from model_3_validation import parse_validation_record, save_validated_records


def test_parse_validation_record_preserves_fields() -> None:
    record = {
        "firm_name": "Zendesk",
        "founding_year": 2007,
        "founded_in_denmark": "true",
        "moved_hq_abroad": "true",
        "sources_founding": ["https://example.com/founding"],
    }
    payload = {
        "record": {
            "validation_label": "true",
            "validation_reason": "The company appears founded in Copenhagen and later run from San Francisco.",
            "exclusion_reason": None,
            "evidence_summary": "Founding in Denmark and later executive HQ abroad are both source-backed.",
            "needs_human_review": True,
        }
    }

    parsed = parse_validation_record(record, payload)

    assert parsed["firm_name"] == "Zendesk"
    assert parsed["validation_label"] == "true"
    assert parsed["validation_reason"].startswith("The company appears founded")
    assert parsed["needs_human_review"] is True
    assert parsed["prompt_version_model3"] == "2026-04-28-model3-v2"


def test_save_validated_records_writes_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "model3" / "validated.jsonl"
    records = [
        {
            "firm_name": "Zendesk",
            "validation_label": "true",
            "validation_reason": "Strong evidence.",
            "exclusion_reason": None,
            "evidence_summary": "Summary.",
            "needs_human_review": True,
            "prompt_version_model3": "2026-04-28-model3-v2",
        }
    ]

    save_validated_records(records, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == records[0]


def test_parse_validation_record_forces_false_for_founding_year_1999() -> None:
    record = {
        "firm_name": "OldCo",
        "founding_year": 1999,
    }
    payload = {
        "record": {
            "validation_label": "true",
            "validation_reason": "Other evidence looked strong.",
            "exclusion_reason": None,
            "evidence_summary": "Summary.",
            "needs_human_review": True,
        }
    }

    parsed = parse_validation_record(record, payload)

    assert parsed["validation_label"] == "false"
    assert parsed["exclusion_reason"] == "Founded before 2000."


def test_parse_validation_record_allows_true_for_founding_year_2000() -> None:
    record = {
        "firm_name": "NewCo",
        "founding_year": 2000,
    }
    payload = {
        "record": {
            "validation_label": "true",
            "validation_reason": "Meets the rule.",
            "exclusion_reason": None,
            "evidence_summary": "Summary.",
            "needs_human_review": False,
        }
    }

    parsed = parse_validation_record(record, payload)

    assert parsed["validation_label"] == "true"


def test_parse_validation_record_unknown_year_stays_unclear_without_post_1999_evidence() -> None:
    record = {
        "firm_name": "UnknownCo",
        "founding_year": None,
        "founding_evidence": "The source ties the firm to Denmark.",
    }
    payload = {
        "record": {
            "validation_label": "false",
            "validation_reason": "Weak evidence overall.",
            "exclusion_reason": "Not enough evidence.",
            "evidence_summary": "Summary.",
            "needs_human_review": True,
        }
    }

    parsed = parse_validation_record(record, payload)

    assert parsed["validation_label"] == "unclear"
    assert "unknown" in parsed["validation_reason"].lower()
