from __future__ import annotations

import csv
from pathlib import Path

from fill_dk_founders_industry import enrich_industries, should_process_row


def _write_input_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["firm", "industry", "founding_year", "location_today_country"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "firm": "Access People",
                "industry": "",
                "founding_year": "2006",
                "location_today_country": "CN",
            }
        )
        writer.writerow(
            {
                "firm": "Anine Bing",
                "industry": "retail/consumer",
                "founding_year": "2012",
                "location_today_country": "US",
            }
        )


def test_should_process_row_respects_overwrite() -> None:
    assert should_process_row({"firm": "TestCo", "industry": ""}, overwrite=False) is True
    assert should_process_row({"firm": "TestCo", "industry": "software"}, overwrite=False) is False
    assert should_process_row({"firm": "TestCo", "industry": "software"}, overwrite=True) is True


def test_enrich_industries_updates_missing_rows_and_writes_logs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    _write_input_csv(input_path)

    monkeypatch.setattr("fill_dk_founders_industry.load_openai_api_key", lambda: "test-key")

    def fake_classify(row: dict[str, str], *, model_name: str | None = None):
        assert row["firm"] == "Access People"
        return (
            {
                "firm": row["firm"],
                "industry": "software",
                "resolved_name": "Access People",
                "confidence": "medium",
                "reasoning": "Mocked result.",
                "sources": ["https://example.com/access-people"],
                "model_name": model_name or "gpt-5-mini",
                "prompt_version": "test",
                "created_at": "2026-05-18T00:00:00+00:00",
            },
            {
                "firm": row["firm"],
                "model_name": model_name or "gpt-5-mini",
                "prompt_version": "test",
                "system_prompt": "test",
                "user_prompt": "test",
                "raw_response_text": "{}",
                "response": {"model": model_name or "gpt-5-mini", "usage": {}, "output": []},
                "created_at": "2026-05-18T00:00:00+00:00",
            },
        )

    monkeypatch.setattr("fill_dk_founders_industry.classify_firm_industry", fake_classify)

    summary = enrich_industries(
        input_path=input_path,
        output_path=output_path,
        model_name="gpt-5-mini",
    )

    assert summary["processed"] == 1

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["industry"] == "software"
    assert rows[1]["industry"] == "retail/consumer"

    assert output_path.with_name("output_industry_enrichment.jsonl").exists()
    assert output_path.with_name("output_industry_raw_responses.jsonl").exists()
    assert output_path.with_name("output_industry_inputs.jsonl").exists()
    assert output_path.with_name("output_industry_api_costs.jsonl").exists()
