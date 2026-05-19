import json
from pathlib import Path

from src.openai_costs import build_cost_record, cost_log_path, estimate_response_cost, sum_cost_records


def test_estimate_response_cost_counts_cached_tokens_and_web_search_calls() -> None:
    raw_response = {
        "model": "gpt-5-mini-2025-08-07",
        "usage": {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 200},
            "output_tokens": 500,
        },
        "output": [
            {"type": "web_search_call", "action": {"type": "search"}},
            {"type": "web_search_call", "action": {"type": "open_page"}},
            {"type": "message"},
        ],
    }

    cost = estimate_response_cost(raw_response)

    assert cost["model_pricing_name"] == "gpt-5-mini"
    assert cost["cached_input_tokens"] == 200
    assert cost["uncached_input_tokens"] == 800
    assert cost["output_tokens"] == 500
    assert cost["web_search_calls"] == 1
    assert cost["input_cost_usd"] == 0.000205
    assert cost["output_cost_usd"] == 0.001
    assert cost["web_search_cost_usd"] == 0.01
    assert cost["estimated_cost_usd"] == 0.011205


def test_build_cost_record_and_sum_cost_records() -> None:
    first = build_cost_record(
        stage="discovery",
        request_kind="initial",
        raw_response={
            "id": "resp_1",
            "model": "gpt-5-mini-2025-08-07",
            "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 50},
            "output": [],
        },
        requested_model="gpt-5-mini",
    )
    second = build_cost_record(
        stage="model2",
        request_kind="enrichment",
        raw_response={
            "id": "resp_2",
            "model": "gpt-5-mini-2025-08-07",
            "usage": {"input_tokens": 200, "input_tokens_details": {"cached_tokens": 100}, "output_tokens": 25},
            "output": [{"type": "web_search_call", "action": {"type": "search"}}],
        },
        requested_model="gpt-5-mini",
    )

    totals = sum_cost_records([first, second])

    assert first["response_id"] == "resp_1"
    assert totals["requests"] == 2
    assert totals["input_tokens"] == 300
    assert totals["cached_input_tokens"] == 100
    assert totals["output_tokens"] == 75
    assert totals["web_search_calls"] == 1
    assert totals["estimated_cost_usd"] > 0.0


def test_cost_log_path_uses_output_stem(tmp_path: Path) -> None:
    output_path = tmp_path / "data" / "model1" / "round_001.jsonl"
    expected = tmp_path / "data" / "model1" / "round_001_api_costs.jsonl"

    assert cost_log_path(output_path) == expected
