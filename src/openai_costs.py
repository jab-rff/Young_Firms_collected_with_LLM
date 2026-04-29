"""Helpers for estimating OpenAI Responses API costs from saved response payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_PRICING_USD_PER_MILLION: dict[str, dict[str, float | None]] = {
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.4},
    "gpt-5-pro": {"input": 15.0, "cached_input": None, "output": 120.0},
}

WEB_SEARCH_SEARCH_CALL_USD_PER_1K = 10.0


def serialize_openai_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    return {"response_repr": repr(response)}


def normalize_model_name(model_name: str | None) -> str | None:
    text = str(model_name or "").strip()
    if not text:
        return None
    for known_name in sorted(MODEL_PRICING_USD_PER_MILLION, key=len, reverse=True):
        if text == known_name or text.startswith(f"{known_name}-"):
            return known_name
    return text


def estimate_response_cost(raw_response: dict[str, Any], requested_model: str | None = None) -> dict[str, Any]:
    usage = dict(raw_response.get("usage") or {})
    model_name = normalize_model_name(str(raw_response.get("model") or requested_model or ""))
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model_name or "")
    if pricing is None:
        raise ValueError(f"No pricing configured for model: {model_name or requested_model}")

    input_tokens = int(usage.get("input_tokens") or 0)
    cached_input_tokens = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    output_tokens = int(usage.get("output_tokens") or 0)
    web_search_calls = count_web_search_search_calls(raw_response)

    uncached_input_cost = uncached_input_tokens * float(pricing["input"] or 0.0) / 1_000_000
    cached_rate = pricing.get("cached_input")
    cached_input_cost = cached_input_tokens * float(cached_rate or 0.0) / 1_000_000
    output_cost = output_tokens * float(pricing["output"] or 0.0) / 1_000_000
    web_search_cost = web_search_calls * WEB_SEARCH_SEARCH_CALL_USD_PER_1K / 1_000
    total_cost = uncached_input_cost + cached_input_cost + output_cost + web_search_cost

    return {
        "model_pricing_name": model_name,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "web_search_calls": web_search_calls,
        "input_cost_usd": round(uncached_input_cost + cached_input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "web_search_cost_usd": round(web_search_cost, 8),
        "estimated_cost_usd": round(total_cost, 8),
    }


def count_web_search_search_calls(raw_response: dict[str, Any]) -> int:
    calls = 0
    for item in raw_response.get("output") or []:
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        if action.get("type") == "search":
            calls += 1
    return calls


def build_cost_record(
    *,
    stage: str,
    request_kind: str,
    raw_response: Any,
    requested_model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    serialized = serialize_openai_response(raw_response)
    cost = estimate_response_cost(serialized, requested_model=requested_model)
    record = {
        "stage": stage,
        "request_kind": request_kind,
        "response_id": serialized.get("id"),
        "requested_model": requested_model,
        "response_model": serialized.get("model"),
        **cost,
    }
    if metadata:
        record.update(metadata)
    return record


def cost_log_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_api_costs.jsonl")


def sum_cost_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "requests": len(records),
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "web_search_calls": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "web_search_cost_usd": 0.0,
        "estimated_cost_usd": 0.0,
    }
    for record in records:
        totals["input_tokens"] += int(record.get("input_tokens") or 0)
        totals["cached_input_tokens"] += int(record.get("cached_input_tokens") or 0)
        totals["uncached_input_tokens"] += int(record.get("uncached_input_tokens") or 0)
        totals["output_tokens"] += int(record.get("output_tokens") or 0)
        totals["web_search_calls"] += int(record.get("web_search_calls") or 0)
        totals["input_cost_usd"] += float(record.get("input_cost_usd") or 0.0)
        totals["output_cost_usd"] += float(record.get("output_cost_usd") or 0.0)
        totals["web_search_cost_usd"] += float(record.get("web_search_cost_usd") or 0.0)
        totals["estimated_cost_usd"] += float(record.get("estimated_cost_usd") or 0.0)
    for key in ("input_cost_usd", "output_cost_usd", "web_search_cost_usd", "estimated_cost_usd"):
        totals[key] = round(float(totals[key]), 8)
    return totals
