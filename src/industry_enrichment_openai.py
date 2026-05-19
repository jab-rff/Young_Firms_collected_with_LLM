"""Explicit OpenAI-backed industry enrichment for firm-level CSV rows."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from src.openai_costs import serialize_openai_response

PROMPT_VERSION = "2026-05-18-industry-v1"
DEFAULT_MODEL = "gpt-5-mini"

ALLOWED_INDUSTRIES = [
    "software",
    "fintech",
    "biotech",
    "medtech",
    "gaming",
    "cleantech",
    "logistics",
    "retail/consumer",
    "hospitality",
    "design",
    "healthcare",
    "media",
    "education",
    "consulting/services",
    "hardware",
    "industrial",
    "real estate",
    "food/agriculture",
    "other",
    "unclear",
]

SYSTEM_PROMPT = """You are classifying the primary industry of one company.

Use web evidence to identify the firm and assign exactly one industry label from
the allowed list.

Rules:
- identify the specific firm first; use row context to disambiguate
- prefer the firm's primary operating industry, not an investor category
- use the narrowest allowed label that fits
- if the firm is difficult to identify, return "unclear"
- do not invent facts or sources
- return only JSON matching the schema"""


def classify_firm_industry(
    row: dict[str, Any],
    *,
    model_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call OpenAI for one row and return parsed + raw serialized response."""
    selected_model = model_name or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI industry enrichment requires the openai package. Install dependencies "
            "with `pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    response = client.responses.create(
        model=selected_model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        text={
            "format": {
                "type": "json_schema",
                "name": "firm_industry_classification",
                "strict": True,
                "schema": _json_schema(),
            }
        },
    )

    raw_text = response.output_text
    parsed_payload = json.loads(raw_text)
    created_at = datetime.now(timezone.utc).isoformat()
    result = {
        "firm": str(row.get("firm") or "").strip(),
        "industry": str(parsed_payload.get("industry") or "").strip(),
        "resolved_name": str(parsed_payload.get("resolved_name") or "").strip(),
        "confidence": str(parsed_payload.get("confidence") or "").strip(),
        "reasoning": str(parsed_payload.get("reasoning") or "").strip(),
        "sources": list(parsed_payload.get("sources") or []),
        "model_name": selected_model,
        "prompt_version": PROMPT_VERSION,
        "created_at": created_at,
    }
    raw_record = {
        "firm": str(row.get("firm") or "").strip(),
        "model_name": selected_model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(row),
        "raw_response_text": raw_text,
        "response": serialize_openai_response(response),
        "created_at": created_at,
    }
    return result, raw_record


def build_input_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "firm": str(row.get("firm") or "").strip(),
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(row),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_user_prompt(row: dict[str, Any]) -> str:
    payload = {
        "task": "Identify the firm and classify its primary industry.",
        "allowed_industries": ALLOWED_INDUSTRIES,
        "row_context": {
            "firm": str(row.get("firm") or "").strip(),
            "location_today_country": str(row.get("location_today_country") or "").strip(),
            "location_today_city": str(row.get("location_today_city") or "").strip(),
            "real_move_to_country": str(row.get("real_move_to_country") or "").strip(),
            "real_move_to_city": str(row.get("real_move_to_city") or "").strip(),
            "status_today_manual": str(row.get("status_today_manual") or "").strip(),
            "founding_year": str(row.get("founding_year") or "").strip(),
            "source_total": str(row.get("source_total") or "").strip(),
            "today_source": str(row.get("today_source") or "").strip(),
            "additional_comment": str(row.get("additional_comment") or "").strip(),
            "comment_final": str(row.get("comment_final") or "").strip(),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["resolved_name", "industry", "confidence", "reasoning", "sources"],
        "properties": {
            "resolved_name": {"type": "string"},
            "industry": {"type": "string", "enum": ALLOWED_INDUSTRIES},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reasoning": {"type": "string"},
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
        },
    }
