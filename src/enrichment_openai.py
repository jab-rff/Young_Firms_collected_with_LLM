"""Optional OpenAI enrichment for candidate-level structured signals."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from src.data_models import CandidateFirm, EnrichmentResult, Mention

PROMPT_VERSION = "2026-04-21-v1"
DEFAULT_MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """You are extracting structured research signals about a company.
Be conservative. If unsure, return null or 'uncertain'.
Do NOT guess. Only use evidence or highly plausible inference.

Denmark is the only relevant founding country for classification.
"Danish founders" is NOT sufficient for "founded in Denmark".
Opening offices abroad is not the same as headquarters relocation.
Legal registration change is not the same as executive headquarters relocation.

Return only JSON matching the requested schema."""


def enrich_candidate(
    candidate: CandidateFirm,
    mentions: list[Mention],
    model_name: str | None = None,
) -> EnrichmentResult:
    """Call OpenAI for one candidate and return a transparent structured result."""
    selected_model = model_name or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompt = build_user_prompt(candidate, mentions)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI enrichment requires the openai package. Install dependencies with "
            "`pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    response = client.responses.create(
        model=selected_model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "company_enrichment",
                "strict": True,
                "schema": _json_schema(),
            },
        },
    )

    raw_response = response.output_text
    parsed = json.loads(raw_response)
    return _result_from_model_json(
        parsed=parsed,
        candidate=candidate,
        model_name=selected_model,
        raw_response=raw_response,
    )


def build_user_prompt(candidate: CandidateFirm, mentions: list[Mention]) -> str:
    evidence = [
        {
            "mention_id": mention.mention_id,
            "source_name": mention.source_name,
            "url": mention.url,
            "title": mention.title,
            "query_text": mention.query_text,
            "evidence_text": mention.evidence_text,
        }
        for mention in mentions
    ]
    payload = {
        "candidate": asdict(candidate),
        "evidence_mentions": evidence,
        "task": (
            "Extract recall-stage signals about founding in Denmark, executive HQ "
            "relocation abroad, and whether relocation co-occurred with M&A."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_enrichment_input_record(candidate: CandidateFirm, mentions: list[Mention]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "firm_name": candidate.firm_name,
        "mention_ids": [mention.mention_id for mention in mentions],
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(candidate, mentions),
        "sources": sorted({mention.url or mention.source_name for mention in mentions}),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _result_from_model_json(
    parsed: dict[str, Any],
    candidate: CandidateFirm,
    model_name: str,
    raw_response: str,
) -> EnrichmentResult:
    return EnrichmentResult(
        candidate_id=candidate.candidate_id,
        firm_name=str(parsed.get("firm_name") or candidate.firm_name),
        founded_in_denmark=_tri_state(parsed.get("founded_in_denmark")),
        founding_year=parsed.get("founding_year"),
        founding_city=parsed.get("founding_city"),
        founding_country_iso=parsed.get("founding_country_iso"),
        moved_hq_abroad=_tri_state(parsed.get("moved_hq_abroad")),
        move_year=parsed.get("move_year"),
        moved_to_city=parsed.get("moved_to_city"),
        moved_to_country_iso=parsed.get("moved_to_country_iso"),
        relocation_context=parsed.get("relocation_context"),
        co_occured_with_ma=_tri_state(parsed.get("co_occured_with_ma")),
        ma_type=parsed.get("ma_type"),
        reasoning=str(parsed.get("reasoning") or ""),
        sources=list(parsed.get("sources") or []),
        model_name=model_name,
        prompt_version=PROMPT_VERSION,
        raw_response=raw_response,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _tri_state(value: Any) -> str:
    if value in {"true", "false", "uncertain"}:
        return value
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "uncertain"


def _json_schema() -> dict[str, Any]:
    tri = {"type": "string", "enum": ["true", "false", "uncertain"]}
    nullable_string = {"type": ["string", "null"]}
    nullable_integer = {"type": ["integer", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "firm_name",
            "founded_in_denmark",
            "founding_year",
            "founding_city",
            "founding_country_iso",
            "moved_hq_abroad",
            "move_year",
            "moved_to_city",
            "moved_to_country_iso",
            "relocation_context",
            "co_occured_with_ma",
            "ma_type",
            "reasoning",
            "sources",
        ],
        "properties": {
            "firm_name": {"type": "string"},
            "founded_in_denmark": tri,
            "founding_year": nullable_integer,
            "founding_city": nullable_string,
            "founding_country_iso": nullable_string,
            "moved_hq_abroad": tri,
            "move_year": nullable_integer,
            "moved_to_city": nullable_string,
            "moved_to_country_iso": nullable_string,
            "relocation_context": nullable_string,
            "co_occured_with_ma": tri,
            "ma_type": {
                "type": ["string", "null"],
                "enum": ["acquisition", "merger", "unknown", None],
            },
            "reasoning": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
    }
