"""Model 2 stricter enrichment and reconciliation from Model 1 candidate JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.io import save_jsonl
from src.normalization import normalize_company_name
from src.openai_batch import build_batch_request_item, extract_json_output_payload, run_responses_batch
from src.openai_costs import build_cost_record, cost_log_path, sum_cost_records

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

    tqdm.write = print  # type: ignore[attr-defined]

DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "2026-04-30-model2-v3"
_ALLOWED_STATUS_TODAY = {"active", "acquired", "merged", "closed", "uncertain"}
_ALLOWED_MA_TYPES = {"acquisition", "merger", "unknown", None}

SYSTEM_PROMPT = """You are performing stricter candidate-level reconciliation for a recall-first company research pipeline.

Use the provided Model 1 candidate record as starting evidence, but verify and reconcile with web search.
The input record will specify an origin_track.

Rules:
- Be conservative and source-grounded.
- Do not fabricate dates, cities, legal names, acquirers, or countries.
- Use null when unknown.
- Do not confuse the two origin tracks.
- If the evidence clearly points to the other track, set origin_track to that track instead of forcing the requested track.
- For in_denmark, Danish founder abroad is not the same as founded in Denmark.
- For in_denmark, foreign office, subsidiary, or sales office is not enough for headquarters relocation.
- For in_denmark, legal redomiciling is not automatically executive headquarters relocation.
- For in_denmark, acquisition-only cases should not be marked as moved_hq_abroad unless there is evidence that the firm's own headquarters or main operations moved abroad.
- For abroad_danish_founders, verify whether the firm was founded outside Denmark and whether at least one founder appears Danish from real sources.
- Prefer official company pages, registries, filings, reputable news, and archived pages.
- Return exactly one enriched record for the input firm.

Return only JSON matching the requested schema."""


def load_openai_api_key(dotenv_path: Path = Path(".env")) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    if dotenv_path.exists():
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != "OPENAI_API_KEY":
                continue
            parsed = value.strip().strip("'").strip('"')
            if parsed:
                os.environ["OPENAI_API_KEY"] = parsed
                return parsed

    raise RuntimeError("OPENAI_API_KEY is required in the environment or a local .env file.")


def load_model_1_candidates(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_user_prompt(candidate: dict[str, Any]) -> str:
    origin_track = str(candidate.get("origin_track") or "in_denmark")
    payload = {
        "origin_track": origin_track,
        "task": _build_task_text(origin_track),
        "candidate_input": candidate,
        "output_constraints": {
            "one_record_only": True,
            "verify_with_web_search": True,
            "use_null_when_unknown": True,
            "preserve_source_urls": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def enrichment_json_schema() -> dict[str, Any]:
    tri = {"type": "string", "enum": ["true", "false", "uncertain"]}
    nullable_string = {"type": ["string", "null"]}
    nullable_integer = {"type": ["integer", "null"]}
    source_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["record"],
        "properties": {
            "record": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "firm_name",
                    "origin_track",
                    "first_legal_entity_name",
                    "founding_date",
                    "founding_year",
                    "founding_city",
                    "founding_country_iso",
                    "founded_in_denmark",
                    "danish_founders_abroad",
                    "moved_hq_abroad",
                    "move_date",
                    "move_year",
                    "moved_to_city",
                    "moved_to_country_iso",
                    "relocation_context",
                    "ma_after_or_during_move",
                    "ma_type",
                    "acquirer",
                    "acq_date",
                    "ma_context",
                    "hq_today_city",
                    "hq_today_country_iso",
                    "status_today",
                    "status_today_context",
                    "sources_founding",
                    "sources_founder_identity",
                    "sources_relocation",
                    "sources_ma",
                    "sources_status_today",
                    "confidence",
                    "uncertainty_note",
                    "founder_danish_context",
                ],
                "properties": {
                    "firm_name": {"type": "string"},
                    "origin_track": {"type": "string", "enum": ["in_denmark", "abroad_danish_founders"]},
                    "first_legal_entity_name": nullable_string,
                    "founding_date": nullable_string,
                    "founding_year": nullable_integer,
                    "founding_city": nullable_string,
                    "founding_country_iso": nullable_string,
                    "founded_in_denmark": tri,
                    "danish_founders_abroad": tri,
                    "moved_hq_abroad": tri,
                    "move_date": nullable_string,
                    "move_year": nullable_integer,
                    "moved_to_city": nullable_string,
                    "moved_to_country_iso": nullable_string,
                    "relocation_context": nullable_string,
                    "ma_after_or_during_move": tri,
                    "ma_type": {
                        "type": ["string", "null"],
                        "enum": ["acquisition", "merger", "unknown", None],
                    },
                    "acquirer": nullable_string,
                    "acq_date": nullable_string,
                    "ma_context": nullable_string,
                    "hq_today_city": nullable_string,
                    "hq_today_country_iso": nullable_string,
                    "status_today": {
                        "type": "string",
                        "enum": ["active", "acquired", "merged", "closed", "uncertain"],
                    },
                    "status_today_context": nullable_string,
                    "sources_founding": source_array,
                    "sources_founder_identity": source_array,
                    "sources_relocation": source_array,
                    "sources_ma": source_array,
                    "sources_status_today": source_array,
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "uncertainty_note": {"type": "string"},
                    "founder_danish_context": nullable_string,
                },
            }
        },
    }


def build_response_request_body(candidate: dict[str, Any], model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(candidate)},
        ],
        "tools": [{"type": "web_search"}],
        "include": ["web_search_call.action.sources"],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "model_2_enrichment",
                "strict": True,
                "schema": enrichment_json_schema(),
            }
        },
    }


def enrich_candidate(candidate: dict[str, Any], model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    load_openai_api_key()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Model 2 enrichment requires the openai package. Install dependencies with "
            "`pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    response = client.responses.create(**build_response_request_body(candidate, model_name))
    parsed = json.loads(response.output_text)
    return (
        parse_enriched_record(candidate, parsed),
        build_cost_record(
            stage="model2",
            request_kind="enrichment",
            raw_response=response,
            requested_model=model_name,
            metadata={"firm_name": str(candidate.get("firm_name") or "").strip()},
        ),
    )


def parse_enriched_record(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    record = payload.get("record") or {}
    fallback_sources = list(candidate.get("sources") or [])
    return {
        "firm_name": str(record.get("firm_name") or candidate.get("firm_name") or "").strip(),
        "origin_track": _normalize_origin_track(record.get("origin_track") or candidate.get("origin_track")),
        "first_legal_entity_name": record.get("first_legal_entity_name"),
        "founding_date": record.get("founding_date"),
        "founding_year": record.get("founding_year"),
        "founding_city": record.get("founding_city"),
        "founding_country_iso": record.get("founding_country_iso"),
        "founded_in_denmark": _tri_state(record.get("founded_in_denmark")),
        "danish_founders_abroad": _tri_state(record.get("danish_founders_abroad")),
        "moved_hq_abroad": _tri_state(record.get("moved_hq_abroad")),
        "move_date": record.get("move_date"),
        "move_year": record.get("move_year"),
        "moved_to_city": record.get("moved_to_city"),
        "moved_to_country_iso": record.get("moved_to_country_iso"),
        "relocation_context": record.get("relocation_context"),
        "ma_after_or_during_move": _tri_state(record.get("ma_after_or_during_move")),
        "ma_type": _normalize_ma_type(record.get("ma_type")),
        "acquirer": record.get("acquirer"),
        "acq_date": record.get("acq_date"),
        "ma_context": record.get("ma_context"),
        "hq_today_city": record.get("hq_today_city"),
        "hq_today_country_iso": record.get("hq_today_country_iso"),
        "status_today": _normalize_status_today(record.get("status_today")),
        "status_today_context": record.get("status_today_context"),
        "sources_founding": list(record.get("sources_founding") or fallback_sources),
        "sources_founder_identity": list(record.get("sources_founder_identity") or fallback_sources),
        "sources_relocation": list(record.get("sources_relocation") or fallback_sources),
        "sources_ma": list(record.get("sources_ma") or fallback_sources),
        "sources_status_today": list(record.get("sources_status_today") or fallback_sources),
        "confidence": _normalize_confidence(record.get("confidence")),
        "uncertainty_note": str(record.get("uncertainty_note") or ""),
        "founder_danish_context": record.get("founder_danish_context"),
        "prompt_version": PROMPT_VERSION,
    }


def save_enriched_records(records: list[dict[str, Any]], output_path: Path) -> None:
    save_jsonl(records, output_path)


def run_model_2(
    input_path: Path,
    output_path: Path,
    model_name: str,
    limit: int | None = None,
    batch: bool = False,
) -> list[dict[str, Any]]:
    candidates = reconcile_model_1_duplicates(load_model_1_candidates(input_path))
    if limit is not None:
        candidates = candidates[:limit]
    records: list[dict[str, Any]] = []
    cost_records: list[dict[str, Any]] = []
    if batch and candidates:
        load_openai_api_key()
        from openai import OpenAI

        client = OpenAI()
        request_items = [
            build_batch_request_item(
                custom_id=f"model2-{index}",
                body=build_response_request_body(candidate, model_name),
            )
            for index, candidate in enumerate(candidates)
        ]
        responses_by_custom_id, _batch_payload = run_responses_batch(client=client, request_items=request_items)
        for index, candidate in enumerate(candidates):
            response_body = responses_by_custom_id[f"model2-{index}"]
            parsed = extract_json_output_payload(response_body)
            records.append(parse_enriched_record(candidate, parsed))
            cost_records.append(
                build_cost_record(
                    stage="model2",
                    request_kind="enrichment_batch",
                    raw_response=response_body,
                    requested_model=model_name,
                    metadata={"firm_name": str(candidate.get("firm_name") or "").strip(), "batch": True},
                )
            )
    else:
        for candidate in tqdm(candidates, total=len(candidates), desc="Model 2", unit="firm"):
            record, cost_record = enrich_candidate(candidate=candidate, model_name=model_name)
            records.append(record)
            cost_records.append(cost_record)
    save_enriched_records(records, output_path)
    save_jsonl(cost_records, cost_log_path(output_path))
    totals = sum_cost_records(cost_records)
    tqdm.write(f"model_2_input_records={len(candidates)}")
    tqdm.write(f"estimated_cost_usd={totals['estimated_cost_usd']:.6f}")
    tqdm.write(f"api_cost_log_path={cost_log_path(output_path)}")
    return records


def reconcile_model_1_duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        firm_name = str(record.get("firm_name") or "").strip()
        normalized = normalize_company_name(firm_name)
        if not normalized:
            continue
        if normalized not in deduped:
            merged = dict(record)
            merged["sources"] = _merge_unique_strings([], list(record.get("sources") or []))
            merged["source_records"] = [record.get("source_record")] if record.get("source_record") else []
            deduped[normalized] = merged
            continue

        current = deduped[normalized]
        current["sources"] = _merge_unique_strings(list(current.get("sources") or []), list(record.get("sources") or []))
        current["source_records"] = list(current.get("source_records") or [])
        if record.get("source_record"):
            current["source_records"].append(record["source_record"])
        for field in (
            "founding_evidence",
            "founder_danish_evidence",
            "relocation_evidence",
            "ma_evidence",
            "reasoning",
            "confidence_note",
        ):
            current[field] = _merge_text(current.get(field), record.get(field))
        current["discovery_buckets"] = _merge_unique_strings(
            list(current.get("discovery_buckets") or []),
            list(record.get("discovery_buckets") or []),
        )
    return list(deduped.values())


def _tri_state(value: Any) -> str:
    if value in {"true", "false", "uncertain"}:
        return value
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "uncertain"


def _normalize_ma_type(value: Any) -> str | None:
    if value in _ALLOWED_MA_TYPES:
        return value
    return "unknown"


def _normalize_status_today(value: Any) -> str:
    if value in _ALLOWED_STATUS_TODAY:
        return str(value)
    return "uncertain"


def _normalize_confidence(value: Any) -> str:
    if value in {"high", "medium", "low"}:
        return str(value)
    return "low"


def _normalize_origin_track(value: Any) -> str:
    if value == "abroad_danish_founders":
        return "abroad_danish_founders"
    return "in_denmark"


def _build_task_text(origin_track: str) -> str:
    if origin_track == "abroad_danish_founders":
        return (
            "Reconcile this candidate into the best conservative structured interpretation of founding abroad by Danish "
            "founders, current status, and any secondary relocation or M&A context using better cross-source evidence."
        )
    return (
        "Reconcile this candidate into the best conservative structured interpretation of founding in Denmark, "
        "headquarters relocation abroad, M&A context, and current status using better cross-source evidence."
    )


def _merge_unique_strings(left: list[Any], right: list[Any]) -> list[str]:
    merged: list[str] = []
    for value in [*left, *right]:
        text = str(value or "").strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def _merge_text(left: Any, right: Any) -> str | None:
    parts = _merge_unique_strings([left], [right])
    if not parts:
        return None
    return " || ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model 2 stricter enrichment from Model 1 candidate JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Path to Model 1 candidate JSONL input.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write enriched candidate JSONL output.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for small debugging runs.")
    parser.add_argument("--batch", action="store_true", help="Submit requests through the OpenAI Batch API.")
    args = parser.parse_args()

    records = run_model_2(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        limit=args.limit,
        batch=args.batch,
    )
    print(f"prompt_version={PROMPT_VERSION}")
    print(f"enriched_records={len(records)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
