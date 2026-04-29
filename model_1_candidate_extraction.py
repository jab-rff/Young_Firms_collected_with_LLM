"""Model 1 recall-first candidate extraction from deduplicated snowball discoveries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.founding_eligibility import is_post_1998_eligible_or_supported
from src.openai_batch import build_batch_request_item, run_responses_batch
from src.io import save_jsonl
from src.openai_costs import build_cost_record, cost_log_path, sum_cost_records

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

    tqdm.write = print  # type: ignore[attr-defined]

DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "2026-04-28-model1-v3"

SYSTEM_PROMPT = """You are performing recall-first candidate extraction from snowball discovery evidence.

Question:
Could this plausibly be a Danish-founded firm that later moved its own HQ, main office, or main operations abroad?

Rules:
- allow uncertain but plausible candidates
- exclude candidates clearly founded before 1999
- if founding_year is known and less than 1999, do not output the candidate
- if founding_year is unknown, allow uncertain candidates only when other evidence strongly suggests a 1999-or-later startup or young firm
- do not fabricate years, cities, or countries
- Danish founder abroad is not the same as founded in Denmark
- foreign office openings alone are not enough
- acquisition-only cases are not enough unless the focal firm appears to continue with its own HQ or main operations abroad
- return one row for the focal firm candidate described in the input record
- return empty candidates if no plausible focal firm remains

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


def load_discovery_candidates(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_user_prompt(candidate: dict[str, Any]) -> str:
    payload = {
        "task": (
            "Convert this deduplicated snowball discovery record into a recall-first structured candidate "
            "assessment for possible founding in Denmark and possible HQ or main-operations relocation abroad."
        ),
        "snowball_candidate": candidate,
        "output_constraints": {
            "one_row_per_focal_firm": True,
            "keep_uncertain_plausible_candidates": True,
            "empty_candidates_if_none": True,
            "preserve_source_urls": True,
            "founding_year_rule": "Exclude founding_year < 1999. If founding_year is unknown, keep only when evidence strongly suggests a 1999-or-later startup or young firm.",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def candidate_json_schema() -> dict[str, Any]:
    tri = {"type": "string", "enum": ["true", "false", "uncertain"]}
    nullable_string = {"type": ["string", "null"]}
    nullable_integer = {"type": ["integer", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
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
                        "ma_co_occurred",
                        "ma_type",
                        "founding_evidence",
                        "relocation_evidence",
                        "ma_evidence",
                        "reasoning",
                        "sources",
                        "confidence_note",
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
                        "ma_co_occurred": tri,
                        "ma_type": {
                            "type": ["string", "null"],
                            "enum": ["acquisition", "merger", "unknown", None],
                        },
                        "founding_evidence": nullable_string,
                        "relocation_evidence": nullable_string,
                        "ma_evidence": nullable_string,
                        "reasoning": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "confidence_note": {"type": "string"},
                    },
                },
            }
        },
    }


def build_response_request_body(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "model_1_candidate_extraction",
                "strict": True,
                "schema": candidate_json_schema(),
            }
        },
    }


def extract_candidates_from_record(record: dict[str, Any], model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    load_openai_api_key()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Model 1 candidate extraction requires the openai package. Install dependencies with "
            "`pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    response = client.responses.create(**build_response_request_body(record, model_name))
    parsed = json.loads(response.output_text)
    return (
        parse_candidate_cases(record, parsed),
        build_cost_record(
            stage="model1",
            request_kind="candidate_extraction",
            raw_response=response,
            requested_model=model_name,
            metadata={"firm_name": str(record.get("firm_name") or "").strip()},
        ),
    )


def parse_candidate_cases(record: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    fallback_sources = list(record.get("source_urls") or [])
    parsed_cases: list[dict[str, Any]] = []

    for row in payload.get("candidates", []):
        founding_year = row.get("founding_year")
        parsed_case = {
            "firm_name": str(row.get("firm_name") or record.get("firm_name") or "").strip(),
            "founded_in_denmark": _tri_state(row.get("founded_in_denmark")),
            "founding_year": founding_year,
            "founding_city": row.get("founding_city"),
            "founding_country_iso": row.get("founding_country_iso"),
            "moved_hq_abroad": _tri_state(row.get("moved_hq_abroad")),
            "move_year": row.get("move_year"),
            "moved_to_city": row.get("moved_to_city"),
            "moved_to_country_iso": row.get("moved_to_country_iso"),
            "ma_co_occurred": _tri_state(row.get("ma_co_occurred")),
            "ma_type": row.get("ma_type"),
            "founding_evidence": row.get("founding_evidence"),
            "relocation_evidence": row.get("relocation_evidence"),
            "ma_evidence": row.get("ma_evidence"),
            "reasoning": str(row.get("reasoning") or ""),
            "sources": list(row.get("sources") or fallback_sources),
            "confidence_note": str(row.get("confidence_note") or ""),
            "discovery_bucket": record.get("discovery_bucket"),
            "discovery_buckets": list(record.get("discovery_buckets") or []),
            "source_record": record,
            "prompt_version": PROMPT_VERSION,
        }
        if is_post_1998_eligible_or_supported(
            founding_year,
            parsed_case["founding_evidence"],
            parsed_case["reasoning"],
            parsed_case["confidence_note"],
            parsed_case["relocation_evidence"],
            parsed_case["ma_evidence"],
            record.get("short_reason"),
            record.get("possible_founding_location"),
            record.get("possible_abroad_location"),
            record.get("signal_type"),
            record.get("signal_strength"),
        ):
            parsed_cases.append(parsed_case)
    return parsed_cases


def save_candidate_cases(records: list[dict[str, Any]], output_path: Path) -> None:
    save_jsonl(records, output_path)


def run_model_1(
    input_path: Path,
    output_path: Path,
    model_name: str,
    limit: int | None = None,
    batch: bool = False,
) -> list[dict[str, Any]]:
    records_in = load_discovery_candidates(input_path)
    if limit is not None:
        records_in = records_in[:limit]
    records_out: list[dict[str, Any]] = []
    cost_records: list[dict[str, Any]] = []
    if batch and records_in:
        load_openai_api_key()
        from openai import OpenAI

        client = OpenAI()
        request_items = [
            build_batch_request_item(
                custom_id=f"model1-{index}",
                body=build_response_request_body(record, model_name),
            )
            for index, record in enumerate(records_in)
        ]
        responses_by_custom_id, batch_payload = run_responses_batch(client=client, request_items=request_items)
        tqdm.write(f"batch_id={batch_payload.get('id')}")
        for index, record in enumerate(records_in):
            custom_id = f"model1-{index}"
            response_body = responses_by_custom_id[custom_id]
            parsed = json.loads(str(response_body.get("output_text") or ""))
            records_out.extend(parse_candidate_cases(record, parsed))
            cost_records.append(
                build_cost_record(
                    stage="model1",
                    request_kind="candidate_extraction_batch",
                    raw_response=response_body,
                    requested_model=model_name,
                    metadata={"firm_name": str(record.get("firm_name") or "").strip(), "batch": True},
                )
            )
    else:
        for record in tqdm(records_in, total=len(records_in), desc="Model 1", unit="firm"):
            parsed_records, cost_record = extract_candidates_from_record(record=record, model_name=model_name)
            records_out.extend(parsed_records)
            cost_records.append(cost_record)
    save_candidate_cases(records_out, output_path)
    save_jsonl(cost_records, cost_log_path(output_path))
    totals = sum_cost_records(cost_records)
    tqdm.write(f"model_1_input_records={len(records_in)}")
    tqdm.write(f"estimated_cost_usd={totals['estimated_cost_usd']:.6f}")
    tqdm.write(f"api_cost_log_path={cost_log_path(output_path)}")
    return records_out


def _tri_state(value: Any) -> str:
    if value in {"true", "false", "uncertain"}:
        return value
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser(description="Model 1 candidate extraction from deduplicated snowball JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Path to deduplicated snowball JSONL input.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write candidate-case JSONL output.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for small debugging runs.")
    parser.add_argument("--batch", action="store_true", help="Submit requests through the OpenAI Batch API.")
    args = parser.parse_args()

    records = run_model_1(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        limit=args.limit,
        batch=args.batch,
    )
    print(f"prompt_version={PROMPT_VERSION}")
    print(f"candidate_cases={len(records)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
