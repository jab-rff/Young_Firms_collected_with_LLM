"""Filter true reviewed rows and rerun Danish founder name + LinkedIn enrichment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.openai_costs import build_cost_record, cost_log_path, sum_cost_records

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

    tqdm.write = print  # type: ignore[attr-defined]


DEFAULT_INPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_unique_firms_with_diffbot_review.csv")
DEFAULT_OUTPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_unique_firms_with_diffbot_review_true_founders_linkedin.csv")
DEFAULT_RAW_OUTPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_unique_firms_with_diffbot_review_true_founders_linkedin_raw.jsonl")
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_ORIGIN_TRACK = "abroad_danish_founders"
PROMPT_VERSION = "2026-05-07-true-review-founder-linkedin-v1"

SYSTEM_PROMPT = """You extract Danish founder names and LinkedIn URLs from a reviewed firm record.

Rules:
- Use web search.
- Focus only on founders with Danish identity/background relevant to the row's validated case.
- Return up to 5 founders, ordered by confidence and relevance.
- Normalize names to plain person names only.
- Do not include organizations, publications, descriptors, or evidence fragments as names.
- Prefer the firm LinkedIn page in the form https://www.linkedin.com/company/... when available.
- Prefer exact LinkedIn profile URLs for people in the form https://www.linkedin.com/in/... or country-subdomain equivalent.
- Return null for a founder's LinkedIn URL when uncertain.
- Be conservative. Do not invent founders or URLs.

Return only JSON matching the schema."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun founder name + LinkedIn enrichment for validation_label=true rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--origin-track", default=DEFAULT_ORIGIN_TRACK)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.input).where(pd.notna, "").to_dict(orient="records")
    rows = [
        row
        for row in rows
        if str(row.get("validation_label") or "").strip().casefold() == "true"
        and str(row.get("origin_track") or "").strip().casefold() == args.origin_track.strip().casefold()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    enriched_rows = load_existing_output_rows(args.output)
    processed_keys = {make_row_key(row) for row in enriched_rows}
    cost_records = load_jsonl_records(cost_log_path(args.output))
    pending_rows = [row for row in rows if make_row_key(row) not in processed_keys]

    tqdm.write(f"filtered_true_rows={len(rows)}")
    tqdm.write(f"origin_track_filter={args.origin_track}")
    if enriched_rows:
        tqdm.write(f"resume_detected_existing_rows={len(enriched_rows)}")
    tqdm.write(f"pending_rows={len(pending_rows)}")

    for row in tqdm(pending_rows, total=len(pending_rows), desc="True Founder LinkedIn Rerun", unit="firm"):
        enriched_row, raw_record, cost_record = enrich_row_with_retries(
            row=row,
            model_name=args.model,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        enriched_rows.append(enriched_row)
        cost_records.append(cost_record)
        append_jsonl_record(raw_record, args.raw_output)
        append_jsonl_record(cost_record, cost_log_path(args.output))
        write_csv(enriched_rows, args.output)

    write_csv(enriched_rows, args.output)
    totals = sum_cost_records(cost_records)
    tqdm.write(f"rows={len(enriched_rows)}")
    tqdm.write(f"estimated_cost_usd={totals['estimated_cost_usd']:.6f}")
    tqdm.write(f"output_path={args.output}")
    tqdm.write(f"raw_output_path={args.raw_output}")


def enrich_row(row: dict[str, Any], model_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    load_openai_api_key()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Founder LinkedIn enrichment requires the openai package. Install dependencies with "
            "`pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    request_body = build_response_request_body(row, model_name)
    response = client.responses.create(**request_body)
    parsed = json.loads(response.output_text)
    enriched = apply_founder_result(row, parsed)
    raw_record = {
        "firm_name": str(row.get("firm_name") or ""),
        "source": str(row.get("source") or ""),
        "prompt_version": PROMPT_VERSION,
        "request": request_body,
        "response": response.model_dump(mode="json") if hasattr(response, "model_dump") else {"repr": repr(response)},
        "parsed": parsed,
    }
    cost_record = build_cost_record(
        stage="true_review_founder_linkedin_rerun",
        request_kind="web_search",
        raw_response=response,
        requested_model=model_name,
        metadata={"firm_name": str(row.get("firm_name") or ""), "source": str(row.get("source") or "")},
    )
    return enriched, raw_record, cost_record


def enrich_row_with_retries(
    row: dict[str, Any],
    model_name: str,
    max_retries: int,
    retry_delay_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    total_attempts = max(1, max_retries + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            return enrich_row(row, model_name=model_name)
        except Exception as exc:
            last_error = exc
            if not is_retryable_error(exc) or attempt >= total_attempts:
                raise
            firm_name = str(row.get("firm_name") or "").strip()
            delay = retry_delay_seconds * attempt
            tqdm.write(
                f"retrying_firm={firm_name or '<missing>'} attempt={attempt}/{total_attempts - 1} "
                f"delay_seconds={delay:.1f} error={type(exc).__name__}"
            )
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry loop exited unexpectedly without a result or captured exception.")


def build_response_request_body(row: dict[str, Any], model_name: str) -> dict[str, Any]:
    payload = {
        "firm": {
            "source": str(row.get("source") or ""),
            "firm_name": str(row.get("firm_name") or ""),
            "first_legal_entity_name": str(row.get("first_legal_entity_name") or ""),
            "origin_track": str(row.get("origin_track") or ""),
            "validation_label": str(row.get("validation_label") or ""),
            "founded_in_denmark": str(row.get("founded_in_denmark") or ""),
            "danish_founders_abroad": str(row.get("danish_founders_abroad") or ""),
            "founding_year": str(row.get("founding_year") or ""),
            "founding_city": str(row.get("founding_city") or ""),
            "founding_country_iso": str(row.get("founding_country_iso") or ""),
            "evidence_summary": str(row.get("evidence_summary") or ""),
            "founding_evidence": str(row.get("founding_evidence") or ""),
            "founder_danish_context": str(row.get("founder_danish_context") or ""),
            "relocation_evidence": str(row.get("relocation_evidence") or ""),
            "sources_founder_identity": str(row.get("sources_founder_identity") or ""),
            "sources_status_today": str(row.get("sources_status_today") or ""),
            "existing_founder_columns": [
                str(row.get("dk_founder_1") or ""),
                str(row.get("dk_founder_2") or ""),
                str(row.get("dk_founder_3") or ""),
                str(row.get("dk_founder_4") or ""),
                str(row.get("dk_founder_5") or ""),
            ],
            "existing_linkedin_columns": [
                str(row.get("dk_founder_1_linkedin") or ""),
                str(row.get("dk_founder_2_linkedin") or ""),
                str(row.get("dk_founder_3_linkedin") or ""),
                str(row.get("dk_founder_4_linkedin") or ""),
                str(row.get("dk_founder_5_linkedin") or ""),
            ],
            "existing_firm_linkedin": str(row.get("firm_linkedin_url_search") or ""),
        },
        "task": (
            "Identify the firm's LinkedIn company page plus up to five Danish founders relevant to this validated firm record "
            "and provide their LinkedIn URLs. Correct any noisy founder-name columns. Overwrite the firm and founder "
            "LinkedIn slots with the best current results."
        ),
    }
    return {
        "model": model_name,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        "tools": [{"type": "web_search"}],
        "include": ["web_search_call.action.sources"],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "true_review_founder_linkedin_rerun",
                "strict": True,
                "schema": founder_json_schema(),
            }
        },
    }


def founder_json_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    founder_result = {
        "type": "object",
        "additionalProperties": False,
        "required": ["slot", "name", "linkedin_url"],
        "properties": {
            "slot": {"type": "integer"},
            "name": nullable_string,
            "linkedin_url": nullable_string,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["company_linkedin_url", "founders"],
        "properties": {
            "company_linkedin_url": nullable_string,
            "founders": {"type": "array", "items": founder_result},
        },
    }


def apply_founder_result(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["firm_linkedin_url_search"] = str(payload.get("company_linkedin_url") or "").strip()
    for slot in range(1, 6):
        updated[f"dk_founder_{slot}"] = ""
        updated[f"dk_founder_{slot}_linkedin"] = ""

    founder_results = payload.get("founders") or []
    for item in founder_results:
        slot = int(item.get("slot") or 0)
        if slot < 1 or slot > 5:
            continue
        name = str(item.get("name") or "").strip()
        linkedin_url = str(item.get("linkedin_url") or "").strip()
        updated[f"dk_founder_{slot}"] = name
        updated[f"dk_founder_{slot}_linkedin"] = linkedin_url
    return updated


def make_row_key(row: dict[str, Any]) -> str:
    return " | ".join(
        [
            str(row.get("source") or "").strip().casefold(),
            str(row.get("firm_name") or "").strip().casefold(),
            str(row.get("first_legal_entity_name") or "").strip().casefold(),
            str(row.get("founding_year") or "").strip().casefold(),
        ]
    )


def is_retryable_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    retry_markers = (
        "apiconnectionerror",
        "connecterror",
        "readtimeout",
        "timeout",
        "temporar",
        "connection reset",
        "forbidden by its access permissions",
        "winerror 10013",
        "rate limit",
        "429",
        "503",
        "502",
        "504",
    )
    return any(marker in text for marker in retry_markers)


def load_existing_output_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).where(pd.notna, "").to_dict(orient="records")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl_record(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if "firm_linkedin_url_search" not in fieldnames:
        insert_at = fieldnames.index("dk_founder_1") if "dk_founder_1" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "firm_linkedin_url_search")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
