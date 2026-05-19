"""Search for founder and firm LinkedIn URLs for the merged abroad-Danish-founders master CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.io import save_jsonl
from src.openai_costs import build_cost_record, cost_log_path, sum_cost_records

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

    tqdm.write = print  # type: ignore[attr-defined]


DEFAULT_INPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_merged_master.csv")
DEFAULT_OUTPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_merged_master_linkedin.csv")
DEFAULT_RAW_OUTPUT = Path("data/cumulative/final_review_master_abroad_danish_founders_merged_master_linkedin_raw.jsonl")
DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "2026-05-07-founder-linkedin-v1"

SYSTEM_PROMPT = """You are finding LinkedIn URLs for a company and named founders.

Rules:
- Use web search.
- Prefer exact LinkedIn profile URLs for people in the form https://www.linkedin.com/in/... or a country subdomain equivalent.
- Prefer the company LinkedIn page in the form https://www.linkedin.com/company/... when available.
- Be conservative. Return null when uncertain.
- Do not invent URLs.
- Match founders to the provided firm context; avoid unrelated people with the same name.

Return only JSON matching the schema."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich merged abroad-Danish-founders master with LinkedIn URLs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
    if args.limit is not None:
        rows = rows[: args.limit]

    enriched_rows = load_existing_output_rows(args.output)
    processed_keys = {make_row_key(row) for row in enriched_rows}
    cost_records = load_jsonl_records(cost_log_path(args.output))

    pending_rows = [row for row in rows if make_row_key(row) not in processed_keys]
    if enriched_rows:
        tqdm.write(f"resume_detected_existing_rows={len(enriched_rows)}")
    tqdm.write(f"pending_rows={len(pending_rows)}")

    for row in tqdm(pending_rows, total=len(pending_rows), desc="Founder LinkedIn Search", unit="firm"):
        enriched_row, raw_record, cost_record = enrich_row_with_retries(
            row,
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
    enriched = apply_linkedin_result(row, parsed)
    raw_record = {
        "name": str(row.get("name") or ""),
        "prompt_version": PROMPT_VERSION,
        "request": request_body,
        "response": response.model_dump(mode="json") if hasattr(response, "model_dump") else {"repr": repr(response)},
        "parsed": parsed,
    }
    cost_record = build_cost_record(
        stage="founder_linkedin_search",
        request_kind="web_search",
        raw_response=response,
        requested_model=model_name,
        metadata={"name": str(row.get("name") or "")},
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
            firm_name = str(row.get("name") or "").strip()
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
    founders = []
    for index in range(1, 6):
        founder_name = str(row.get(f"dk_founder_{index}") or "").strip()
        existing_linkedin = str(row.get(f"dk_founder_{index}_linkedin") or "").strip()
        if founder_name:
            founders.append(
                {
                    "slot": index,
                    "name": founder_name,
                    "existing_linkedin": existing_linkedin or None,
                }
            )

    payload = {
        "firm": {
            "name": str(row.get("name") or ""),
            "first_legal_name": str(row.get("first_legal_name") or ""),
            "source_list": str(row.get("source_list") or ""),
            "existing_company_linkedin": str(row.get("diffbot_company_linkedin") or ""),
            "founder_context": str(row.get("founder_danish_context") or ""),
        },
        "founders": founders,
        "task": "Find the best LinkedIn company URL and founder LinkedIn profile URLs for this row.",
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
                "name": "founder_linkedin_search",
                "strict": True,
                "schema": linkedin_json_schema(),
            }
        },
    }


def linkedin_json_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    founder_result = {
        "type": "object",
        "additionalProperties": False,
        "required": ["slot", "name", "linkedin_url"],
        "properties": {
            "slot": {"type": "integer"},
            "name": {"type": "string"},
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


def apply_linkedin_result(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    company_linkedin = str(payload.get("company_linkedin_url") or "").strip()
    if company_linkedin:
        updated["firm_linkedin_url_search"] = company_linkedin
    elif "firm_linkedin_url_search" not in updated:
        updated["firm_linkedin_url_search"] = ""

    founder_results = payload.get("founders") or []
    for item in founder_results:
        slot = int(item.get("slot") or 0)
        if slot < 1 or slot > 5:
            continue
        linkedin_url = str(item.get("linkedin_url") or "").strip()
        column = f"dk_founder_{slot}_linkedin"
        if linkedin_url:
            updated[column] = linkedin_url
    return updated


def make_row_key(row: dict[str, Any]) -> str:
    return " | ".join(
        [
            str(row.get("name") or "").strip().casefold(),
            str(row.get("first_legal_name") or "").strip().casefold(),
            str(row.get("source_list") or "").strip().casefold(),
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
        insert_at = fieldnames.index("dk_founder_1_linkedin") if "dk_founder_1_linkedin" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "firm_linkedin_url_search")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
