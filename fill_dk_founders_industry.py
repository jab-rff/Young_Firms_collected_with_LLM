"""Fill missing industry values in the DK founders CSV using the OpenAI API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from snowball_discovery import load_openai_api_key
from src.industry_enrichment_openai import DEFAULT_MODEL, build_input_record, classify_firm_industry
from src.io import save_jsonl
from src.normalization import normalize_company_name
from src.openai_costs import build_cost_record


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field) or "") for field in fieldnames})


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    records = load_jsonl_records(path)
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = normalize_company_name(record.get("firm"))
        if key:
            indexed[key] = record
    return indexed


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


def should_process_row(row: dict[str, str], *, overwrite: bool) -> bool:
    industry = str(row.get("industry") or "").strip()
    if overwrite:
        return bool(str(row.get("firm") or "").strip())
    return bool(str(row.get("firm") or "").strip()) and not industry


def print_progress(current: int, total: int, firm: str) -> None:
    total = max(total, 1)
    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    message = f"\r[{bar}] {current}/{total} {firm[:50]}"
    print(message, end="", file=sys.stdout, flush=True)


def persist_outputs(
    *,
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    parsed_records: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    input_records: list[dict[str, Any]],
    cost_records: list[dict[str, Any]],
) -> None:
    parsed_output_path = output_path.with_name(f"{output_path.stem}_industry_enrichment.jsonl")
    raw_output_path = output_path.with_name(f"{output_path.stem}_industry_raw_responses.jsonl")
    input_log_path = output_path.with_name(f"{output_path.stem}_industry_inputs.jsonl")
    cost_output_path = output_path.with_name(f"{output_path.stem}_industry_api_costs.jsonl")
    write_csv_rows(output_path, fieldnames, rows)
    save_jsonl(parsed_records, parsed_output_path)
    save_jsonl(raw_records, raw_output_path)
    save_jsonl(input_records, input_log_path)
    save_jsonl(cost_records, cost_output_path)


def classify_with_retries(
    row: dict[str, str],
    *,
    model_name: str,
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    delays = [2, 5, 10]
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return classify_firm_industry(row, model_name=model_name)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait_seconds = delays[min(attempt, len(delays) - 1)]
            firm = str(row.get("firm") or "").strip()
            print(
                f"\nRequest failed for {firm}. Retrying in {wait_seconds}s "
                f"({attempt + 1}/{max_retries})...",
                file=sys.stdout,
                flush=True,
            )
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def enrich_industries(
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    overwrite: bool = False,
    limit: int | None = None,
    checkpoint_every: int = 1,
    max_retries: int = 3,
) -> dict[str, Any]:
    load_openai_api_key()
    fieldnames, rows = load_csv_rows(input_path)
    if "industry" not in fieldnames:
        raise ValueError(f"CSV is missing required 'industry' column: {input_path}")

    parsed_output_path = output_path.with_name(f"{output_path.stem}_industry_enrichment.jsonl")
    raw_output_path = output_path.with_name(f"{output_path.stem}_industry_raw_responses.jsonl")
    input_log_path = output_path.with_name(f"{output_path.stem}_industry_inputs.jsonl")
    cost_output_path = output_path.with_name(f"{output_path.stem}_industry_api_costs.jsonl")

    existing_results = load_existing_results(parsed_output_path)
    input_records = load_jsonl_records(input_log_path)
    parsed_records = load_jsonl_records(parsed_output_path)
    raw_records = load_jsonl_records(raw_output_path)
    cost_records = load_jsonl_records(cost_output_path)
    rows_to_process = [row for row in rows if should_process_row(row, overwrite=overwrite)]
    total_targets = len(rows_to_process)

    processed = 0
    for row in rows:
        if not should_process_row(row, overwrite=overwrite):
            continue
        firm_key = normalize_company_name(row.get("firm"))
        if not firm_key:
            continue
        if firm_key in existing_results:
            row["industry"] = str(existing_results[firm_key].get("industry") or "")
            processed += 1
            print_progress(processed, total_targets, str(row.get("firm") or "").strip())
            continue
        if limit is not None and processed >= limit:
            continue

        input_record = build_input_record(row)
        try:
            result, raw_record = classify_with_retries(
                row,
                model_name=model_name,
                max_retries=max_retries,
            )
        except Exception:
            persist_outputs(
                output_path=output_path,
                fieldnames=fieldnames,
                rows=rows,
                parsed_records=parsed_records,
                raw_records=raw_records,
                input_records=input_records,
                cost_records=cost_records,
            )
            if total_targets:
                print("", file=sys.stdout, flush=True)
            raise
        row["industry"] = str(result.get("industry") or "")
        input_records.append(input_record)
        parsed_records.append(result)
        raw_records.append(raw_record)
        cost_records.append(
            build_cost_record(
                stage="industry_enrichment",
                request_kind="firm_industry",
                raw_response=raw_record["response"],
                requested_model=model_name,
                metadata={"firm": str(row.get("firm") or "").strip()},
            )
        )
        existing_results[firm_key] = result
        processed += 1
        print_progress(processed, total_targets, str(row.get("firm") or "").strip())
        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            persist_outputs(
                output_path=output_path,
                fieldnames=fieldnames,
                rows=rows,
                parsed_records=parsed_records,
                raw_records=raw_records,
                input_records=input_records,
                cost_records=cost_records,
            )

    persist_outputs(
        output_path=output_path,
        fieldnames=fieldnames,
        rows=rows,
        parsed_records=parsed_records,
        raw_records=raw_records,
        input_records=input_records,
        cost_records=cost_records,
    )
    if total_targets:
        print("", file=sys.stdout, flush=True)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "processed": processed,
        "rows_total": len(rows),
        "parsed_output_path": str(parsed_output_path),
        "raw_output_path": str(raw_output_path),
        "input_log_path": str(input_log_path),
        "cost_output_path": str(cost_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing industry values for the DK founders CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dk_founders/results_true_dk_founders_llm.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dk_founders/results_true_dk_founders_llm_with_industry.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help="OpenAI Responses model.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute industry even when already filled.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on rows to enrich.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save progress every N processed rows.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry failed API requests this many times.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = enrich_industries(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        overwrite=args.overwrite,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        max_retries=args.max_retries,
    )
    print(
        f"Processed {summary['processed']} rows. Updated CSV: {summary['output_path']}. "
        f"Parsed log: {summary['parsed_output_path']}"
    )


if __name__ == "__main__":
    main()
