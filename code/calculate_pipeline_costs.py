"""Summarize estimated OpenAI API costs for a pipeline round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.openai_costs import sum_cost_records
from src.pipeline_rounds import build_round_paths


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_cost_log_paths(round_number: int) -> dict[str, Path]:
    paths = build_round_paths(round_number)
    return {
        "discovery": paths.discovery.with_name(f"{paths.discovery.stem}_api_costs.jsonl"),
        "model1": paths.model1.with_name(f"{paths.model1.stem}_api_costs.jsonl"),
        "model2": paths.model2.with_name(f"{paths.model2.stem}_api_costs.jsonl"),
        "model3": paths.model3.with_name(f"{paths.model3.stem}_api_costs.jsonl"),
    }


def summarize_round_costs(round_number: int) -> dict[str, Any]:
    stage_totals: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    for stage, path in build_cost_log_paths(round_number).items():
        records = load_jsonl(path)
        totals = sum_cost_records(records)
        totals["cost_log_path"] = str(path)
        stage_totals[stage] = totals
        total_cost += float(totals["estimated_cost_usd"])
    return {
        "round_number": round_number,
        "stages": stage_totals,
        "estimated_cost_usd_total": round(total_cost, 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize estimated OpenAI costs for a pipeline round.")
    parser.add_argument("--round", required=True, type=int, dest="round_number", help="Pipeline round number.")
    args = parser.parse_args()

    summary = summarize_round_costs(args.round_number)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
