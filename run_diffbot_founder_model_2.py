"""Run Model 2 on the separate Diffbot founder candidate set."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_2_enrichment import DEFAULT_MODEL, PROMPT_VERSION, run_model_2
from src.diffbot_founder_pipeline import DEFAULT_PIPELINE_SLUG, build_diffbot_paths


def main() -> None:
    default_paths = build_diffbot_paths(DEFAULT_PIPELINE_SLUG)
    parser = argparse.ArgumentParser(description="Run Model 2 for the separate Diffbot founder dataset.")
    parser.add_argument("--input", type=Path, default=default_paths.candidates, help="Prepared candidate JSONL input path.")
    parser.add_argument("--output", type=Path, default=default_paths.model2, help="Enriched JSONL output path.")
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
