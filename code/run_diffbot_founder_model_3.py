"""Run Model 3 on the separate Diffbot founder dataset with isolated cumulative outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_3_validation import DEFAULT_MODEL, PROMPT_VERSION, run_model_3
from src.diffbot_founder_pipeline import DEFAULT_PIPELINE_SLUG, build_diffbot_paths


def main() -> None:
    default_paths = build_diffbot_paths(DEFAULT_PIPELINE_SLUG)
    parser = argparse.ArgumentParser(description="Run Model 3 for the separate Diffbot founder dataset.")
    parser.add_argument("--input", type=Path, default=default_paths.model2, help="Enriched JSONL input path.")
    parser.add_argument("--output", type=Path, default=default_paths.model3, help="Validated JSONL output path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for small debugging runs.")
    parser.add_argument("--batch", action="store_true", help="Submit requests through the OpenAI Batch API.")
    parser.add_argument(
        "--master-validated-output",
        type=Path,
        default=default_paths.master_validated,
        help="Separate cumulative master validated JSONL path for this Diffbot run family.",
    )
    parser.add_argument(
        "--master-review-output",
        type=Path,
        default=default_paths.master_review,
        help="Separate cumulative master review CSV path for this Diffbot run family.",
    )
    args = parser.parse_args()

    records = run_model_3(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        limit=args.limit,
        batch=args.batch,
        master_validated_path=args.master_validated_output,
        master_review_path=args.master_review_output,
    )
    print(f"prompt_version={PROMPT_VERSION}")
    print(f"validated_records={len(records)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
