"""Command line entry point for the recall pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recall-first company pipeline.")
    parser.add_argument("--input", required=True, type=Path, help="Path to retrieved JSONL input.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for pipeline outputs.")
    parser.add_argument(
        "--run-enrichment",
        default="false",
        choices=["true", "false"],
        help="Whether to run optional OpenAI enrichment.",
    )
    args = parser.parse_args()

    mentions, candidates, enrichment_results = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        run_enrichment=args.run_enrichment == "true",
    )

    print(f"mentions={len(mentions)}")
    print(f"candidates={len(candidates)}")
    print(f"enrichment_results={len(enrichment_results)}")
    print(f"output_dir={args.output}")


if __name__ == "__main__":
    main()
