"""Run the separate Diffbot founder preparation, Model 2, Model 3, and review export."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_final_review import export_final_review, load_jsonl
from model_2_enrichment import DEFAULT_MODEL as MODEL2_DEFAULT, run_model_2
from model_3_validation import DEFAULT_MODEL as MODEL3_DEFAULT, run_model_3
from src.diffbot_founder_pipeline import (
    DEFAULT_DIFFBOT_INPUT_PATH,
    DEFAULT_PIPELINE_SLUG,
    build_diffbot_candidate,
    build_diffbot_paths,
    load_diffbot_rows,
)
from src.io import save_jsonl


def main() -> None:
    default_paths = build_diffbot_paths(DEFAULT_PIPELINE_SLUG)
    parser = argparse.ArgumentParser(description="Run the separate Diffbot founder pipeline end-to-end.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DIFFBOT_INPUT_PATH, help="Path to Diffbot CSV input.")
    parser.add_argument("--slug", default=DEFAULT_PIPELINE_SLUG, help="Output slug token.")
    parser.add_argument("--model2", default=MODEL2_DEFAULT, help=f"Model 2 OpenAI model (default: {MODEL2_DEFAULT})")
    parser.add_argument("--model3", default=MODEL3_DEFAULT, help=f"Model 3 OpenAI model (default: {MODEL3_DEFAULT})")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for small debugging runs.")
    parser.add_argument("--batch", action="store_true", help="Submit Model 2 and Model 3 through the OpenAI Batch API.")
    args = parser.parse_args()

    paths = build_diffbot_paths(args.slug)
    rows = load_diffbot_rows(args.input)
    candidates = [build_diffbot_candidate(row) for row in rows if str(row.get("name") or "").strip()]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    save_jsonl(candidates, paths.candidates)

    run_model_2(
        input_path=paths.candidates,
        output_path=paths.model2,
        model_name=args.model2,
        limit=None,
        batch=args.batch,
    )
    run_model_3(
        input_path=paths.model2,
        output_path=paths.model3,
        model_name=args.model3,
        limit=None,
        batch=args.batch,
        master_validated_path=paths.master_validated,
        master_review_path=paths.master_review,
    )

    validated = load_jsonl(paths.model3)
    export_final_review(validated, paths.review)

    print(f"candidate_rows={len(candidates)}")
    print(f"candidate_output_path={paths.candidates}")
    print(f"model2_output_path={paths.model2}")
    print(f"model3_output_path={paths.model3}")
    print(f"review_output_path={paths.review}")
    print(f"master_validated_path={paths.master_validated}")
    print(f"master_review_path={paths.master_review}")


if __name__ == "__main__":
    main()
