"""Export the filtered case subset that feeds the manual close-reading app."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.manual_close_reading import (
    DEFAULT_FINAL_DATASET_PATH,
    DEFAULT_MANUAL_REVIEW_PATH,
    DEFAULT_VALIDATED_MASTER_PATH,
    build_manual_close_reading_rows,
    load_existing_manual_rows,
    load_jsonl,
    sanitize_manual_rows,
    save_manual_close_reading_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the filtered manual close-reading case subset to CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VALIDATED_MASTER_PATH,
        help="Validated master JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANUAL_REVIEW_PATH,
        help="Working close-reading CSV path.",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=DEFAULT_FINAL_DATASET_PATH,
        help="Final-dataset-style CSV path.",
    )
    parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Rebuild from validated records only and do not merge existing manual CSV rows.",
    )
    args = parser.parse_args()

    validated_records = load_jsonl(args.input)
    existing_rows = {} if args.ignore_existing else load_existing_manual_rows(args.output)
    rows = build_manual_close_reading_rows(validated_records, existing_rows=existing_rows)
    sanitized = sanitize_manual_rows(rows)
    save_manual_close_reading_rows(
        sanitized,
        manual_review_path=args.output,
        final_dataset_path=args.final_output,
    )
    print(f"close_reading_rows={len(sanitized)}")
    print(f"working_output_path={args.output}")
    print(f"final_output_path={args.final_output}")


if __name__ == "__main__":
    main()
