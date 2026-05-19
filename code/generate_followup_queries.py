"""Generate iterative follow-up discovery queries with known-firm exclusions."""

from __future__ import annotations

import argparse
from pathlib import Path

from discovery_memory import export_followup_queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate follow-up discovery queries.")
    parser.add_argument("--memory", required=True, type=Path, help="Path to known firm memory CSV or JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write follow-up query JSONL.")
    parser.add_argument("--round", required=True, type=int, help="Discovery round number.")
    args = parser.parse_args()

    queries = export_followup_queries(output_path=args.output, memory_path=args.memory, round_number=args.round)
    print(f"followup_queries={len(queries)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
