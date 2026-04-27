"""Export candidates to CSV format for manual review."""

from pathlib import Path
import json
import argparse
from src.data_models import CandidateFirm, Mention
from src.export import export_candidates_to_csv, export_candidates_filtered_csv


def load_candidates_and_mentions(candidates_path: Path, mentions_path: Path) -> tuple[list[CandidateFirm], list[Mention]]:
    """Load candidates and mentions from JSONL files."""
    candidates = []
    with open(candidates_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                candidates.append(CandidateFirm(**data))
    
    mentions = []
    with open(mentions_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                mentions.append(Mention(**data))
    
    return candidates, mentions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export candidates to CSV format for manual review."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidates.jsonl from pipeline output"
    )
    parser.add_argument(
        "--mentions",
        type=Path,
        required=True,
        help="Path to mentions.jsonl from pipeline output"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--filter-min-mentions",
        type=int,
        default=0,
        help="Only include candidates with at least this many mentions (default: 0, include all)"
    )
    parser.add_argument(
        "--filter-min-sources",
        type=int,
        help="Only include candidates with mentions from at least this many unique sources"
    )
    
    args = parser.parse_args()
    
    print(f"Loading candidates from {args.candidates}")
    print(f"Loading mentions from {args.mentions}")
    
    candidates, mentions = load_candidates_and_mentions(args.candidates, args.mentions)
    print(f"Loaded: {len(candidates)} candidates, {len(mentions)} mentions")
    print()
    
    if args.filter_min_mentions > 0 or args.filter_min_sources:
        export_candidates_filtered_csv(
            args.output,
            candidates,
            mentions,
            min_mention_count=args.filter_min_mentions,
            min_unique_sources=args.filter_min_sources,
        )
    else:
        export_candidates_to_csv(args.output, candidates, mentions)


if __name__ == "__main__":
    main()
