"""Generate and export queries for the recall-first pipeline."""

from pathlib import Path
import json
import argparse
from src.query_generation import generate_all_queries, queries_summary


def export_queries_to_jsonl(
    output_path: Path,
    include_exploratory: bool = False,
) -> None:
    """Generate queries and save to JSONL format.
    
    Args:
        output_path: Path to write queries.jsonl
        include_exploratory: Whether to include exploratory queries
    """
    queries = generate_all_queries(include_exploratory=include_exploratory)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for query in queries:
            # Convert dataclass to dict
            query_dict = {
                "query_id": query.query_id,
                "query_text": query.query_text,
                "language": query.language,
                "family": query.family,
                "created_at": query.created_at,
            }
            f.write(json.dumps(query_dict) + "\n")
    
    print(f"✓ Exported {len(queries)} queries to {output_path}")
    print()
    print(queries_summary(queries))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and export search queries for recall-first discovery."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/queries.jsonl"),
        help="Output path for queries.jsonl (default: data/queries.jsonl)",
    )
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help="Include exploratory queries (lower precision, higher breadth)",
    )
    
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    export_queries_to_jsonl(args.output, include_exploratory=args.include_exploratory)


if __name__ == "__main__":
    main()
