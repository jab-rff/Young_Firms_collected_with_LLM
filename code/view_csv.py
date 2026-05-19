"""View CSV export in a readable format."""

import csv
import sys
from pathlib import Path


def view_csv(csv_path: Path, max_rows: int = 10, show_fields: list[str] = None) -> None:
    """Display CSV in readable format."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print(f"No rows in {csv_path}")
        return
    
    # Default fields to show
    if not show_fields:
        show_fields = [
            "firm_name",
            "mention_count",
            "languages",
            "unique_sources",
            "confidence",
            "sources",
        ]
    
    print(f"\n{csv_path.name} ({len(rows)} total rows)")
    print("=" * 120)
    
    for i, row in enumerate(rows[:max_rows], 1):
        print(f"\n{i}. {row['firm_name']} (ID: {row['candidate_id']})")
        print(f"   Mentions: {row['mention_count']} | Languages: {row['languages']} | Confidence: {row['confidence']}")
        print(f"   Sources ({row['unique_sources']}): {row['sources']}")
        
        if row['key_evidence']:
            evidence = row['key_evidence'][:150] + ("..." if len(row['key_evidence']) > 150 else "")
            print(f"   Evidence: {evidence}")
        
        if row['raw_name_variants']:
            print(f"   Name variants: {row['raw_name_variants']}")
    
    print(f"\n... and {len(rows) - max_rows} more rows" if len(rows) > max_rows else "")


if __name__ == "__main__":
    csv_path = Path("data/candidates/test_run_001/candidates_review.csv")
    view_csv(csv_path, max_rows=15)
