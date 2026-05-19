"""View generated queries in a friendly format."""

import json
from pathlib import Path
from collections import defaultdict

def view_queries(queries_file: Path) -> None:
    """Display queries organized by family and language."""
    with open(queries_file) as f:
        queries = [json.loads(line) for line in f]
    
    print(f"\nQueries from: {queries_file.name}")
    print("=" * 90)
    print(f"Total: {len(queries)} queries\n")
    
    # Group by family then language
    by_family = defaultdict(lambda: defaultdict(list))
    for q in queries:
        by_family[q['family']][q['language']].append(q)
    
    for family in sorted(by_family.keys()):
        print(f"\n{family.upper()}")
        print("-" * 90)
        
        for lang in sorted(by_family[family].keys()):
            queries_lang = by_family[family][lang]
            lang_label = "English" if lang == "en" else "Danish"
            print(f"  {lang_label} ({len(queries_lang)}):")
            
            for i, q in enumerate(queries_lang[:3], 1):
                print(f"    {i}. {q['query_text'][:75]}")
            
            if len(queries_lang) > 3:
                print(f"    ... and {len(queries_lang) - 3} more")


if __name__ == "__main__":
    view_queries(Path("data/queries.jsonl"))
    print("\n")
    view_queries(Path("data/queries_extended.jsonl"))
