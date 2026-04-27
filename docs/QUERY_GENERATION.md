# Query Generation

This document describes the query generation stage of the recall-first company discovery pipeline.

## Overview

Query generation produces diverse, high-recall search queries in **English** and **Danish** designed to surface companies that:
- Were founded in Denmark (Copenhagen, Aarhus, etc.)
- Later moved their executive headquarters abroad

The pipeline prioritizes **recall over precision** at this stage, so queries are intentionally broad and varied to avoid missing relevant results.

## Query Structure

Each query is a `Query` object with:
- `query_id`: Unique identifier (e.g., `q_en_001`)
- `query_text`: The actual search query string
- `language`: `"en"` or `"da"` (English or Danish)
- `family`: Semantic grouping of related queries
- `created_at`: ISO datetime timestamp

## Query Families

Queries are organized into **7 semantic families**:

### 1. **hq_move** (10 queries)
Direct headquarters relocation language.
- Examples: `"founded in Denmark" "headquarters"`, `"Denmark" "moved headquarters"`
- Best for: High-precision HQ move detection
- Priority: High (recall-first stage 1)

### 2. **startup_hq** (8 queries)
Startup + headquarters combinations.
- Examples: `"Danish startup" "headquarters" "United States"`, `"Danish company" "moved" "San Francisco"`
- Best for: Tech startup context
- Priority: High

### 3. **location_pairs** (11 queries)
Specific city/country pair mentions.
- Examples: `"Copenhagen" "San Francisco"`, `"Copenhagen" "Palo Alto"`
- Best for: Cross-border location mentions
- Priority: High

### 4. **acquisition** (6 queries)
M&A context combined with founding location.
- Examples: `"Danish company" acquired "headquarters"`, `"founded in Denmark" acquired "moved"`
- Best for: Acquisition-related relocations
- Priority: Medium (2)

### 5. **industry_dk** (8 queries)
Industry + Danish origin combinations.
- Examples: `"Danish software" company "headquarters"`, `"Danish SaaS" company moved`
- Best for: Sector-specific discovery
- Priority: Medium

### 6. **relocation** (6 queries)
Generic relocation context.
- Examples: `"Denmark" "relocated headquarters" company`
- Best for: General relocation mentions
- Priority: Medium

### 7. **exploratory** (9 queries, optional)
Broader, less structured exploratory queries.
- Examples: `Danish company Silicon Valley`, `Danish founders moved headquarters`
- Best for: Higher-breadth, lower-precision searching
- Priority: Standard (3) - included with `--include-exploratory`

## Language Balance

Queries are balanced between **English** and **Danish**:
- **Standard set**: 49 queries (25 EN, 24 DA)
- **Extended set**: 58 queries (30 EN, 28 DA)

## Usage

### Generate Queries

```bash
# Generate standard seed queries only
python generate_queries.py --output data/queries.jsonl

# Generate extended queries including exploratory variants
python generate_queries.py --output data/queries_extended.jsonl --include-exploratory
```

### View Queries

```bash
python view_queries.py
```

### Use in Code

```python
from src.query_generation import generate_seed_queries, generate_all_queries

# Get only seed queries
queries = generate_seed_queries()

# Get all queries including exploratory
all_queries = generate_all_queries(include_exploratory=True)

# Group by family
from src.query_generation import queries_by_family
by_family = queries_by_family(queries)

# Get summary
from src.query_generation import queries_summary
print(queries_summary(queries))
```

## Output Formats

### JSONL Format
Each line is a JSON object:
```json
{
  "query_id": "q_en_001",
  "query_text": "\"founded in Denmark\" \"headquarters\"",
  "language": "en",
  "family": "hq_move",
  "created_at": "2025-04-27T14:30:00+00:00"
}
```

## Design Principles

1. **High Recall**: Queries cast a wide net to avoid missing companies
2. **Balanced Languages**: English and Danish queries are proportionally equal
3. **Semantic Families**: Grouping enables strategic query sequencing and analysis
4. **Phrase Matching**: Most queries use quoted phrases for precision
5. **Broad Coverage**: Multiple angles (location pairs, industry, M&A context)
6. **Non-exclusive**: Queries can overlap; recall-first pipeline expects some redundancy

## Future Enhancements

- [ ] Add country-specific query variants (e.g., Swedish, Norwegian)
- [ ] Implement query prioritization based on historical effectiveness
- [ ] Add dynamic query generation based on discovered companies
- [ ] Create query templates for programmatic expansion
- [ ] Add query validation against known datasets

## Testing

Run tests to verify query generation:

```bash
pytest tests/test_query_generation.py -v
```

Tests validate:
- Query count and structure
- Unique IDs and texts
- Language balance
- Family distribution
- Format consistency
