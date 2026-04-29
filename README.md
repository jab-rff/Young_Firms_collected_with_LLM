# Young Firms Staged Pipeline

This repository is organized around a clean staged pipeline for finding firms that may have been founded in Denmark and later moved their own executive headquarters or main operations abroad.

The process now starts from a validated seed list and uses LLM-driven snowball discovery to find additional, less-known firms. The pipeline remains recall-first through Model 1, becomes stricter in Model 2, uses a conservative gate in Model 3, and then hands off to human close reading.

## Stages
1. Validated seed list
   Load `preliminary_data_28_04.csv`. Use all names as exclusions. Treat `founding_origin == "in Denmark"` as the core relocation subset.
2. Snowball discovery
   Use OpenAI `web_search` to find additional firms outside the known seed list, using sector and destination buckets only.
3. Deduplication against known firms
   Remove already-known firms and merge repeated candidates by normalized firm name.
4. Model 1 candidate extraction
   Recall-oriented structuring of the deduplicated snowball outputs.
5. Model 2 enrichment/reconciliation
   Stricter cross-source reconciliation with cleaned names and grouped evidence.
6. Model 3 strict validation
   Conservative final LLM gate against the research definition.
7. Human close reading
   Review CSV output with evidence snippets, URLs, and validation labels preserved.

## Key Files
- `src/seed_list.py`
- `snowball_discovery.py`
- `deduplicate_snowball_candidates.py`
- `model_1_candidate_extraction.py`
- `model_2_enrichment.py`
- `model_3_validation.py`
- `export_final_review.py`

## Command Order
```bash
python snowball_discovery.py --known preliminary_data_28_04.csv --output data/discovery/snowball_round_001.jsonl --round 1 --model gpt-5-mini --max-buckets 5

python deduplicate_snowball_candidates.py --input data/discovery/snowball_round_001.jsonl --known preliminary_data_28_04.csv --output data/discovery/snowball_round_001_deduped.jsonl

python model_1_candidate_extraction.py --input data/discovery/snowball_round_001_deduped.jsonl --output data/model1/snowball_round_001_candidates.jsonl --model gpt-5-mini

python model_2_enrichment.py --input data/model1/snowball_round_001_candidates.jsonl --output data/model2/snowball_round_001_enriched.jsonl --model gpt-5-mini

python model_3_validation.py --input data/model2/snowball_round_001_enriched.jsonl --output data/model3/snowball_round_001_validated.jsonl --model gpt-5-mini

python export_final_review.py --input data/model3/snowball_round_001_validated.jsonl --output data/review/snowball_round_001_review.csv
```

## Development
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m pytest -v
```
