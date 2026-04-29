This repository builds a staged research pipeline to identify companies that may have been founded in Denmark and later moved their main executive headquarters abroad.

The pipeline now starts from a validated seed list and uses LLM-driven snowball discovery to find additional, less-known firms. It is recall-first through Model 1, stricter in Model 2, conservative in Model 3, and then hands off to human close reading.

Core principles:
- Python only
- Keep code simple, modular, and easy to inspect
- Prefer functions and dataclasses over heavy abstractions
- Preserve provenance at every stage
- Avoid premature optimization
- No frontend
- No scraping yet unless explicitly requested
- External APIs should be minimized
- OpenAI API usage is allowed only in explicit discovery/model scripts
- All OpenAI outputs must be stored with raw responses + parsed outputs
- The system should remain runnable without OpenAI for deterministic stages
- Intermediate outputs should be saved to disk in transparent formats like JSONL and CSV

Research framing:
- `founding_origin == "in Denmark"` are core target cases
- `founding_origin == "abroad (Danish founders)"` are exclusions/context, not core targets
- Danish founder abroad is not the same as founded in Denmark
- foreign office alone is not enough
- acquisition-only is not enough
- legal redomiciling is not automatically executive HQ relocation

Pipeline stages:
1. Validated seed list
   Load `preliminary_data_28_04.csv`, use all names as exclusions, and keep the `in Denmark` subset as core examples.
2. Snowball discovery
   Use OpenAI `web_search` to find additional firms not already in the seed list. Use sector and destination buckets only. Prioritize obscure or long-tail firms.
3. Deduplication against known firms
   Remove already-known firms and merge duplicates by normalized firm name.
4. Model 1 candidate extraction
   Recall-oriented. Ask whether the firm could plausibly be Danish-founded and later moved its own HQ or main operations abroad.
5. Model 2 enrichment/reconciliation
   Stricter cross-source reconciliation with grouped evidence and cleaned names.
6. Model 3 strict validation
   Final conservative LLM gate against the research definition.
7. Human close reading
   Export one row per candidate firm with evidence and validation labels preserved.

Exact command order:
1. `python snowball_discovery.py --known preliminary_data_28_04.csv --output data/discovery/snowball_round_001.jsonl --round 1 --model gpt-5-mini --max-buckets 5`
2. `python deduplicate_snowball_candidates.py --input data/discovery/snowball_round_001.jsonl --known preliminary_data_28_04.csv --output data/discovery/snowball_round_001_deduped.jsonl`
3. `python model_1_candidate_extraction.py --input data/discovery/snowball_round_001_deduped.jsonl --output data/model1/snowball_round_001_candidates.jsonl --model gpt-5-mini`
4. `python model_2_enrichment.py --input data/model1/snowball_round_001_candidates.jsonl --output data/model2/snowball_round_001_enriched.jsonl --model gpt-5-mini`
5. `python model_3_validation.py --input data/model2/snowball_round_001_enriched.jsonl --output data/model3/snowball_round_001_validated.jsonl --model gpt-5-mini`
6. `python export_final_review.py --input data/model3/snowball_round_001_validated.jsonl --output data/review/snowball_round_001_review.csv`
