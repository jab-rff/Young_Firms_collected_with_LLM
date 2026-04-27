This repository builds a recall-first research pipeline to identify companies that may have been founded in Denmark and later moved their main executive headquarters abroad.

The pipeline is designed for high recall first, not final precision. Its job is to surface candidate firms with preserved provenance so they can later be verified with stricter logic and additional evidence.

Core principles:
- Python only
- Keep code simple, modular, and easy to inspect
- Prefer functions and dataclasses over heavy abstractions
- Preserve provenance at every stage
- Avoid premature optimization
- No frontend
- No scraping yet unless explicitly requested
- External APIs should be minimized
- OpenAI API usage is allowed for GPT-based query generation and structured extraction
- All OpenAI calls must preserve provenance, including prompt, model, raw response, parsed output, and timestamp
- Deterministic stages should remain runnable without OpenAI
- Intermediate outputs should be saved to disk in transparent formats like JSONL, CSV, or parquet

Research framing:
We are not yet trying to prove that a company truly qualifies.
We are only trying to maximize recall of plausible candidates.

A company is a candidate if there is any plausible evidence that:
- it may have been founded in Denmark
- it may later have established its main headquarters abroad

At the recall stage, uncertain candidates should usually be kept, not discarded.

Target candidate-level fields:
- firm_name
- founding_city
- founding_country
- founding_year
- moved_hq_abroad
- move_city
- move_country
- move_year
- move_context
- move_related_to_acquisition_or_merger
- acquisition_or_merger_context
- sources
- evidence_snippets

Planned stages:
1. Query generation
   Generate broad, diverse search queries and query families in English and Danish.
2. Retrieval
   Collect raw search results or input documents and store them unchanged.
3. Candidate mention extraction
   Extract company mentions from search results or text snippets.
4. Normalization
   Normalize company names conservatively.
5. Aggregation
   Group mentions into candidate firms while preserving all evidence.
6. Deduplication
   Apply simple deterministic deduplication only.
7. Export
   Save review-ready candidate outputs for later verification.

Important constraints:
- No fuzzy matching yet
- No final inclusion/exclusion logic yet
- No LLM adjudication yet
- A Danish founder starting a company abroad is not the same as a company being founded in Denmark
- Foreign offices, subsidiaries, sales offices, manufacturing moves, or legal redomiciling should not automatically count as executive HQ relocation

Definition of success for the recall stage:
- The system produces a candidate-level dataset
- Each candidate firm is linked to one or more source-backed mentions
- Every mention retains query/source provenance
- The output is easy to inspect manually
