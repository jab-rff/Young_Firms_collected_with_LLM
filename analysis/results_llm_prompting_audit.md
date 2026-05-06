# Results LLM Prompting Audit

## Overview
- Total workbook rows audited: 54
- Output CSV: `analysis/results_llm_prompting_audit.csv`

## Manual Outcome Counts
- `already_found_borsen`: 6
- `duplicate`: 4
- `manual_false`: 20
- `manual_true`: 21
- `manual_unclear`: 3

## Pipeline Outcome Counts
- `already_found_by_original_method`: 6
- `duplicate_variant`: 4
- `manual_outcome_unclear`: 3
- `never_seen_in_local_pipeline_outputs`: 3
- `pipeline_captured_true_case`: 13
- `pipeline_surfaced_but_final_label_differs`: 5
- `pipeline_surfaced_but_manual_close_reading_rejected`: 20

## Manual True Cases
- `never_seen_in_local_pipeline_outputs`: 3
- `pipeline_captured_true_case`: 13
- `pipeline_surfaced_but_final_label_differs`: 5

## Manual True Cases Never Seen Locally
- Royal Republiq
- Swipes
- Filecamp

## Manual True Cases With Different Local Final Labels
- Synamics Therapeutics | master_review_label=false | likely_reason=The firm reached late-stage validation, but the local pipeline classified it as a non-target case.
- Hemab | master_review_label=false | likely_reason=The firm reached late-stage validation, but the local pipeline classified it as a non-target case.
- Issuu | master_review_label=false | likely_reason=The firm reached late-stage validation, but the local pipeline classified it as a non-target case.
- Linde Werdelin | master_review_label=unclear | likely_reason=The firm reached late-stage validation, but the local pipeline kept it as uncertain rather than true.
- True North Cosmetics | master_review_label=unclear | likely_reason=The firm reached late-stage validation, but the local pipeline kept it as uncertain rather than true.
