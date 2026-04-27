# CSV Review Guide

This guide explains how to use the exported CSV files for manual verification of candidates.

## Overview

The CSV export provides a structured format for reviewing extracted candidates before final inclusion. Each row represents one candidate company with all supporting evidence in a single, scannable format.

## CSV Fields Explained

### Identification Fields

**candidate_id**
- Unique SHA1-based identifier for this candidate (e.g., `cand_7ab789cbd3d2`)
- Used to track candidates across multiple runs
- Cannot be changed during review

**firm_name**
- Human-readable company name (primary variant)
- Usually the most common or formal name variant found
- This is what appears in your final output

**normalized_name**
- Lowercase, whitespace-normalized version of firm_name
- Used for deduplication (internal use)
- Information only

### Evidence Fields

**mention_count**
- Number of times this company was mentioned across all sources
- Higher = stronger evidence for inclusion
- Typically 1-10+

**languages**
- Which languages the mentions were found in
- `en` = English only
- `da` = Danish only
- `en/da` = both languages (stronger signal)

**unique_sources**
- How many distinct sources mentioned this company
- Examples: Wikipedia, Tech News, Company Blog, LinkedIn, etc.
- Higher = more reliable

**sources**
- Semicolon-separated list of all sources
- Use to validate against your own research
- Examples: `Wikipedia; Company Blog; LinkedIn`

**source_urls**
- Up to 3 URLs from the source documents
- Click to verify the claim directly
- Truncated if many sources exist

### Confidence Scoring

**confidence**
- Automatic assessment: `high`, `medium`, or `low`
- Based on mention count, unique sources, and language coverage
- **High**: 3+ mentions with multi-source coverage
- **Medium**: 2-3 mentions, single or dual source
- **Low**: Single mention only

**confidence_signals**
- Specific reasons for the confidence score
- Examples:
  - `3+ mentions (6)` = 6 total mentions
  - `multi-source (3)` = mentioned by 3 different sources
  - `bilingual` = found in both English and Danish

### Content Fields

**key_evidence**
- Most revealing excerpt showing:
  - Company founding in Denmark
  - HQ relocation to another country
- Up to 500 characters
- Read this first to understand why it was flagged

**raw_name_variants**
- All name variations found in sources
- Semicolon-separated (up to 5 variants)
- Examples: `Zendesk Inc; Zendesk, Inc.; Zendesk A/S`
- Helps identify if names should be merged in deduplication

### Review Fields (For You to Fill)

**verification_status**
- Empty initially; **you** fill this in during review
- Recommended values:
  - `verified` = Confirmed as Danish-founded + moved HQ abroad
  - `rejected` = False positive, doesn't meet criteria
  - `needs_more_info` = Promising but needs additional research
  - `duplicate` = Appears to be duplicate of another candidate
  - `edge_case` = Borderline (e.g., acquisition, subsidiary)

**reviewer_notes**
- Empty initially; **you** can add comments
- Examples:
  - "Founded by Danish team but company registered in Germany"
  - "UK subsidiary, not main HQ"
  - "Founder-led company moved, not corporate relocation"
  - "Strong candidate, recommend inclusion"

## Review Workflow

### Step 1: Sort by Confidence
1. Open the CSV in Excel, Google Sheets, or your favorite tool
2. Sort by `confidence` column (descending) to prioritize review
3. Or filter to `confidence == high` to start with strongest candidates

### Step 2: Review Each Candidate
For each row:
1. Read the `key_evidence` field
2. Click `source_urls` to verify directly (if available)
3. Check `sources` to understand where claims came from
4. Review `raw_name_variants` to spot merge opportunities

### Step 3: Make a Decision
Fill in `verification_status` with one of:

| Status | Meaning | Action |
|--------|---------|--------|
| `verified` | Confirmed match | Include in final output |
| `rejected` | Doesn't meet criteria | Remove from results |
| `needs_more_info` | Unclear | Research manually, decide later |
| `duplicate` | Same company as another row | Mark both rows, keep one |
| `edge_case` | Borderline situation | Document in notes, decide on inclusion rules |

### Step 4: Add Notes (Optional)
In the `reviewer_notes` field, briefly explain your decision:
- Why you rejected it
- Why you marked it as edge case
- Any caveats for inclusion
- Links to additional research

## Practical Tips

### Handling False Positives
Some rows will be false positives (e.g., "Palo Alto" as a company name). These are normal in high-recall extraction. Mark them as `rejected`.

### Handling Duplicates
If you see multiple rows for the same company (e.g., `Zendesk Inc`, `Zendesk, Inc.`, `Zendesk A/S`), they're already identified in raw_name_variants. Mark all but one as `duplicate`.

### Multi-language Candidates
Candidates with `bilingual` in confidence_signals are stronger — they appear in both English and Danish sources, indicating broader evidence coverage.

### Trust the Confidence Scores
- **High confidence** candidates should almost always be included
- **Medium confidence** candidates need brief verification
- **Low confidence** candidates (single mention) are exploratory — verify or reject

### Using Source URLs
If a source_url is provided, click it to verify. However, note that our sample data is synthetic, so real URLs won't work in test runs.

## Output After Review

Once you complete the review:
1. Save the marked-up CSV
2. Filter to `verification_status == verified`
3. Export/count your verified candidates
4. Run deduplication on the `candidate_id` field to merge variants

## Example Review

| firm_name | confidence | status | notes |
|-----------|-----------|--------|-------|
| Unity Technologies | high | verified | Clear founding in Copenhagen + HQ move to SF |
| Netlify Inc | high | verified | Strong evidence from multiple sources |
| San Francisco. The | high | rejected | False positive - location name, not company |
| Podio ApS | medium | verified | Founding in Denmark confirmed, later US HQ |
| Zendesk Inc | medium | duplicate | Same as Zendesk A/S in row 8 |

## Questions?

Common issues:
- **Q: Can I modify candidate_id?** No, these are fixed identifiers for tracking
- **Q: Should I combine duplicate rows?** Mark them as `duplicate` in status, leave separate
- **Q: Can I add new columns?** Yes, you can add columns but keep the original ones
- **Q: What if the evidence seems incomplete?** Mark as `needs_more_info` and note what's missing

## Next Steps After Review

Once all candidates are marked:
1. Filter to `verification_status` in [verified, edge_case]
2. Run deduplication on remaining candidates
3. Export final list for publication/use
4. Save marked-up CSV as your verification record
