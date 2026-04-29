# Pipeline Audit: Round 099

## Scope

This audit reviews the current round outputs in:

- `data/discovery/snowball_round_099.jsonl`
- `data/discovery/snowball_round_099_deduped.jsonl`
- `data/model1/snowball_round_099_candidates.jsonl`
- `data/model2/snowball_round_099_enriched.jsonl`
- `data/model3/snowball_round_099_validated.jsonl`
- `data/review/snowball_round_099_review.csv`
- `data/runs/snowball_round_099_manifest.json`

Important context for interpretation:

- Round 099 was a small debugging run, not a full search run.
- The manifest shows `max_buckets = 3` and `limit = 50`.
- Only three sector buckets were used in discovery: `biotech`, `SaaS/software`, and `fintech`.
- No destination buckets were used in this run.
- The seed list in `preliminary_data_28_04.csv` is used as an exclusion list, so known firms are intentionally filtered out of discovery.

Because of that, this audit is useful for diagnosing pipeline behavior, but it is not a clean measure of total recall.

## 1. Discovery Funnel Performance

Note: the snowball stage stores source-backed candidate rows rather than atomic web mentions. In this audit, “raw mentions” means raw discovery candidate rows.

| Stage | Rows | % of previous stage | % of raw discovery |
|---|---:|---:|---:|
| Raw discovery candidate rows | 10 | 100.0% | 100.0% |
| After deduplication | 9 | 90.0% | 90.0% |
| After Model 1 | 9 | 100.0% | 90.0% |
| After Model 2 | 9 | 100.0% | 90.0% |
| After Model 3 records | 9 | 100.0% | 90.0% |
| Final confirmed relocations (`validation_label=true`) | 2 | 22.2% | 20.0% |

### Label breakdown at Model 3

- `true`: 2
- `false`: 7
- `unclear`: 0

### Final confirmed relocations

- `Allarity Therapeutics, Inc. (formerly Allarity Therapeutics A/S / Oncology Venture A/S)`
- `United Fintech`

### What the funnel implies

- Discovery is currently the main bottleneck. The run only surfaced 10 raw rows.
- Dedup removed only 1 duplicate, which means the problem is not excessive duplication.
- Model 1 and Model 2 did not narrow the set at all in this run. They preserved recall, but they also did not screen out obvious non-cases before Model 3.
- Most of the drop happens only at final validation, where 7 of 9 surviving candidates are rejected.
- The final precision of raw discovery candidates is 20.0% for this run.

## 2. False Positive Patterns

The rejected cases are not random. They cluster around a few recurring narrative types.

### Recurring rejection patterns

- `foreign offices mistaken as HQ moves`: 5 rejected cases
- `acquisitions mistaken as relocations`: 4 rejected cases
- `legal registration only`: 3 rejected cases
- `Danish founders abroad`: 1 rejected case
- `unclear evidence / ambiguous HQ definition`: 7 rejected cases

These categories overlap. Many rejected firms trigger more than one.

### Concrete rejected examples

| Firm | Main failure mode |
|---|---|
| `IO Biotech` | Delaware parent / share-exchange / legal parent creation, but executive offices remain in Copenhagen |
| `NMD Pharma` | U.S. subsidiary and office expansion, but parent remains headquartered in Aarhus |
| `Ciklum` | Danish founder abroad, not Denmark-founded |
| `Monta` | U.S. / Americas HQ language, but looks like regional HQ rather than full relocation |
| `Templafy` | North American HQ / New York office narrative without proof of global HQ move |
| `SteelSeries` | Acquisition + foreign operations, but no firm-level HQ relocation proof |
| `Trustpilot` | UK parent / IPO restructuring, but Copenhagen still appears to remain the operational HQ |

### Diagnostic interpretation

The pipeline is currently especially good at finding:

- legal parent restructurings
- IPO-related re-domiciliation stories
- U.S. office / regional HQ announcements
- firms with strong foreign presence but unclear executive-HQ change

That is useful, but it also means the discovery prompts are gravitating toward “headquarters-adjacent” stories rather than clean relocation stories.

In other words, the pipeline is finding boundary cases more reliably than straightforward relocations.

## 3. Missing Firm Detection

## Key point

Direct comparison to `preliminary_data_28_04.csv` must be interpreted carefully.

The discovery stage explicitly excludes known firms from the seed list. That means known relocation firms are supposed to be missing from discovery outputs. This is by design, not a bug.

### What is missing from Round 099 compared with the known seed list

- Known Denmark-founded firms in seed list with `founded >= 2000`: 43
- Overlap between those known firms and Round 099 deduplicated discovery candidates: 0
- Overlap between those known firms and Round 099 final `true` cases: 0

So all 43 known post-1999 seed firms are absent from Round 099, but that mostly reflects the exclusion rule.

### Why this still matters diagnostically

Even though known firms are excluded, they still show which *types* of true relocation firms the pipeline should be able to find adjacent examples for.

The seed list suggests true relocation cases are concentrated in:

- `US` destinations: 22 known cases
- `UK` or `GB`: 8 known cases combined
- `DE`: 4 known cases
- `SE`: 3 known cases

And by industry:

- `Tech`: 14
- `Biotech`: 9
- `SAAS`: 4
- `Software`: 4
- then long-tail categories like retail, design, hospitality, adtech, consumer mobile, photography, services, transportation

### Likely reasons equivalent firms are being missed

- The run used only 3 sector buckets, so most of the known pattern space was never searched.
- No destination buckets were used, even though the seed distribution is destination-heavy, especially toward the U.S. and UK.
- The current search logic seems to over-index on legal restructuring and regional-HQ language.
- Renaming / alias complexity remains important for this domain.
  - Examples from the seed list include patterns like `Evolva Biotech (Combio)`, `Renovaro (Dandrit Biotech)`, `Veloxis Pharmaceuticals (Lifecycle Pharma)`, `Fastr (Zmags)`.
- Some known-firm analogs are likely described in media as:
  - “opened U.S. headquarters”
  - “moved leadership to London”
  - “established global HQ in New York”
  - “parent company incorporated in the UK/US”
  rather than using the exact phrase “moved headquarters”.

## 4. Discovery Coverage Analysis

Round 099 coverage is clearly too narrow to support strong discovery.

### What was actually searched

The raw discovery rows came only from:

- `sector:biotech`
- `sector:SaaS/software`
- `sector:fintech`

Raw candidate rows by bucket:

- `sector:SaaS/software`: 4
- `sector:biotech`: 3
- `sector:fintech`: 3

### What was not searched in this run

- all destination buckets
- gaming
- medtech
- logistics
- cleantech
- retail / consumer
- hospitality
- design

### Why this matters

The known seed list contains real relocation examples in categories not covered here:

- design
- hospitality
- adtech
- retail
- consumer mobile
- photography
- transportation
- services

It also contains many destination patterns not covered here:

- U.S.
- UK
- Germany
- Sweden
- Switzerland
- Netherlands
- Italy
- Spain
- South Africa
- Ghana

### Missing query bucket families worth adding or emphasizing

- acquisition-related relocation
- founder relocation narratives
- IPO / listing / holding-company restructuring narratives
- stealth startup / low-coverage startup narratives
- Danish media sources
- foreign media sources
- city-pair searches
- specific industry searches beyond software / biotech / fintech

### High-priority missing bucket ideas

- `destination:US` with executive-HQ wording
- `destination:UK` with London / Cambridge / principal office wording
- `destination:Germany`
- `destination:Sweden`
- `destination:Netherlands`
- `sector:design`
- `sector:retail/consumer`
- `sector:hospitality`
- `sector:gaming`
- `sector:medtech`

## 5. High-Value Candidate Expansion Ideas

The fastest way to find more true firms is not to redesign the architecture. It is to widen discovery coverage while keeping the same downstream conservative gate.

### Better search query families

- Queries that combine founding in Denmark with executive-location language:
  - `founded in Copenhagen principal executive offices New York`
  - `Danish company principal executive offices London`
  - `Denmark-founded company headquartered in Boston`
- Queries that combine destination country with global-HQ phrasing:
  - `Danish-founded global headquarters London`
  - `Danish startup moved headquarters to New York`
- Queries that combine parent-company restructuring with location change:
  - `Danish company Delaware parent principal executive offices`
  - `Danish company UK parent headquarters London`
- Queries that target “group HQ” and “principal executive offices”, not just “headquarters”.

### Better snowball logic

- Use confirmed `true` firms as pattern seeds, not only as exclusions.
- For each confirmed case, generate adjacent searches by:
  - sector
  - destination country
  - relocation mechanism
  - funding / IPO context
- Example:
  - `Allarity` suggests more biotech firms with U.S. public-parent restructurings.
  - `United Fintech` suggests more fintech groups with Danish operating roots and UK holding structures.

### Public datasets and corpora to mine

- Danish startup databases and founder lists
- funding / venture databases
- company registry-based lists with Danish origin + foreign current HQ
- IPO and prospectus corpora
- M&A and legal-advisory press releases
- startup awards / accelerator portfolios
- archived startup ecosystem lists

### Better use of Børsen-style and Danish-language material

- The seed file already suggests Børsen has yielded many true cases.
- That implies Danish-language source targeting is probably underused in the current snowball stage.
- Good additions:
  - Danish media query families
  - Danish relocation terms
  - Danish reporting about founder moves, group restructurings, and foreign HQ announcements

### Practical query themes to add

- relocation-specific terms:
  - `principal executive offices`
  - `group headquarters`
  - `global headquarters`
  - `moved executive team`
  - `redomiciled parent`
  - `corporate reorganization`
  - `listed in London/Nasdaq after Danish founding`
- city-pair searches:
  - `Copenhagen to London`
  - `Aarhus to New York`
  - `Copenhagen to San Francisco`
  - `Denmark to Cambridge MA`
- event-driven searches:
  - `IPO + headquarters`
  - `Series B + moved to US`
  - `opened US headquarters`
  - `formed Delaware parent`

## 6. Precision vs Recall Diagnosis

For Round 099, the pipeline looks **recall-constrained at discovery and strict at the end**, not wildly noisy.

### Why

- The run only found 10 raw discovery rows.
- 9 of those survived through Model 2 unchanged.
- Model 3 rejected 7 of 9.
- The rejected cases are mostly plausible boundary cases, not garbage.

### Diagnosis

- Discovery is too narrow for strong recall.
- Downstream validation is conservative, which is appropriate for this research definition.
- The current system is not “exploding” with false positives.
- The larger problem is that it is not exploring enough of the search space.

So the overall pipeline is best described as:

- **balanced to slightly strict downstream**
- **too narrow upstream**

## 7. Actionable Recommendations

These are ordered by practical value, not by architectural ambition.

### Quick wins

- Run more than 3 discovery buckets.
- Use destination buckets in every serious run, especially `US`, `UK`, `Germany`, `Sweden`, and `Netherlands`.
- Add query wording around:
  - `principal executive offices`
  - `global headquarters`
  - `group headquarters`
  - `Delaware parent`
  - `UK parent`
- Add Danish-language discovery prompts and Danish media-oriented queries.
- Add alias-aware discovery prompts:
  - `formerly`
  - `rebranded as`
  - `X / Y`
  - `old legal name`

### Medium improvements

- Add query families specifically for:
  - IPO / listing relocations
  - acquisition-adjacent relocations
  - U.S. expansion that later became executive-HQ migration
  - founder/CEO relocation narratives
- Add a lightweight “adjacent-firm expansion” stage using confirmed true firms as templates.
- Add more destination-country prompts before adding more downstream filtering.
- Track which buckets yield `true` firms versus only false positives, then reweight future runs.

### Major structural improvements

- Build a small library of relocation narrative templates from confirmed cases:
  - operational HQ move
  - legal parent + executive HQ shift
  - IPO-related re-domiciliation with real executive move
  - acquisition-linked relocation where the operating company persists
- Use those templates explicitly in discovery prompts to search for analog firms.
- Add a recall-oriented evidence harvesting layer before Model 1 that searches for:
  - company timeline pages
  - investor pages
  - principal executive office wording
  - archived “about us” pages

This would improve discovery without rewriting the architecture.

## Bottom Line

Round 099 does not show a broken pipeline. It shows an **under-expanded discovery run**.

The strongest signals from this audit are:

- the run was too small to explore the space properly
- discovery is finding interesting edge cases, but not enough volume
- the current candidate pool is dominated by:
  - regional HQ announcements
  - legal parent restructurings
  - Danish-founder-abroad cases
- destination coverage is the biggest missing lever
- the best path to more true firms is to broaden query coverage and snowball from confirmed cases, not to make Model 3 looser

If the goal is to find more true relocation firms without blowing up false positives, the next best move is:

1. run discovery across the full bucket set
2. add destination-heavy and Danish-language query families
3. add relocation-mechanism query families such as IPO, parent-company restructuring, and executive-office phrasing
4. keep Model 3 conservative
