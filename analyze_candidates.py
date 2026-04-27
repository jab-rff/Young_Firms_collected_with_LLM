import json

with open('data/candidates/test_run_001/candidates.jsonl') as f:
    lines = [json.loads(line) for line in f]

# Filter to show likely real companies
real_candidates = [c for c in lines if len(c['normalized_name']) > 3 and c['evidence_count'] >= 2]
real_candidates.sort(key=lambda x: x['evidence_count'], reverse=True)

print(f'Total candidates: {len(lines)}')
print(f'Candidates with 2+ mentions: {len(real_candidates)}')
print()
print('Top candidates by mention count:')
for c in real_candidates[:15]:
    print(f"  {c['firm_name']:30} | normalized: {c['normalized_name']:25} | count: {c['evidence_count']}")
