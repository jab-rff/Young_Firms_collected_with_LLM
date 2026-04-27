"""Export candidates to CSV format for manual review."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from src.data_models import CandidateFirm, Mention


def prepare_review_row(
    candidate: CandidateFirm,
    mentions: list[Mention],
) -> dict[str, str]:
    """Prepare a candidate row for CSV export with all review context.
    
    Args:
        candidate: The candidate firm
        mentions: All mentions linked to this candidate
        
    Returns:
        Dictionary with all fields needed for review
    """
    # Get unique languages
    languages = sorted(set(m.language for m in mentions))
    lang_label = "/".join(languages) if languages else "unknown"
    
    # Get unique sources
    unique_sources = sorted(set(m.source_name for m in mentions if m.source_name))
    
    # Get unique URLs
    unique_urls = sorted(set(m.url for m in mentions if m.url))
    
    # Get all evidence texts and pick best snippets
    evidence_snippets = [m.evidence_text for m in mentions if m.evidence_text]
    
    # Find the most informative snippet (longest or contains key phrases)
    key_evidence = ""
    if evidence_snippets:
        # Prefer snippets mentioning "founded" or "moved"
        for snippet in evidence_snippets:
            snippet_lower = snippet.lower()
            if ("founded" in snippet_lower or "move" in snippet_lower) and "denmark" in snippet_lower:
                key_evidence = snippet[:500]  # Truncate to 500 chars
                break
        
        # If no key snippet found, use the longest
        if not key_evidence and evidence_snippets:
            key_evidence = max(evidence_snippets, key=len)[:500]
    
    # Combine top 2-3 evidence snippets for context
    evidence_context = " | ".join(evidence_snippets[:2])[:800]
    
    # Confidence signals
    confidence_signals = []
    if len(mentions) >= 3:
        confidence_signals.append(f"3+ mentions ({len(mentions)})")
    if len(unique_sources) >= 2:
        confidence_signals.append(f"multi-source ({len(unique_sources)})")
    if len(languages) > 1:
        confidence_signals.append("bilingual")
    
    confidence_score = (
        "high"
        if len(unique_sources) >= 2 and len(mentions) >= 2
        else ("medium" if len(mentions) >= 2 else "low")
    )
    
    return {
        "candidate_id": candidate.candidate_id,
        "firm_name": candidate.firm_name,
        "normalized_name": candidate.normalized_name,
        "mention_count": str(candidate.evidence_count),
        "languages": lang_label,
        "unique_sources": str(len(unique_sources)),
        "sources": "; ".join(unique_sources),
        "source_urls": "; ".join(unique_urls[:3]),  # Top 3 URLs
        "key_evidence": key_evidence,
        "raw_name_variants": "; ".join(candidate.raw_name_variants[:5]),  # Top 5 variants
        "confidence": confidence_score,
        "confidence_signals": "; ".join(confidence_signals),
        "verification_status": "",  # For reviewer to fill in
        "reviewer_notes": "",  # For reviewer to fill in
    }


def export_candidates_to_csv(
    output_path: Path,
    candidates: list[CandidateFirm],
    mentions: list[Mention],
) -> None:
    """Export candidates to CSV with full review context.
    
    Args:
        output_path: Path to write CSV file
        candidates: List of candidates
        mentions: List of mentions (should include all mentions for each candidate)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create mention lookup by mention_id
    mentions_by_id = {m.mention_id: m for m in mentions}
    
    # Prepare rows
    rows = []
    for candidate in candidates:
        # Get mentions for this candidate
        candidate_mentions = [
            mentions_by_id[mid] for mid in candidate.mention_ids
            if mid in mentions_by_id
        ]
        row = prepare_review_row(candidate, candidate_mentions)
        rows.append(row)
    
    # Sort by mention count (descending) then by firm name
    rows.sort(key=lambda r: (-int(r["mention_count"]), r["firm_name"]))
    
    # Write CSV
    if not rows:
        print(f"No candidates to export to {output_path}")
        return
    
    fieldnames = [
        "candidate_id",
        "firm_name",
        "normalized_name",
        "mention_count",
        "languages",
        "unique_sources",
        "sources",
        "source_urls",
        "key_evidence",
        "raw_name_variants",
        "confidence",
        "confidence_signals",
        "verification_status",
        "reviewer_notes",
    ]
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Exported {len(rows)} candidates to {output_path}")
    print(f"  - {sum(1 for r in rows if r['confidence'] == 'high')} high confidence")
    print(f"  - {sum(1 for r in rows if r['confidence'] == 'medium')} medium confidence")
    print(f"  - {sum(1 for r in rows if r['confidence'] == 'low')} low confidence")


def export_candidates_filtered_csv(
    output_path: Path,
    candidates: list[CandidateFirm],
    mentions: list[Mention],
    min_mention_count: int = 2,
    min_unique_sources: Optional[int] = None,
) -> None:
    """Export only high-quality candidates to CSV (filtered by mention count and sources).
    
    Args:
        output_path: Path to write CSV
        candidates: List of candidates
        mentions: List of mentions
        min_mention_count: Only include candidates with at least this many mentions
        min_unique_sources: Only include candidates with at least this many unique sources
    """
    # Filter candidates
    filtered_candidates = [c for c in candidates if c.evidence_count >= min_mention_count]
    
    if min_unique_sources:
        # Create mention lookup by mention_id
        mentions_by_id = {m.mention_id: m for m in mentions}
        
        # Further filter by unique sources
        filtered_final = []
        for candidate in filtered_candidates:
            candidate_mentions = [
                mentions_by_id[mid] for mid in candidate.mention_ids
                if mid in mentions_by_id
            ]
            unique_sources = set(m.source_name for m in candidate_mentions if m.source_name)
            if len(unique_sources) >= min_unique_sources:
                filtered_final.append(candidate)
        
        filtered_candidates = filtered_final
    
    print(f"Filtered: {len(candidates)} → {len(filtered_candidates)} candidates")
    
    export_candidates_to_csv(output_path, filtered_candidates, mentions)
