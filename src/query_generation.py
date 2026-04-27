"""Broad query generation for recall-first discovery."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data_models import Query


def generate_seed_queries() -> list[Query]:
    """Generate comprehensive search queries in English and Danish.
    
    Returns a list of diverse, high-recall queries organized by family and priority.
    Queries are designed to surface companies that:
    - Were founded in Denmark (Copenhagen, Aarhus, etc.)
    - Later moved their executive HQ abroad
    - May have been acquired or merged with international companies
    
    Query families:
    - hq_move: Direct HQ relocation language
    - startup_hq: Startup + headquarters combinations
    - relocation: Generic relocation context
    - location_pairs: Specific city/country pairs (Copenhagen → SF, etc.)
    - acquisition: M&A context combined with founding
    - industry_dk: Industry + Danish origin combinations
    """
    created_at = datetime.now(timezone.utc).isoformat()
    
    # Each tuple: (query_id, query_text, language, family, priority)
    # Priority: 1=highest recall, 2=high, 3=standard, 4=exploratory
    query_specs = [
        # English: Direct HQ move language (priority 1)
        ("q_en_001", '"founded in Denmark" "headquarters"', "en", "hq_move", 1),
        ("q_en_002", '"Denmark" "moved headquarters"', "en", "hq_move", 1),
        ("q_en_003", '"Copenhagen" "relocated headquarters"', "en", "hq_move", 1),
        ("q_en_004", '"Danish startup" "moved HQ"', "en", "hq_move", 1),
        ("q_en_005", '"founded in Copenhagen" "now headquartered"', "en", "hq_move", 1),
        
        # English: Startup + HQ combinations (priority 1-2)
        ("q_en_006", '"Danish startup" "headquarters" "United States"', "en", "startup_hq", 1),
        ("q_en_007", '"Danish company" "moved" "San Francisco"', "en", "startup_hq", 1),
        ("q_en_008", '"Danish tech" "headquarters" abroad', "en", "startup_hq", 2),
        ("q_en_009", '"Danish founded" "US headquarters"', "en", "startup_hq", 2),
        
        # English: Location pairs (priority 1-2)
        ("q_en_010", '"Copenhagen" "San Francisco" "founder"', "en", "location_pairs", 1),
        ("q_en_011", '"Copenhagen" "Palo Alto" company', "en", "location_pairs", 1),
        ("q_en_012", '"Copenhagen" "Boston" "headquarters"', "en", "location_pairs", 2),
        ("q_en_013", '"Copenhagen" "New York" company moved', "en", "location_pairs", 2),
        ("q_en_014", '"Copenhagen" "London" "headquarters"', "en", "location_pairs", 2),
        ("q_en_015", '"Denmark" "Chicago" company HQ', "en", "location_pairs", 2),
        
        # English: Generic relocation (priority 2)
        ("q_en_016", '"Denmark" "relocated headquarters" company', "en", "relocation", 2),
        ("q_en_017", 'Danish company "moved" headquarters', "en", "relocation", 2),
        ("q_en_018", '"Danish" "executive HQ" abroad', "en", "relocation", 2),
        
        # English: Acquisition context (priority 2-3)
        ("q_en_019", '"Danish company" acquired "headquarters"', "en", "acquisition", 2),
        ("q_en_020", '"founded in Denmark" acquired "moved"', "en", "acquisition", 2),
        ("q_en_021", '"Danish startup" "acquired by" US company', "en", "acquisition", 3),
        
        # English: Industry + Danish combinations (priority 2-3)
        ("q_en_022", '"Danish software" company "headquarters"', "en", "industry_dk", 2),
        ("q_en_023", '"Danish SaaS" company moved', "en", "industry_dk", 2),
        ("q_en_024", '"Danish tech company" relocated', "en", "industry_dk", 2),
        ("q_en_025", '"Danish startup" US tech', "en", "industry_dk", 3),
        
        # Danish: Direct HQ move language (priority 1)
        ("q_da_001", '"grundlagt i Danmark" "hovedkontor"', "da", "hq_move", 1),
        ("q_da_002", '"Danmark" "flyttede hovedkvarter"', "da", "hq_move", 1),
        ("q_da_003", '"København" "flyttet hovedkvarter"', "da", "hq_move", 1),
        ("q_da_004", '"dansk startup" "hovedkvarter" udlandet', "da", "hq_move", 1),
        ("q_da_005", '"grundlagt København" "nu hovedkvarter"', "da", "hq_move", 1),
        
        # Danish: Startup + HQ combinations (priority 1-2)
        ("q_da_006", '"dansk startup" "hovedkvarter" "USA"', "da", "startup_hq", 1),
        ("q_da_007", '"dansk virksomhed" "San Francisco"', "da", "startup_hq", 1),
        ("q_da_008", '"dansk tech" "hovedkvarter" "Silicon Valley"', "da", "startup_hq", 2),
        ("q_da_009", '"dansk grundlagt" "amerikanske" hovedkvarter', "da", "startup_hq", 2),
        
        # Danish: Location pairs (priority 1-2)
        ("q_da_010", '"København" "San Francisco" virksomhed', "da", "location_pairs", 1),
        ("q_da_011", '"København" "Palo Alto" stifter', "da", "location_pairs", 1),
        ("q_da_012", '"København" "Boston" "hovedkvarter"', "da", "location_pairs", 2),
        ("q_da_013", '"København" "New York" virksomhed', "da", "location_pairs", 2),
        ("q_da_014", '"Danmark" "Chicago" selskab', "da", "location_pairs", 2),
        
        # Danish: Generic relocation (priority 2)
        ("q_da_015", '"Danmark" "ledelsesmæssigt" hovedkvarter udlandet', "da", "relocation", 2),
        ("q_da_016", 'dansk virksomhed "flyttet" hovedkvarter', "da", "relocation", 2),
        ("q_da_017", '"dansk" "ledelses HQ" udlandet', "da", "relocation", 2),
        
        # Danish: Acquisition context (priority 2-3)
        ("q_da_018", '"dansk virksomhed" opkøbt "hovedkvarter"', "da", "acquisition", 2),
        ("q_da_019", '"grundlagt Danmark" opkøbt "flyttet"', "da", "acquisition", 2),
        ("q_da_020", '"dansk startup" "købt af" amerikansk', "da", "acquisition", 3),
        
        # Danish: Industry + Danish combinations (priority 2-3)
        ("q_da_021", '"dansk software" virksomhed "hovedkvarter"', "da", "industry_dk", 2),
        ("q_da_022", '"dansk SaaS" virksomhed flyttet', "da", "industry_dk", 2),
        ("q_da_023", '"dansk tech-virksomhed" "hovedkvarter"', "da", "industry_dk", 2),
        ("q_da_024", '"dansk startup" USA teknologi', "da", "industry_dk", 3),
    ]
    
    return [
        Query(
            query_id=query_id,
            query_text=query_text,
            language=language,
            family=family,
            created_at=created_at,
        )
        for query_id, query_text, language, family, _priority in query_specs
    ]


def generate_extended_queries() -> list[Query]:
    """Generate additional exploratory queries for higher recall.
    
    These queries are less structured and more exploratory, designed to capture
    edge cases and less obvious mentions of HQ relocations.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    
    query_specs = [
        # English: Broader exploratory queries
        ("q_en_ex_001", "Danish company Silicon Valley", "en", "exploratory", 3),
        ("q_en_ex_002", "Danish founders moved headquarters", "en", "exploratory", 3),
        ("q_en_ex_003", "Copenhagen startup relocated", "en", "exploratory", 3),
        ("q_en_ex_004", "Danish SaaS company US expansion", "en", "exploratory", 3),
        ("q_en_ex_005", "Danish tech unicorn moved HQ", "en", "exploratory", 3),
        
        # Danish: Broader exploratory queries
        ("q_da_ex_001", "dansk virksomhed Silicon Valley", "da", "exploratory", 3),
        ("q_da_ex_002", "danske stiftere hovedkvarter", "da", "exploratory", 3),
        ("q_da_ex_003", "København startup Amerika", "da", "exploratory", 3),
        ("q_da_ex_004", "dansk software-virksomhed USA", "da", "exploratory", 3),
    ]
    
    return [
        Query(
            query_id=query_id,
            query_text=query_text,
            language=language,
            family=family,
            created_at=created_at,
        )
        for query_id, query_text, language, family, _priority in query_specs
    ]


def generate_all_queries(include_exploratory: bool = False) -> list[Query]:
    """Generate all seed queries, optionally including exploratory queries.
    
    Args:
        include_exploratory: If True, include broader exploratory queries (lower precision).
        
    Returns:
        Sorted list of queries by family and language.
    """
    queries = generate_seed_queries()
    if include_exploratory:
        queries.extend(generate_extended_queries())
    
    # Sort by language, then family, then query_id for consistent output
    return sorted(queries, key=lambda q: (q.language, q.family, q.query_id))


def queries_by_family(queries: list[Query]) -> dict[str, list[Query]]:
    """Group queries by their family."""
    grouped: dict[str, list[Query]] = {}
    for query in queries:
        if query.family not in grouped:
            grouped[query.family] = []
        grouped[query.family].append(query)
    return grouped


def queries_summary(queries: list[Query]) -> str:
    """Generate a human-readable summary of the query set."""
    by_family = queries_by_family(queries)
    by_language = {}
    for query in queries:
        lang = query.language
        if lang not in by_language:
            by_language[lang] = 0
        by_language[lang] += 1
    
    lines = [
        f"Query Generation Summary",
        f"{'='*50}",
        f"Total queries: {len(queries)}",
        f"Languages: {', '.join(sorted(by_language.keys()))} ({', '.join(f'{lang}:{count}' for lang, count in sorted(by_language.items()))})",
        f"Families: {', '.join(sorted(by_family.keys()))}",
        f"",
    ]
    
    for family in sorted(by_family.keys()):
        queries_in_family = by_family[family]
        en_count = sum(1 for q in queries_in_family if q.language == "en")
        da_count = sum(1 for q in queries_in_family if q.language == "da")
        lines.append(f"  {family:20} | EN: {en_count:2} | DA: {da_count:2} | Total: {len(queries_in_family):2}")
    
    return "\n".join(lines)
