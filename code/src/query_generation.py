"""Broad query generation for recall-first discovery."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data_models import Query


def generate_seed_queries() -> list[Query]:
    """Generate HQ-relocation-first search queries in English and Danish."""
    created_at = datetime.now(timezone.utc).isoformat()

    query_specs = [
        # explicit_hq_move
        ("q_en_001", '"Danish startup moved headquarters abroad"', "en", "explicit_hq_move"),
        ("q_en_002", '"Danish company moved HQ from Copenhagen"', "en", "explicit_hq_move"),
        ("q_en_003", '"Danish company relocated headquarters to the US"', "en", "explicit_hq_move"),
        ("q_en_004", '"Danish startup moved main office to London"', "en", "explicit_hq_move"),
        ("q_da_001", '"dansk virksomhed flyttede hovedkvarter til udlandet"', "da", "explicit_hq_move"),
        ("q_da_002", '"dansk startup flyttede hovedkontor til USA"', "da", "explicit_hq_move"),
        ("q_da_003", '"dansk firma flyttede hovedkontor fra København"', "da", "explicit_hq_move"),
        ("q_da_004", '"dansk virksomhed flyttede ledelsen til udlandet"', "da", "explicit_hq_move"),

        # implicit_foreign_hq
        ("q_en_005", '"founded in Denmark now headquartered in"', "en", "implicit_foreign_hq"),
        ("q_en_006", '"Danish-founded company now based in the United States"', "en", "implicit_foreign_hq"),
        ("q_en_007", '"Denmark-founded software company headquarters San Francisco"', "en", "implicit_foreign_hq"),
        ("q_en_008", '"Danish company now headquartered in London"', "en", "implicit_foreign_hq"),
        ("q_da_005", '"grundlagt i Danmark nu hovedkvarter i udlandet"', "da", "implicit_foreign_hq"),
        ("q_da_006", '"dansk softwarevirksomhed nu baseret i USA"', "da", "implicit_foreign_hq"),
        ("q_da_007", '"dansk virksomhed nu hovedkvarter i London"', "da", "implicit_foreign_hq"),
        ("q_da_008", '"grundlagt i Danmark nu baseret i udlandet"', "da", "implicit_foreign_hq"),

        # copenhagen_to_foreign_hq
        ("q_en_009", '"founded in Copenhagen headquartered in San Francisco"', "en", "copenhagen_to_foreign_hq"),
        ("q_en_010", '"Copenhagen company now headquartered in New York"', "en", "copenhagen_to_foreign_hq"),
        ("q_en_011", '"Copenhagen startup headquartered in London"', "en", "copenhagen_to_foreign_hq"),
        ("q_en_012", '"Copenhagen founded company headquarters Boston"', "en", "copenhagen_to_foreign_hq"),
        ("q_da_009", '"grundlagt i København hovedkvarter i San Francisco"', "da", "copenhagen_to_foreign_hq"),
        ("q_da_010", '"København virksomhed nu hovedkvarter i New York"', "da", "copenhagen_to_foreign_hq"),
        ("q_da_011", '"København startup nu hovedkontor i London"', "da", "copenhagen_to_foreign_hq"),
        ("q_da_012", '"grundlagt i København nu baseret i USA"', "da", "copenhagen_to_foreign_hq"),

        # danish_founded_now_abroad
        ("q_en_013", '"Danish-founded company now abroad headquarters"', "en", "danish_founded_now_abroad"),
        ("q_en_014", '"founded in Denmark company now based abroad"', "en", "danish_founded_now_abroad"),
        ("q_en_015", '"Danish startup now operating from the US"', "en", "danish_founded_now_abroad"),
        ("q_en_016", '"founded in Denmark now operating from London"', "en", "danish_founded_now_abroad"),
        ("q_da_013", '"grundlagt i Danmark nu virksomhed i udlandet"', "da", "danish_founded_now_abroad"),
        ("q_da_014", '"dansk startup nu baseret i USA"', "da", "danish_founded_now_abroad"),
        ("q_da_015", '"grundlagt i Danmark nu opererer fra London"', "da", "danish_founded_now_abroad"),
        ("q_da_016", '"dansk virksomhed nu opererer fra udlandet"', "da", "danish_founded_now_abroad"),

        # executive_base_abroad
        ("q_en_017", '"Danish company executive base abroad"', "en", "executive_base_abroad"),
        ("q_en_018", '"founded in Denmark main office in the US"', "en", "executive_base_abroad"),
        ("q_en_019", '"Danish startup main operations in London"', "en", "executive_base_abroad"),
        ("q_en_020", '"Denmark-founded company leadership moved abroad"', "en", "executive_base_abroad"),
        ("q_da_017", '"dansk virksomhed ledelsesbase i udlandet"', "da", "executive_base_abroad"),
        ("q_da_018", '"grundlagt i Danmark hovedledelse i USA"', "da", "executive_base_abroad"),
        ("q_da_019", '"dansk startup hovedkontor og drift i London"', "da", "executive_base_abroad"),
        ("q_da_020", '"dansk virksomhed flyttede hoveddriften til udlandet"', "da", "executive_base_abroad"),

        # acquisition_overlap (small secondary family only)
        ("q_en_021", '"Danish startup moved headquarters then acquired"', "en", "acquisition_overlap"),
        ("q_en_022", '"founded in Denmark moved HQ acquired"', "en", "acquisition_overlap"),
        ("q_da_021", '"dansk virksomhed flyttede hovedkvarter og blev opkøbt"', "da", "acquisition_overlap"),
        ("q_da_022", '"grundlagt i Danmark flyttede hovedkontor og blev opkøbt"', "da", "acquisition_overlap"),
    ]

    return [
        Query(
            query_id=query_id,
            query_text=query_text,
            language=language,
            family=family,
            created_at=created_at,
        )
        for query_id, query_text, language, family in query_specs
    ]


def generate_extended_queries() -> list[Query]:
    """Generate additional HQ-relocation exploratory queries for higher recall."""
    created_at = datetime.now(timezone.utc).isoformat()

    query_specs = [
        ("q_en_ex_001", "Danish startup now headquartered abroad", "en", "exploratory"),
        ("q_en_ex_002", "Copenhagen company main office United States", "en", "exploratory"),
        ("q_en_ex_003", "founded in Denmark executive base London", "en", "exploratory"),
        ("q_en_ex_004", "Danish software company operating from San Francisco", "en", "exploratory"),
        ("q_da_ex_001", "dansk startup nu hovedkvarter i udlandet", "da", "exploratory"),
        ("q_da_ex_002", "København virksomhed hovedkontor USA", "da", "exploratory"),
        ("q_da_ex_003", "grundlagt i Danmark ledelse i London", "da", "exploratory"),
        ("q_da_ex_004", "dansk softwarevirksomhed opererer fra San Francisco", "da", "exploratory"),
    ]

    return [
        Query(
            query_id=query_id,
            query_text=query_text,
            language=language,
            family=family,
            created_at=created_at,
        )
        for query_id, query_text, language, family in query_specs
    ]


def generate_all_queries(include_exploratory: bool = False) -> list[Query]:
    """Generate all seed queries, optionally including exploratory queries."""
    queries = generate_seed_queries()
    if include_exploratory:
        queries.extend(generate_extended_queries())
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
    by_language: dict[str, int] = {}
    for query in queries:
        by_language[query.language] = by_language.get(query.language, 0) + 1

    lines = [
        "Query Generation Summary",
        f"{'=' * 50}",
        f"Total queries: {len(queries)}",
        "Languages: "
        + ", ".join(sorted(by_language.keys()))
        + " ("
        + ", ".join(f"{lang}:{count}" for lang, count in sorted(by_language.items()))
        + ")",
        f"Families: {', '.join(sorted(by_family.keys()))}",
        "",
    ]

    for family in sorted(by_family.keys()):
        queries_in_family = by_family[family]
        en_count = sum(1 for q in queries_in_family if q.language == "en")
        da_count = sum(1 for q in queries_in_family if q.language == "da")
        lines.append(f"  {family:24} | EN: {en_count:2} | DA: {da_count:2} | Total: {len(queries_in_family):2}")

    return "\n".join(lines)
