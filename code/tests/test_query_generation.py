"""Tests for query generation."""

from src.query_generation import (
    generate_all_queries,
    generate_extended_queries,
    generate_seed_queries,
    queries_by_family,
    queries_summary,
)


EXPECTED_FAMILIES = {
    "explicit_hq_move",
    "implicit_foreign_hq",
    "copenhagen_to_foreign_hq",
    "danish_founded_now_abroad",
    "executive_base_abroad",
    "acquisition_overlap",
}


def test_generate_seed_queries() -> None:
    queries = generate_seed_queries()

    assert len(queries) == 44
    assert all(q.query_id.startswith(("q_en_", "q_da_")) for q in queries)
    assert all(q.language in ("en", "da") for q in queries)
    assert all(q.family in EXPECTED_FAMILIES for q in queries)
    assert all(q.created_at for q in queries)


def test_generate_seed_queries_balanced() -> None:
    queries = generate_seed_queries()

    en_count = sum(1 for q in queries if q.language == "en")
    da_count = sum(1 for q in queries if q.language == "da")

    assert en_count == 22
    assert da_count == 22


def test_generate_extended_queries() -> None:
    queries = generate_extended_queries()

    assert len(queries) == 8
    assert all(q.family == "exploratory" for q in queries)
    assert all(q.query_id.startswith(("q_en_ex_", "q_da_ex_")) for q in queries)


def test_generate_all_queries() -> None:
    seed_only = generate_all_queries(include_exploratory=False)
    extended = generate_all_queries(include_exploratory=True)

    assert len(seed_only) == 44
    assert len(extended) == 52


def test_queries_by_family() -> None:
    queries = generate_seed_queries()
    by_family = queries_by_family(queries)

    assert set(by_family.keys()) == EXPECTED_FAMILIES
    assert len(by_family["acquisition_overlap"]) == 4
    total = sum(len(group) for group in by_family.values())
    assert total == len(queries)


def test_queries_summary() -> None:
    queries = generate_seed_queries()
    summary = queries_summary(queries)

    assert "Query Generation Summary" in summary
    assert "44" in summary
    assert "en:22" in summary
    assert "da:22" in summary
    assert "explicit_hq_move" in summary
    assert "acquisition_overlap" in summary


def test_query_uniqueness() -> None:
    queries = generate_all_queries(include_exploratory=True)

    query_ids = [q.query_id for q in queries]
    query_texts = [q.query_text for q in queries]

    assert len(query_ids) == len(set(query_ids)), "Duplicate query IDs found"
    assert len(query_texts) == len(set(query_texts)), "Duplicate query texts found"


def test_query_language_distribution() -> None:
    queries = generate_seed_queries()

    for family in EXPECTED_FAMILIES:
        family_queries = [q for q in queries if q.family == family]
        en = sum(1 for q in family_queries if q.language == "en")
        da = sum(1 for q in family_queries if q.language == "da")
        assert en == da, f"Family {family} should be balanced: EN={en}, DA={da}"


def test_acquisition_overlap_is_small_relative_to_hq_move_families() -> None:
    queries = generate_seed_queries()
    by_family = queries_by_family(queries)

    acquisition_count = len(by_family["acquisition_overlap"])
    primary_hq_count = sum(
        len(by_family[family])
        for family in {
            "explicit_hq_move",
            "implicit_foreign_hq",
            "copenhagen_to_foreign_hq",
            "danish_founded_now_abroad",
            "executive_base_abroad",
        }
    )

    assert acquisition_count < 0.2 * primary_hq_count


def test_seed_queries_include_required_primary_examples() -> None:
    query_texts = {q.query_text for q in generate_seed_queries()}

    for expected in {
        '"Danish startup moved headquarters abroad"',
        '"Danish company moved HQ from Copenhagen"',
        '"founded in Denmark now headquartered in"',
        '"founded in Copenhagen headquartered in San Francisco"',
        '"Danish-founded company now based in the United States"',
        '"Denmark-founded software company headquarters San Francisco"',
        '"Danish startup moved main office to London"',
        '"Danish company relocated headquarters to the US"',
        '"dansk virksomhed flyttede hovedkvarter til udlandet"',
        '"dansk startup flyttede hovedkontor til USA"',
        '"grundlagt i Danmark nu hovedkvarter i udlandet"',
        '"grundlagt i København hovedkvarter i San Francisco"',
        '"dansk firma flyttede hovedkontor fra København"',
        '"dansk softwarevirksomhed nu baseret i USA"',
        '"dansk virksomhed flyttede hoveddriften til udlandet"',
        '"Danish startup moved headquarters then acquired"',
        '"dansk virksomhed flyttede hovedkvarter og blev opkøbt"',
        '"founded in Denmark moved HQ acquired"',
    }:
        assert expected in query_texts


def test_avoid_acquisition_only_seed_queries() -> None:
    query_texts = {q.query_text for q in generate_seed_queries()}

    rejected = {
        '"dansk virksomhed" opkøbt "hovedkvarter"',
        '"grundlagt Danmark" opkøbt "flyttet"',
        '"dansk startup" "købt af" amerikansk',
        '"Danish startup" "acquired by" US company',
        '"Danish company" acquired "headquarters"',
    }
    for text in rejected:
        assert text not in query_texts
