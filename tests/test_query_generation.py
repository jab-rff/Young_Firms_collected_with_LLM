"""Tests for query generation."""

import pytest
from src.query_generation import (
    generate_seed_queries,
    generate_extended_queries,
    generate_all_queries,
    queries_by_family,
    queries_summary,
)


def test_generate_seed_queries():
    """Test seed query generation."""
    queries = generate_seed_queries()
    
    assert len(queries) == 49
    assert all(q.query_id.startswith(("q_en_", "q_da_")) for q in queries)
    assert all(q.language in ("en", "da") for q in queries)
    assert all(q.family in ("hq_move", "startup_hq", "relocation", "location_pairs", "acquisition", "industry_dk") for q in queries)
    assert all(q.created_at for q in queries)


def test_generate_seed_queries_balanced():
    """Test that seed queries are balanced between languages."""
    queries = generate_seed_queries()
    
    en_count = sum(1 for q in queries if q.language == "en")
    da_count = sum(1 for q in queries if q.language == "da")
    
    assert en_count == 25
    assert da_count == 24


def test_generate_extended_queries():
    """Test extended (exploratory) query generation."""
    queries = generate_extended_queries()
    
    assert len(queries) == 9
    assert all(q.family == "exploratory" for q in queries)
    assert all(q.query_id.startswith(("q_en_ex_", "q_da_ex_")) for q in queries)


def test_generate_all_queries():
    """Test combined query generation."""
    seed_only = generate_all_queries(include_exploratory=False)
    extended = generate_all_queries(include_exploratory=True)
    
    assert len(seed_only) == 49
    assert len(extended) == 58


def test_queries_by_family():
    """Test family grouping."""
    queries = generate_seed_queries()
    by_family = queries_by_family(queries)
    
    assert len(by_family) == 6
    assert all(family in by_family for family in ["hq_move", "startup_hq", "relocation", "location_pairs", "acquisition", "industry_dk"])
    
    # Verify count
    total = sum(len(q) for q in by_family.values())
    assert total == len(queries)


def test_queries_summary():
    """Test summary generation."""
    queries = generate_seed_queries()
    summary = queries_summary(queries)
    
    assert "Query Generation Summary" in summary
    assert "49" in summary
    assert "en" in summary
    assert "da" in summary
    assert "hq_move" in summary


def test_query_uniqueness():
    """Test that all queries have unique IDs and texts."""
    queries = generate_all_queries(include_exploratory=True)
    
    query_ids = [q.query_id for q in queries]
    query_texts = [q.query_text for q in queries]
    
    assert len(query_ids) == len(set(query_ids)), "Duplicate query IDs found"
    assert len(query_texts) == len(set(query_texts)), "Duplicate query texts found"


def test_query_language_distribution():
    """Test language distribution across families."""
    queries = generate_seed_queries()
    
    for family in ["hq_move", "startup_hq", "location_pairs", "acquisition", "industry_dk", "relocation"]:
        family_queries = [q for q in queries if q.family == family]
        
        if family != "relocation":  # Some families have 4 each, some 5+5 split
            en = sum(1 for q in family_queries if q.language == "en")
            da = sum(1 for q in family_queries if q.language == "da")
            
            # All families should have balanced EN/DA coverage
            assert abs(en - da) <= 1, f"Family {family} has imbalanced languages: EN={en}, DA={da}"
