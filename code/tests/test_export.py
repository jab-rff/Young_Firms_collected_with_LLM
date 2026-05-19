"""Tests for CSV export functionality."""

import pytest
import csv
import json
from pathlib import Path
from src.data_models import CandidateFirm, Mention
from src.export import prepare_review_row, export_candidates_to_csv


@pytest.fixture
def sample_candidates():
    """Create sample candidates."""
    return [
        CandidateFirm(
            candidate_id="cand_test_001",
            firm_name="Test Company",
            normalized_name="test company",
            raw_name_variants=["Test Company Inc", "Test Company", "TestCo Inc"],
            mention_ids=["ment_001", "ment_002"],
            source_names=["Wikipedia", "Tech News"],
            source_urls=["https://example.com/1", "https://example.com/2"],
            evidence_count=2,
        ),
        CandidateFirm(
            candidate_id="cand_test_002",
            firm_name="Another",
            normalized_name="another",
            raw_name_variants=["Another Corp"],
            mention_ids=["ment_003"],
            source_names=["LinkedIn"],
            source_urls=["https://example.com/3"],
            evidence_count=1,
        ),
    ]


@pytest.fixture
def sample_mentions():
    """Create sample mentions."""
    return [
        Mention(
            mention_id="ment_001",
            retrieved_item_id="ret_001",
            query_id="q_en_001",
            query_text="founded in Denmark headquarters",
            source_name="Wikipedia",
            title="Test Company",
            url="https://example.com/1",
            language="en",
            retrieved_at="2025-04-27T10:00:00Z",
            firm_name_raw="Test Company Inc",
            normalized_name="test company",
            evidence_text="Test Company Inc was founded in Copenhagen, Denmark and later moved to San Francisco.",
        ),
        Mention(
            mention_id="ment_002",
            retrieved_item_id="ret_002",
            query_id="q_en_002",
            query_text="Danish startup moved headquarters",
            source_name="Tech News",
            title="Test Company Moves HQ",
            url="https://example.com/2",
            language="en",
            retrieved_at="2025-04-27T11:00:00Z",
            firm_name_raw="Test Company",
            normalized_name="test company",
            evidence_text="The company relocated its headquarters from Copenhagen to San Francisco.",
        ),
        Mention(
            mention_id="ment_003",
            retrieved_item_id="ret_003",
            query_id="q_da_001",
            query_text="grundlagt i Danmark hovedkvarter",
            source_name="LinkedIn",
            title="Another Corp",
            url="https://example.com/3",
            language="da",
            retrieved_at="2025-04-27T12:00:00Z",
            firm_name_raw="Another Corp",
            normalized_name="another",
            evidence_text="Another Corp blev grundlagt i Danmark og flyttede senere til USA.",
        ),
    ]


def test_prepare_review_row(sample_candidates, sample_mentions):
    """Test review row preparation."""
    candidate = sample_candidates[0]
    mentions = sample_mentions[:2]
    
    row = prepare_review_row(candidate, mentions)
    
    assert row["candidate_id"] == "cand_test_001"
    assert row["firm_name"] == "Test Company"
    assert "Test Company Inc" in row["raw_name_variants"]
    assert row["mention_count"] == "2"
    assert row["languages"] == "en"
    assert row["unique_sources"] == "2"
    assert "Wikipedia" in row["sources"]
    assert "Tech News" in row["sources"]
    # With 2 mentions and 2 sources, should be high confidence
    assert row["confidence"] == "high"


def test_prepare_review_row_low_confidence(sample_candidates, sample_mentions):
    """Test low confidence row (single mention)."""
    candidate = sample_candidates[1]
    mentions = sample_mentions[2:3]
    
    row = prepare_review_row(candidate, mentions)
    
    assert row["mention_count"] == "1"
    assert row["confidence"] == "low"


def test_prepare_review_row_bilingual(sample_candidates, sample_mentions):
    """Test bilingual candidate detection."""
    candidate = sample_candidates[0]
    # Modify second mention to be Danish
    mention_en = sample_mentions[0]
    mention_da = Mention(
        mention_id="ment_002b",
        retrieved_item_id="ret_002b",
        query_id="q_da_002",
        query_text="dansk startup",
        source_name="Danish News",
        title="Test Company Da",
        url="https://example.com/2b",
        language="da",
        retrieved_at="2025-04-27T11:00:00Z",
        firm_name_raw="Test Company",
        normalized_name="test company",
        evidence_text="Test Company blev grundlagt i Danmark.",
    )
    mentions = [mention_en, mention_da]
    
    row = prepare_review_row(candidate, mentions)
    
    assert row["languages"] == "da/en"
    assert "bilingual" in row["confidence_signals"]


def test_export_csv_structure(tmp_path, sample_candidates, sample_mentions):
    """Test that exported CSV has correct structure."""
    output_path = tmp_path / "test_export.csv"
    
    export_candidates_to_csv(output_path, sample_candidates, sample_mentions)
    
    assert output_path.exists()
    
    # Check CSV structure
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 2
    
    # Check required columns
    required_cols = [
        "candidate_id",
        "firm_name",
        "normalized_name",
        "mention_count",
        "languages",
        "unique_sources",
        "sources",
        "verification_status",
        "reviewer_notes",
    ]
    
    for row in rows:
        for col in required_cols:
            assert col in row
            assert row[col] is not None


def test_export_csv_sorting(tmp_path, sample_candidates, sample_mentions):
    """Test that CSV is sorted by mention count."""
    output_path = tmp_path / "test_export_sort.csv"
    
    export_candidates_to_csv(output_path, sample_candidates, sample_mentions)
    
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Should be sorted by mention count (descending)
    # First candidate has 2 mentions, second has 1
    assert int(rows[0]["mention_count"]) >= int(rows[1]["mention_count"])


def test_export_csv_review_fields_empty(tmp_path, sample_candidates, sample_mentions):
    """Test that review fields are empty for fresh export."""
    output_path = tmp_path / "test_export_review.csv"
    
    export_candidates_to_csv(output_path, sample_candidates, sample_mentions)
    
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in rows:
        assert row["verification_status"] == ""
        assert row["reviewer_notes"] == ""


def test_export_csv_confidence_levels(tmp_path, sample_candidates, sample_mentions):
    """Test that confidence levels are assigned correctly."""
    output_path = tmp_path / "test_export_confidence.csv"
    
    # The sample candidates already have the correct mention_ids linked
    export_candidates_to_csv(output_path, sample_candidates, sample_mentions)
    
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # First candidate has 2 mentions from 2 sources (bilingual prep isn't there, but multi-source is)
    # So it should be high or at least medium confidence
    assert rows[0]["confidence"] in ["high", "medium"]
    # Second candidate has 1 mention only
    assert rows[1]["confidence"] == "low"


def test_export_csv_unicode(tmp_path, sample_candidates):
    """Test that CSV handles Unicode characters (Danish text)."""
    # Create candidate with Danish characters and matching mention
    candidate = CandidateFirm(
        candidate_id="cand_unicode_test",
        firm_name="Søren Løsing",
        normalized_name="søren løsing",
        raw_name_variants=["Søren Løsing A/S", "Søren Løsing"],
        mention_ids=["ment_unicode"],
        source_names=["Dansk Nyheder"],
        source_urls=["https://example.com/da"],
        evidence_count=1,
    )
    
    mention = Mention(
        mention_id="ment_unicode",
        retrieved_item_id="ret_unicode",
        query_id="q_da_001",
        query_text="Danish company",
        source_name="Dansk Nyheder",
        title="Virksomhed",
        url="https://example.com/da",
        language="da",
        retrieved_at="2025-04-27T10:00:00Z",
        firm_name_raw="Søren Løsing A/S",
        normalized_name="søren løsing",
        evidence_text="Grundlagt i København, Danmark. Flyttet hovedkvarter til USA.",
    )
    
    output_path = tmp_path / "test_unicode.csv"
    
    export_candidates_to_csv(output_path, [candidate], [mention])
    
    with open(output_path, encoding="utf-8") as f:
        content = f.read()
    
    # Check that Unicode is preserved
    assert "Søren" in content or "Soren" in content  # Should preserve or normalize Danish chars
    assert "København" in content or "Kobenhavn" in content  # Should preserve or have key evidence
