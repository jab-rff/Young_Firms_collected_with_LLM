from pathlib import Path

from recall.candidate_extraction import extract_candidates
from recall.deduplication import deduplicate_candidates
from recall.export import export_candidates
from recall.query_generation import generate_queries


def test_generate_queries_drops_empty_values() -> None:
    assert generate_queries([" ai startups ", "", " climate "]) == ["ai startups", "climate"]


def test_extract_and_deduplicate_candidates() -> None:
    extracted = extract_candidates(["Acme", "", "Beta", "Acme"])
    assert deduplicate_candidates(extracted) == ["Acme", "Beta"]


def test_export_candidates_writes_one_per_line(tmp_path: Path) -> None:
    output_path = tmp_path / "candidates.txt"
    export_candidates(["Acme", "Beta"], output_path)
    assert output_path.read_text(encoding="utf-8") == "Acme\nBeta\n"
