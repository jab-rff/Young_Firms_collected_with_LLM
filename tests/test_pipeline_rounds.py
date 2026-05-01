import csv
from pathlib import Path

from src.pipeline_rounds import build_round_paths, count_rows


def test_build_round_paths_uses_round_specific_output_paths() -> None:
    paths = build_round_paths(1)

    assert paths.origin_track == "in_denmark"
    assert paths.discovery == Path("data/discovery/snowball_round_001.jsonl")
    assert paths.dedup == Path("data/discovery/snowball_round_001_deduped.jsonl")
    assert paths.model1 == Path("data/model1/snowball_round_001_candidates.jsonl")
    assert paths.model2 == Path("data/model2/snowball_round_001_enriched.jsonl")
    assert paths.model3 == Path("data/model3/snowball_round_001_validated.jsonl")
    assert paths.review == Path("data/review/snowball_round_001_review.csv")
    assert paths.manifest == Path("data/runs/snowball_round_001_manifest.json")


def test_build_round_paths_uses_track_suffix_for_founder_abroad_rounds() -> None:
    paths = build_round_paths(2, origin_track="abroad_danish_founders")

    assert paths.origin_track == "abroad_danish_founders"
    assert paths.discovery == Path("data/discovery/snowball_round_002_abroad_danish_founders.jsonl")
    assert paths.review == Path("data/review/snowball_round_002_abroad_danish_founders_review.csv")


def test_count_rows_counts_non_empty_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    assert count_rows(path) == 2


def test_count_rows_counts_csv_rows_without_header(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["firm_name", "validation_label"])
        writer.writerow(["Issuu", "true"])
        writer.writerow(["Zendesk", "true"])

    assert count_rows(path) == 2
