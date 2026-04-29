import csv
from pathlib import Path

import pipeline_status
from src.pipeline_rounds import RoundPaths


def test_build_status_rows_reports_only_round_specific_files(tmp_path: Path, monkeypatch) -> None:
    discovery = tmp_path / "data/discovery/snowball_round_001.jsonl"
    review = tmp_path / "data/review/snowball_round_001_review.csv"
    discovery.parent.mkdir(parents=True, exist_ok=True)
    review.parent.mkdir(parents=True, exist_ok=True)
    discovery.write_text('{"firm_name":"Issuu"}\n{"firm_name":"Zendesk"}\n', encoding="utf-8")
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["firm_name", "validation_label"])
        writer.writerow(["Issuu", "true"])

    monkeypatch.setattr(
        pipeline_status,
        "build_round_paths",
        lambda round_number: RoundPaths(
            round_number=round_number,
            discovery=discovery,
            dedup=tmp_path / "data/discovery/snowball_round_001_deduped.jsonl",
            model1=tmp_path / "data/model1/snowball_round_001_candidates.jsonl",
            model2=tmp_path / "data/model2/snowball_round_001_enriched.jsonl",
            model3=tmp_path / "data/model3/snowball_round_001_validated.jsonl",
            review=review,
            manifest=tmp_path / "data/runs/snowball_round_001_manifest.json",
        ),
    )

    rows = pipeline_status.build_status_rows(1)

    assert rows[0]["exists"] is True
    assert rows[0]["row_count"] == 2
    assert rows[-2]["exists"] is True
    assert rows[-2]["row_count"] == 1
    assert rows[0]["path"].endswith("snowball_round_001.jsonl")
    assert rows[1]["path"].endswith("snowball_round_001_deduped.jsonl")
    assert rows[-2]["path"].endswith("snowball_round_001_review.csv")
    assert rows[-1]["stage"] == "manifest"
    assert rows[-1]["row_count"] is None
