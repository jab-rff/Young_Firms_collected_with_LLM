import subprocess
from json import loads
from pathlib import Path

import pytest

from run_pipeline import StageSpec, build_stage_specs, run_pipeline


def test_build_stage_specs_includes_expected_paths_and_max_buckets() -> None:
    stages = build_stage_specs(
        round_number=1,
        known_path=Path("preliminary_data_28_04.csv"),
        model_name="gpt-5-mini",
        max_buckets=2,
        limit=20,
    )

    assert len(stages) == 6
    assert stages[0].output_path == Path("data/discovery/snowball_round_001.jsonl")
    assert "--max-buckets" in stages[0].command
    assert "--no-followup-pass" not in stages[0].command
    assert "--limit" in stages[2].command
    assert stages[-1].output_path == Path("data/review/snowball_round_001_review.csv")


def test_build_stage_specs_can_disable_followup_discovery() -> None:
    stages = build_stage_specs(
        round_number=1,
        known_path=Path("preliminary_data_28_04.csv"),
        model_name="gpt-5-mini",
        followup_discovery=False,
    )

    assert "--no-followup-pass" in stages[0].command


def test_run_pipeline_skips_existing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    known_path = tmp_path / "known.csv"
    known_path.write_text("name,founding_origin,industry\nIssuu,in Denmark,software\n", encoding="utf-8")

    discovery_output = tmp_path / "data/discovery/snowball_round_001.jsonl"
    discovery_output.parent.mkdir(parents=True, exist_ok=True)
    discovery_output.write_text('{"firm_name":"Issuu"}\n', encoding="utf-8")

    def fake_build_stage_specs(
        round_number: int,
        known_path: Path,
        model_name: str,
        max_buckets=None,
        limit=None,
        followup_discovery=True,
    ):
        return [
            StageSpec(
                slug="discovery",
                label="Snowball discovery",
                script_name="snowball_discovery.py",
                input_path=known_path,
                output_path=discovery_output,
                command=["python", "snowball_discovery.py"],
            )
        ]

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when skipping existing output")

    monkeypatch.setattr("run_pipeline.build_stage_specs", fake_build_stage_specs)
    monkeypatch.setattr(subprocess, "run", fail_run)

    outputs = run_pipeline(
        round_number=1,
        known_path=known_path,
        model_name="gpt-5-mini",
        skip_existing=True,
        stop_after="discovery",
    )

    assert outputs["skipped_stages"] == ["discovery"]
    assert outputs["success"] is True


def test_run_pipeline_dry_run_creates_manifest_without_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    known_path = tmp_path / "known.csv"
    known_path.write_text("name,founding_origin,industry\nIssuu,in Denmark,software\n", encoding="utf-8")
    output_path = tmp_path / "data/discovery/output.jsonl"
    manifest_path = tmp_path / "data/runs/manifest.json"

    def fake_build_stage_specs(
        round_number: int,
        known_path: Path,
        model_name: str,
        max_buckets=None,
        limit=None,
        followup_discovery=True,
    ):
        return [
            StageSpec(
                slug="discovery",
                label="Snowball discovery",
                script_name="snowball_discovery.py",
                input_path=known_path,
                output_path=output_path,
                command=["python", "snowball_discovery.py", "--round", "1"],
            )
        ]

    monkeypatch.setattr("run_pipeline.build_stage_specs", fake_build_stage_specs)
    monkeypatch.setattr(
        "run_pipeline.build_round_paths",
        lambda round_number: __import__("src.pipeline_rounds", fromlist=["RoundPaths"]).RoundPaths(
            round_number=round_number,
            discovery=tmp_path / "data/discovery/snowball_round_001.jsonl",
            dedup=tmp_path / "data/discovery/snowball_round_001_deduped.jsonl",
            model1=tmp_path / "data/model1/snowball_round_001_candidates.jsonl",
            model2=tmp_path / "data/model2/snowball_round_001_enriched.jsonl",
            model3=tmp_path / "data/model3/snowball_round_001_validated.jsonl",
            review=tmp_path / "data/review/snowball_round_001_review.csv",
            manifest=manifest_path,
        ),
    )

    manifest = run_pipeline(
        round_number=1,
        known_path=known_path,
        model_name="gpt-5-mini",
        dry_run=True,
        stop_after="discovery",
    )

    assert manifest["success"] is True
    assert manifest_path.exists()
    saved = loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["dry_run"] is True
    assert saved["stages"][0]["status"] == "dry_run"


def test_run_pipeline_raises_clear_error_for_missing_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    known_path = tmp_path / "known.csv"
    known_path.write_text("name,founding_origin,industry\nIssuu,in Denmark,software\n", encoding="utf-8")
    missing_input = tmp_path / "missing.jsonl"
    output_path = tmp_path / "data/model1/output.jsonl"

    def fake_build_stage_specs(
        round_number: int,
        known_path: Path,
        model_name: str,
        max_buckets=None,
        limit=None,
        followup_discovery=True,
    ):
        return [
            StageSpec(
                slug="model1",
                label="Model 1 candidate extraction",
                script_name="model_1_candidate_extraction.py",
                input_path=missing_input,
                output_path=output_path,
                command=["python", "model_1_candidate_extraction.py"],
            )
        ]

    monkeypatch.setattr("run_pipeline.build_stage_specs", fake_build_stage_specs)

    with pytest.raises(FileNotFoundError, match="Missing input for stage 'model1'"):
        run_pipeline(round_number=1, known_path=known_path, model_name="gpt-5-mini")


def test_run_pipeline_raises_clear_error_for_failed_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    known_path = tmp_path / "known.csv"
    known_path.write_text("name,founding_origin,industry\nIssuu,in Denmark,software\n", encoding="utf-8")
    output_path = tmp_path / "data/discovery/output.jsonl"

    def fake_build_stage_specs(
        round_number: int,
        known_path: Path,
        model_name: str,
        max_buckets=None,
        limit=None,
        followup_discovery=True,
    ):
        return [
            StageSpec(
                slug="discovery",
                label="Snowball discovery",
                script_name="snowball_discovery.py",
                input_path=known_path,
                output_path=output_path,
                command=["python", "snowball_discovery.py", "--round", "1"],
            )
        ]

    def fake_run(command, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr("run_pipeline.build_stage_specs", fake_build_stage_specs)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Stage 'discovery' failed:"):
        run_pipeline(round_number=1, known_path=known_path, model_name="gpt-5-mini")


def test_run_pipeline_writes_manifest_for_completed_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    known_path = tmp_path / "known.csv"
    known_path.write_text("name,founding_origin,industry\nIssuu,in Denmark,software\n", encoding="utf-8")
    output_path = tmp_path / "data/discovery/output.jsonl"
    manifest_path = tmp_path / "data/runs/manifest.json"

    def fake_build_stage_specs(
        round_number: int,
        known_path: Path,
        model_name: str,
        max_buckets=None,
        limit=None,
        followup_discovery=True,
    ):
        return [
            StageSpec(
                slug="discovery",
                label="Snowball discovery",
                script_name="snowball_discovery.py",
                input_path=known_path,
                output_path=output_path,
                command=["python", "snowball_discovery.py", "--round", "1"],
            )
        ]

    def fake_run(command, check):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"firm_name":"Issuu"}\n', encoding="utf-8")

    monkeypatch.setattr("run_pipeline.build_stage_specs", fake_build_stage_specs)
    monkeypatch.setattr(
        "run_pipeline.build_round_paths",
        lambda round_number: __import__("src.pipeline_rounds", fromlist=["RoundPaths"]).RoundPaths(
            round_number=round_number,
            discovery=tmp_path / "data/discovery/snowball_round_001.jsonl",
            dedup=tmp_path / "data/discovery/snowball_round_001_deduped.jsonl",
            model1=tmp_path / "data/model1/snowball_round_001_candidates.jsonl",
            model2=tmp_path / "data/model2/snowball_round_001_enriched.jsonl",
            model3=tmp_path / "data/model3/snowball_round_001_validated.jsonl",
            review=tmp_path / "data/review/snowball_round_001_review.csv",
            manifest=manifest_path,
        ),
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    manifest = run_pipeline(
        round_number=1,
        known_path=known_path,
        model_name="gpt-5-mini",
        stop_after="discovery",
    )

    assert manifest["success"] is True
    saved = loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["row_counts"]["discovery"] == 1
    assert saved["stages"][0]["status"] == "completed"
