"""Lightweight subprocess runner for the staged round pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline_rounds import (
    ORIGIN_TRACK_CHOICES,
    STOP_AFTER_CHOICES,
    RoundPaths,
    build_round_paths,
    count_rows,
    ensure_parent_dir,
    save_json,
)
from src.openai_costs import cost_log_path, sum_cost_records

DEFAULT_MODEL = "gpt-5-mini"
TOTAL_STAGES = 6


@dataclass(frozen=True)
class StageSpec:
    slug: str
    label: str
    script_name: str
    input_path: Path | None
    output_path: Path
    command: list[str]
    allow_empty_output: bool = True


def build_stage_specs(
    round_number: int,
    known_path: Path,
    model_name: str,
    origin_track: str = "in_denmark",
    max_buckets: int | None = None,
    limit: int | None = None,
    followup_discovery_rounds: int = 3,
    batch_llm_stages: bool = False,
    batch_discovery: bool = False,
    skip_processed_buckets: bool = False,
) -> list[StageSpec]:
    paths = build_round_paths(round_number, origin_track=origin_track)
    return [
        StageSpec(
            slug="discovery",
            label="Snowball discovery",
            script_name="snowball_discovery.py",
            input_path=known_path,
            output_path=paths.discovery,
            command=_build_discovery_command(
                round_number,
                known_path,
                paths,
                model_name,
                origin_track,
                max_buckets,
                followup_discovery_rounds,
                batch_discovery,
                skip_processed_buckets,
            ),
        ),
        StageSpec(
            slug="dedup",
            label="Deduplicate",
            script_name="deduplicate_snowball_candidates.py",
            input_path=paths.discovery,
            output_path=paths.dedup,
            command=[
                sys.executable,
                "deduplicate_snowball_candidates.py",
                "--input",
                str(paths.discovery),
                "--known",
                str(known_path),
                "--output",
                str(paths.dedup),
            ],
        ),
        StageSpec(
            slug="model1",
            label="Model 1 candidate extraction",
            script_name="model_1_candidate_extraction.py",
            input_path=paths.dedup,
            output_path=paths.model1,
            command=_build_model_command(
                script_name="model_1_candidate_extraction.py",
                input_path=paths.dedup,
                output_path=paths.model1,
                model_name=model_name,
                limit=limit,
                batch=batch_llm_stages,
            ),
        ),
        StageSpec(
            slug="model2",
            label="Model 2 enrichment",
            script_name="model_2_enrichment.py",
            input_path=paths.model1,
            output_path=paths.model2,
            command=_build_model_command(
                script_name="model_2_enrichment.py",
                input_path=paths.model1,
                output_path=paths.model2,
                model_name=model_name,
                limit=limit,
                batch=batch_llm_stages,
            ),
        ),
        StageSpec(
            slug="model3",
            label="Model 3 validation",
            script_name="model_3_validation.py",
            input_path=paths.model2,
            output_path=paths.model3,
            command=_build_model_command(
                script_name="model_3_validation.py",
                input_path=paths.model2,
                output_path=paths.model3,
                model_name=model_name,
                limit=limit,
                batch=batch_llm_stages,
            ),
        ),
        StageSpec(
            slug="export",
            label="Export final review",
            script_name="export_final_review.py",
            input_path=paths.model3,
            output_path=paths.review,
            command=[
                sys.executable,
                "export_final_review.py",
                "--input",
                str(paths.model3),
                "--output",
                str(paths.review),
            ],
        ),
    ]


def run_pipeline(
    round_number: int,
    known_path: Path,
    model_name: str,
    origin_track: str = "in_denmark",
    max_buckets: int | None = None,
    skip_existing: bool = False,
    stop_after: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    followup_discovery_rounds: int = 3,
    batch_llm_stages: bool = False,
    batch_discovery: bool = False,
    skip_processed_buckets: bool = False,
) -> dict[str, Any]:
    paths = build_round_paths(round_number, origin_track=origin_track)
    stage_spec_kwargs = {
        "round_number": round_number,
        "known_path": known_path,
        "model_name": model_name,
        "origin_track": origin_track,
        "max_buckets": max_buckets,
        "limit": limit,
        "followup_discovery_rounds": followup_discovery_rounds,
        "batch_llm_stages": batch_llm_stages,
        "batch_discovery": batch_discovery,
    }
    if skip_processed_buckets:
        stage_spec_kwargs["skip_processed_buckets"] = True
    stages = build_stage_specs(**stage_spec_kwargs)
    manifest = _build_manifest(
        round_number=round_number,
        known_path=known_path,
        model_name=model_name,
        origin_track=origin_track,
        dry_run=dry_run,
        skip_existing=skip_existing,
        stop_after=stop_after,
        max_buckets=max_buckets,
        limit=limit,
        followup_discovery_rounds=followup_discovery_rounds,
        batch_llm_stages=batch_llm_stages,
        batch_discovery=batch_discovery,
        skip_processed_buckets=skip_processed_buckets,
        manifest_path=paths.manifest,
    )

    try:
        for index, stage in enumerate(stages, start=1):
            print(f"[{index}/{TOTAL_STAGES}] {stage.label}")
            stage_result = _run_stage(
                stage=stage,
                dry_run=dry_run,
                skip_existing=skip_existing,
            )
            manifest["stages"].append(stage_result)
            manifest["commands_run"].append(" ".join(stage.command))
            manifest["output_paths"][stage.slug] = str(stage.output_path)
            if stage_result["row_count"] is not None:
                manifest["row_counts"][stage.slug] = stage_result["row_count"]
            if stage_result.get("estimated_cost_usd") is not None:
                manifest["costs_usd"][stage.slug] = stage_result["estimated_cost_usd"]
            if stage_result["status"] == "skipped":
                manifest["skipped_stages"].append(stage.slug)
            if stage.slug == stop_after:
                break
    except Exception as exc:
        manifest["success"] = False
        manifest["failure"] = str(exc)
        manifest["finished_at"] = _timestamp_now()
        save_json(manifest, paths.manifest)
        raise

    manifest["success"] = True
    manifest["failure"] = None
    manifest["estimated_cost_usd_total"] = round(sum(float(value) for value in manifest["costs_usd"].values()), 8)
    manifest["finished_at"] = _timestamp_now()
    save_json(manifest, paths.manifest)
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the staged snowball pipeline end-to-end.")
    parser.add_argument("--round", required=True, type=int, dest="round_number", help="Pipeline round number.")
    parser.add_argument("--known", required=True, type=Path, help="Known seed CSV path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name for LLM stages.")
    parser.add_argument(
        "--origin-track",
        choices=ORIGIN_TRACK_CHOICES,
        default="in_denmark",
        help="Origin track to run through the pipeline.",
    )
    parser.add_argument("--max-buckets", type=int, default=None, help="Optional bucket cap for snowball discovery.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip stages whose expected output already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, inputs, and outputs without executing.")
    parser.add_argument("--limit", type=int, default=None, help="Optional small-run limit for supported stages.")
    parser.add_argument(
        "--batch-llm-stages",
        action="store_true",
        help="Submit Model 1, Model 2, and Model 3 requests through the OpenAI Batch API.",
    )
    parser.add_argument(
        "--batch-discovery",
        action="store_true",
        help="Submit discovery requests through the OpenAI Batch API, batching each pass across buckets.",
    )
    parser.add_argument(
        "--batch-all-api",
        action="store_true",
        help="Submit discovery and Model 1-3 requests through the OpenAI Batch API.",
    )
    parser.add_argument(
        "--skip-processed-buckets",
        action="store_true",
        help="Skip discovery buckets already present in prior *_bucket_runs.jsonl files for this origin track.",
    )
    parser.add_argument(
        "--followup-discovery-rounds",
        type=int,
        default=3,
        help="Number of follow-up discovery rounds to run per bucket after the initial pass.",
    )
    parser.add_argument(
        "--no-followup-discovery",
        action="store_const",
        const=0,
        dest="followup_discovery_rounds",
        help="Disable follow-up discovery passes. Deprecated alias for --followup-discovery-rounds 0.",
    )
    parser.add_argument(
        "--stop-after",
        choices=STOP_AFTER_CHOICES,
        default=None,
        help="Stop after the named stage.",
    )
    return parser


def _run_stage(stage: StageSpec, dry_run: bool, skip_existing: bool) -> dict[str, Any]:
    _validate_stage_input(stage)
    ensure_parent_dir(stage.output_path)

    if dry_run:
        _print_dry_run(stage)
        return _build_stage_result(stage=stage, status="dry_run", row_count=None)

    if skip_existing and stage.output_path.exists():
        row_count = count_rows(stage.output_path)
        print("SKIPPED existing output:")
        print(stage.output_path)
        print(f"Rows: {row_count}")
        return _build_stage_result(stage=stage, status="skipped", row_count=row_count)

    _run_stage_command(stage)
    row_count = _validate_stage_output(stage)
    print(f"output_path={stage.output_path}")
    print(f"row_count={row_count}")
    stage_result = _build_stage_result(stage=stage, status="completed", row_count=row_count)
    stage_cost_log_path = cost_log_path(stage.output_path)
    if stage_cost_log_path.exists():
        cost_records = _load_jsonl(stage_cost_log_path)
        cost_totals = sum_cost_records(cost_records)
        stage_result["estimated_cost_usd"] = cost_totals["estimated_cost_usd"]
        stage_result["api_cost_log_path"] = str(stage_cost_log_path)
        print(f"estimated_cost_usd={cost_totals['estimated_cost_usd']:.6f}")
        print(f"api_cost_log_path={stage_cost_log_path}")
    return stage_result


def _build_stage_result(stage: StageSpec, status: str, row_count: int | None) -> dict[str, Any]:
    return {
        "stage": stage.slug,
        "label": stage.label,
        "status": status,
        "command": " ".join(stage.command),
        "input_path": str(stage.input_path) if stage.input_path is not None else None,
        "output_path": str(stage.output_path),
        "row_count": row_count,
    }


def _build_manifest(
    round_number: int,
    known_path: Path,
    model_name: str,
    origin_track: str,
    dry_run: bool,
    skip_existing: bool,
    stop_after: str | None,
    max_buckets: int | None,
    limit: int | None,
    followup_discovery_rounds: int,
    batch_llm_stages: bool,
    batch_discovery: bool,
    skip_processed_buckets: bool,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "timestamp": _timestamp_now(),
        "round_number": round_number,
        "model": model_name,
        "origin_track": origin_track,
        "known_file": str(known_path),
        "dry_run": dry_run,
        "skip_existing": skip_existing,
        "stop_after": stop_after,
        "max_buckets": max_buckets,
        "limit": limit,
        "followup_discovery_rounds": followup_discovery_rounds,
        "batch_llm_stages": batch_llm_stages,
        "batch_discovery": batch_discovery,
        "skip_processed_buckets": skip_processed_buckets,
        "commands_run": [],
        "output_paths": {},
        "row_counts": {},
        "costs_usd": {},
        "skipped_stages": [],
        "stages": [],
        "success": False,
        "failure": None,
        "manifest_path": str(manifest_path),
        "estimated_cost_usd_total": 0.0,
        "finished_at": None,
    }


def _build_discovery_command(
    round_number: int,
    known_path: Path,
    paths: RoundPaths,
    model_name: str,
    origin_track: str,
    max_buckets: int | None,
    followup_discovery_rounds: int,
    batch_discovery: bool,
    skip_processed_buckets: bool,
) -> list[str]:
    command = [
        sys.executable,
        "snowball_discovery.py",
        "--known",
        str(known_path),
        "--output",
        str(paths.discovery),
        "--round",
        str(round_number),
        "--model",
        model_name,
        "--origin-track",
        origin_track,
    ]
    if max_buckets is not None:
        command.extend(["--max-buckets", str(max_buckets)])
    command.extend(["--followup-rounds", str(max(0, followup_discovery_rounds))])
    if batch_discovery:
        command.append("--batch")
    if skip_processed_buckets:
        command.append("--skip-processed-buckets")
    return command


def _build_model_command(
    script_name: str,
    input_path: Path,
    output_path: Path,
    model_name: str,
    limit: int | None,
    batch: bool,
) -> list[str]:
    command = [
        sys.executable,
        script_name,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        model_name,
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if batch:
        command.append("--batch")
    return command


def _print_dry_run(stage: StageSpec) -> None:
    print("DRY RUN")
    print(f"Command: {' '.join(stage.command)}")
    print(f"Expected input: {stage.input_path if stage.input_path is not None else '(none)'}")
    print(f"Expected output: {stage.output_path}")


def _validate_stage_input(stage: StageSpec) -> None:
    if stage.input_path is not None and not stage.input_path.exists():
        raise FileNotFoundError(f"Missing input for stage '{stage.slug}': {stage.input_path}")


def _run_stage_command(stage: StageSpec) -> None:
    try:
        subprocess.run(stage.command, check=True)
    except subprocess.CalledProcessError as exc:
        command_text = subprocess.list2cmdline(stage.command)
        raise RuntimeError(f"Stage '{stage.slug}' failed: {command_text}") from exc


def _validate_stage_output(stage: StageSpec) -> int:
    if not stage.output_path.exists():
        raise RuntimeError(f"Stage '{stage.slug}' did not create output: {stage.output_path}")
    row_count = count_rows(stage.output_path)
    if row_count <= 0 and not stage.allow_empty_output:
        raise RuntimeError(f"Stage '{stage.slug}' created empty output: {stage.output_path}")
    if row_count <= 0:
        print(f"WARNING: zero-row output for stage '{stage.slug}'")
    return row_count


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    batch_llm_stages = args.batch_llm_stages or args.batch_all_api
    batch_discovery = args.batch_discovery or args.batch_all_api
    try:
        manifest = run_pipeline(
            round_number=args.round_number,
            known_path=args.known,
            model_name=args.model,
            origin_track=args.origin_track,
            max_buckets=args.max_buckets,
            skip_existing=args.skip_existing,
            stop_after=args.stop_after,
            dry_run=args.dry_run,
            limit=args.limit,
            followup_discovery_rounds=max(0, args.followup_discovery_rounds),
            batch_llm_stages=batch_llm_stages,
            batch_discovery=batch_discovery,
            skip_processed_buckets=args.skip_processed_buckets,
        )
        if not args.dry_run:
            print(f"manifest_path={manifest['manifest_path']}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
