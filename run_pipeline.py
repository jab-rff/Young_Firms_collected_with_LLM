"""Lightweight subprocess runner for the staged round pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline_rounds import (
    STOP_AFTER_CHOICES,
    RoundPaths,
    build_round_paths,
    count_rows,
    ensure_parent_dir,
    save_json,
)

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
    max_buckets: int | None = None,
    limit: int | None = None,
    followup_discovery: bool = True,
) -> list[StageSpec]:
    paths = build_round_paths(round_number)
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
                max_buckets,
                followup_discovery,
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
    max_buckets: int | None = None,
    skip_existing: bool = False,
    stop_after: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    followup_discovery: bool = True,
) -> dict[str, Any]:
    paths = build_round_paths(round_number)
    stages = build_stage_specs(
        round_number=round_number,
        known_path=known_path,
        model_name=model_name,
        max_buckets=max_buckets,
        limit=limit,
        followup_discovery=followup_discovery,
    )
    manifest = _build_manifest(
        round_number=round_number,
        known_path=known_path,
        model_name=model_name,
        dry_run=dry_run,
        skip_existing=skip_existing,
        stop_after=stop_after,
        max_buckets=max_buckets,
        limit=limit,
        followup_discovery=followup_discovery,
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
    manifest["finished_at"] = _timestamp_now()
    save_json(manifest, paths.manifest)
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the staged snowball pipeline end-to-end.")
    parser.add_argument("--round", required=True, type=int, dest="round_number", help="Pipeline round number.")
    parser.add_argument("--known", required=True, type=Path, help="Known seed CSV path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name for LLM stages.")
    parser.add_argument("--max-buckets", type=int, default=None, help="Optional bucket cap for snowball discovery.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip stages whose expected output already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, inputs, and outputs without executing.")
    parser.add_argument("--limit", type=int, default=None, help="Optional small-run limit for supported stages.")
    parser.add_argument(
        "--no-followup-discovery",
        action="store_false",
        dest="followup_discovery",
        help="Disable the bounded second-pass 'find more' prompt in snowball discovery.",
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
    return _build_stage_result(stage=stage, status="completed", row_count=row_count)


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
    dry_run: bool,
    skip_existing: bool,
    stop_after: str | None,
    max_buckets: int | None,
    limit: int | None,
    followup_discovery: bool,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "timestamp": _timestamp_now(),
        "round_number": round_number,
        "model": model_name,
        "known_file": str(known_path),
        "dry_run": dry_run,
        "skip_existing": skip_existing,
        "stop_after": stop_after,
        "max_buckets": max_buckets,
        "limit": limit,
        "followup_discovery": followup_discovery,
        "commands_run": [],
        "output_paths": {},
        "row_counts": {},
        "skipped_stages": [],
        "stages": [],
        "success": False,
        "failure": None,
        "manifest_path": str(manifest_path),
        "finished_at": None,
    }


def _build_discovery_command(
    round_number: int,
    known_path: Path,
    paths: RoundPaths,
    model_name: str,
    max_buckets: int | None,
    followup_discovery: bool,
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
    ]
    if max_buckets is not None:
        command.extend(["--max-buckets", str(max_buckets)])
    if not followup_discovery:
        command.append("--no-followup-pass")
    return command


def _build_model_command(
    script_name: str,
    input_path: Path,
    output_path: Path,
    model_name: str,
    limit: int | None,
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


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        manifest = run_pipeline(
            round_number=args.round_number,
            known_path=args.known,
            model_name=args.model,
            max_buckets=args.max_buckets,
            skip_existing=args.skip_existing,
            stop_after=args.stop_after,
            dry_run=args.dry_run,
            limit=args.limit,
            followup_discovery=args.followup_discovery,
        )
        if not args.dry_run:
            print(f"manifest_path={manifest['manifest_path']}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
