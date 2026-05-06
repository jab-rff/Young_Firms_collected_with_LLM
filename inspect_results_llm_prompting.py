"""Audit `results_llm_prompting.xlsx` against local pipeline outputs.

This mirrors the old "inspect third-party list" workflow:
1. Start from an external/manual list of firms.
2. Normalize firm names and extract reasonable aliases.
3. Check whether each firm appears across staged pipeline outputs.
4. Summarize where a firm was surfaced, filtered out, or already present.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.normalization import normalize_company_name

DEFAULT_INPUT = Path("results_llm_prompting.xlsx")
DEFAULT_SHEET = "close_reading_cases"
DEFAULT_OUTPUT_CSV = Path("analysis/results_llm_prompting_audit.csv")
DEFAULT_OUTPUT_MD = Path("analysis/results_llm_prompting_audit.md")
DEFAULT_FINAL_REVIEW = Path("data/manual_review/final_dataset.csv")
DEFAULT_MASTER_REVIEW = Path("data/cumulative/final_review_master_all_tracks.csv")

HUMAN_TRUE_LABELS = {"true"}
HUMAN_FALSE_LABELS = {"false"}
HUMAN_ALREADY_FOUND_LABELS = {"found w børsen", "found w borsen"}
HUMAN_DUPLICATE_LABELS = {"duplicate"}
HUMAN_UNCLEAR_LABELS = {"unclear"}

STAGE_SPECS = [
    ("discovery", "data/discovery/snowball_round_*.jsonl"),
    ("dedup", "data/discovery/snowball_round_*_deduped.jsonl"),
    ("model1", "data/model1/snowball_round_*_candidates.jsonl"),
    ("model2", "data/model2/snowball_round_*_enriched.jsonl"),
    ("model3", "data/model3/snowball_round_*_validated.jsonl"),
]

_PARENS_RE = re.compile(r"\(([^()]*)\)")
_LEGAL_SUFFIX_TOKENS = {
    "a/s",
    "aps",
    "ab",
    "as",
    "sa",
    "s.a.",
    "inc",
    "inc.",
    "llc",
    "ltd",
    "ltd.",
    "limited",
    "holding",
    "holdings",
}


@dataclass(frozen=True)
class MatchRecord:
    stage: str
    path: str
    matched_name: str
    validation_label: str = ""
    origin_track: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect results_llm_prompting firms across local pipeline stages.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input workbook with manual close-reading results.")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="Workbook sheet name.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Audit CSV output path.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_OUTPUT_MD, help="Markdown summary output path.")
    parser.add_argument(
        "--final-review",
        type=Path,
        default=DEFAULT_FINAL_REVIEW,
        help="Manual final review CSV path used for local matches.",
    )
    parser.add_argument(
        "--master-review",
        type=Path,
        default=DEFAULT_MASTER_REVIEW,
        help="Cumulative review CSV path used for local matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workbook_df = pd.read_excel(args.input, sheet_name=args.sheet)
    stage_indexes = build_stage_indexes()
    final_review_index = build_csv_index(
        args.final_review,
        stage="final_review",
        name_column="firm",
        extra_columns=("human_validation",),
    )
    master_review_index = build_csv_index(
        args.master_review,
        stage="master_review",
        name_column="firm_name",
        extra_columns=("validation_label", "origin_track"),
    )

    audit_rows = []
    for row in workbook_df.to_dict(orient="records"):
        audit_rows.append(
            audit_row(
                row=row,
                stage_indexes=stage_indexes,
                final_review_index=final_review_index,
                master_review_index=master_review_index,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(audit_rows, args.output)
    write_summary(audit_rows, args.summary_output, csv_output_path=args.output)

    print(f"audit_rows={len(audit_rows)}")
    print(f"output_path={args.output}")
    print(f"summary_output_path={args.summary_output}")


def audit_row(
    row: dict[str, Any],
    stage_indexes: dict[str, dict[str, MatchRecord]],
    final_review_index: dict[str, MatchRecord],
    master_review_index: dict[str, MatchRecord],
) -> dict[str, Any]:
    firm = safe_text(row.get("firm"))
    aliases = extract_aliases(firm)
    stage_matches = {stage: first_match(index, aliases) for stage, index in stage_indexes.items()}
    final_review_match = first_match(final_review_index, aliases)
    master_review_match = first_match(master_review_index, aliases)

    human_validation = normalize_label(row.get("human_validation"))
    first_stage_seen = next((stage for stage, match in stage_matches.items() if match is not None), "")
    if not first_stage_seen and final_review_match:
        first_stage_seen = "final_review"
    if not first_stage_seen and master_review_match:
        first_stage_seen = "master_review"

    status_bucket = classify_status_bucket(human_validation)
    pipeline_outcome = classify_pipeline_outcome(
        human_validation=human_validation,
        stage_matches=stage_matches,
        final_review_match=final_review_match,
        master_review_match=master_review_match,
    )
    likely_reason = infer_likely_reason(
        human_validation=human_validation,
        stage_matches=stage_matches,
        final_review_match=final_review_match,
        master_review_match=master_review_match,
        workbook_row=row,
    )

    return {
        "firm": firm,
        "aliases_checked": " | ".join(sorted(aliases)),
        "human_validation": safe_text(row.get("human_validation")),
        "founded_dk_manual": safe_text(row.get("founded_dk")),
        "relocation_manual": safe_text(row.get("relocation")),
        "comment_final": safe_text(row.get("comment_final")),
        "additional_comment": safe_text(row.get("additional_comment")),
        "status_bucket": status_bucket,
        "first_stage_seen": first_stage_seen,
        "in_discovery": bool_text(stage_matches["discovery"] is not None),
        "in_dedup": bool_text(stage_matches["dedup"] is not None),
        "in_model1": bool_text(stage_matches["model1"] is not None),
        "in_model2": bool_text(stage_matches["model2"] is not None),
        "in_model3": bool_text(stage_matches["model3"] is not None),
        "in_final_review": bool_text(final_review_match is not None),
        "in_master_review": bool_text(master_review_match is not None),
        "pipeline_model3_label": stage_matches["model3"].validation_label if stage_matches["model3"] else "",
        "master_review_label": master_review_match.validation_label if master_review_match else "",
        "master_review_origin_track": master_review_match.origin_track if master_review_match else "",
        "pipeline_outcome": pipeline_outcome,
        "likely_reason": likely_reason,
        "discovery_match_name": matched_name(stage_matches["discovery"]),
        "dedup_match_name": matched_name(stage_matches["dedup"]),
        "model1_match_name": matched_name(stage_matches["model1"]),
        "model2_match_name": matched_name(stage_matches["model2"]),
        "model3_match_name": matched_name(stage_matches["model3"]),
        "final_review_match_name": matched_name(final_review_match),
        "master_review_match_name": matched_name(master_review_match),
        "discovery_path": matched_path(stage_matches["discovery"]),
        "dedup_path": matched_path(stage_matches["dedup"]),
        "model1_path": matched_path(stage_matches["model1"]),
        "model2_path": matched_path(stage_matches["model2"]),
        "model3_path": matched_path(stage_matches["model3"]),
    }


def build_stage_indexes() -> dict[str, dict[str, MatchRecord]]:
    indexes: dict[str, dict[str, MatchRecord]] = {}
    for stage, pattern in STAGE_SPECS:
        index: dict[str, MatchRecord] = {}
        for path in sorted(Path().glob(pattern)):
            if should_skip_stage_path(path):
                continue
            load_jsonl_index(path=path, stage=stage, index=index)
        indexes[stage] = index
    return indexes


def should_skip_stage_path(path: Path) -> bool:
    name = path.name
    return name.endswith("_api_costs.jsonl") or name.endswith("_bucket_runs.jsonl")


def load_jsonl_index(path: Path, stage: str, index: dict[str, MatchRecord]) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name = safe_text(record.get("firm_name"))
            validation_label = safe_text(record.get("validation_label"))
            origin_track = safe_text(record.get("origin_track"))
            for alias in extract_aliases(name):
                index.setdefault(
                    alias,
                    MatchRecord(
                        stage=stage,
                        path=str(path),
                        matched_name=name,
                        validation_label=validation_label,
                        origin_track=origin_track,
                    ),
                )


def build_csv_index(
    path: Path,
    stage: str,
    name_column: str,
    extra_columns: tuple[str, ...] = (),
) -> dict[str, MatchRecord]:
    index: dict[str, MatchRecord] = {}
    if not path.exists():
        return index
    df = pd.read_csv(path)
    for row in df.to_dict(orient="records"):
        name = safe_text(row.get(name_column))
        if not name:
            continue
        extras = {column: safe_text(row.get(column)) for column in extra_columns}
        for alias in extract_aliases(name):
            index.setdefault(
                alias,
                MatchRecord(
                    stage=stage,
                    path=str(path),
                    matched_name=name,
                    validation_label=extras.get("validation_label", extras.get("human_validation", "")),
                    origin_track=extras.get("origin_track", ""),
                ),
            )
    return index


def extract_aliases(name: str) -> set[str]:
    raw = safe_text(name).strip()
    if not raw:
        return set()

    candidates = {raw}
    candidates.update(split_alias_chunks(raw))
    for paren_text in _PARENS_RE.findall(raw):
        candidates.add(paren_text.strip())
        candidates.update(split_alias_chunks(paren_text))

    aliases: set[str] = set()
    for candidate in candidates:
        cleaned = re.sub(r"\bformerly\b", "", safe_text(candidate), flags=re.IGNORECASE).strip(" -,:")
        for alias in build_alias_variants(cleaned):
            if is_reasonable_alias(alias):
                aliases.add(alias)
    return aliases


def split_alias_chunks(text: str) -> list[str]:
    chunks = [safe_text(text)]
    separators = [" → ", " -> ", " / ", ";", " | "]
    for separator in separators:
        next_chunks: list[str] = []
        for chunk in chunks:
            next_chunks.extend(chunk.split(separator))
        chunks = next_chunks
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def build_alias_variants(text: str) -> set[str]:
    normalized = normalize_company_name(text)
    variants = {normalized} if normalized else set()

    cleaned_tokens = [token for token in re.split(r"\s+", normalized) if token]
    while cleaned_tokens and cleaned_tokens[-1] in _LEGAL_SUFFIX_TOKENS:
        cleaned_tokens.pop()
    if cleaned_tokens:
        variants.add(" ".join(cleaned_tokens))
    return variants


def is_reasonable_alias(value: str) -> bool:
    if not value:
        return False
    if len(value) < 4:
        return False
    if value in {"group", "holding", "holdings", "international", "company", "companies"}:
        return False
    return any(char.isalpha() for char in value)


def first_match(index: dict[str, MatchRecord], aliases: Iterable[str]) -> MatchRecord | None:
    for alias in sorted(set(aliases), key=lambda value: (-len(value), value)):
        match = index.get(alias)
        if match is not None:
            return match
    return None


def classify_status_bucket(human_validation: str) -> str:
    if human_validation in HUMAN_TRUE_LABELS:
        return "manual_true"
    if human_validation in HUMAN_FALSE_LABELS:
        return "manual_false"
    if human_validation in HUMAN_ALREADY_FOUND_LABELS:
        return "already_found_borsen"
    if human_validation in HUMAN_DUPLICATE_LABELS:
        return "duplicate"
    if human_validation in HUMAN_UNCLEAR_LABELS:
        return "manual_unclear"
    return "other"


def classify_pipeline_outcome(
    human_validation: str,
    stage_matches: dict[str, MatchRecord | None],
    final_review_match: MatchRecord | None,
    master_review_match: MatchRecord | None,
) -> str:
    if human_validation in HUMAN_ALREADY_FOUND_LABELS:
        return "already_found_by_original_method"
    if human_validation in HUMAN_DUPLICATE_LABELS:
        return "duplicate_variant"
    if human_validation in HUMAN_UNCLEAR_LABELS:
        return "manual_outcome_unclear"
    if human_validation in HUMAN_FALSE_LABELS:
        if master_review_match:
            return "pipeline_surfaced_but_manual_close_reading_rejected"
        return "manual_close_reading_rejected"
    if human_validation in HUMAN_TRUE_LABELS:
        if master_review_match and normalize_label(master_review_match.validation_label) == "true":
            return "pipeline_captured_true_case"
        if stage_matches["model3"] or final_review_match or master_review_match:
            return "pipeline_surfaced_but_final_label_differs"
        if any(stage_matches.values()):
            return "seen_early_but_lost_before_final_review"
        return "never_seen_in_local_pipeline_outputs"
    return "unclassified"


def infer_likely_reason(
    human_validation: str,
    stage_matches: dict[str, MatchRecord | None],
    final_review_match: MatchRecord | None,
    master_review_match: MatchRecord | None,
    workbook_row: dict[str, Any],
) -> str:
    if human_validation in HUMAN_ALREADY_FOUND_LABELS:
        return "The workbook already marks this firm as found by the original Børsen-based workflow."
    if human_validation in HUMAN_DUPLICATE_LABELS:
        return "This row is a naming duplicate or alias of another case rather than a distinct miss."
    if human_validation in HUMAN_UNCLEAR_LABELS:
        return "Manual close reading remained ambiguous, so this is not a clean miss to explain."
    if human_validation in HUMAN_FALSE_LABELS:
        comment = combined_comment(workbook_row)
        if "acquisition" in comment or "acquired" in comment or "integrated" in comment:
            return "Manual close reading indicates acquisition/legal-parent change rather than an independent HQ relocation."
        if "two headquarters" in comment or "unclear" in comment or "london" in comment:
            return "Manual close reading indicates ambiguous or split HQ evidence rather than a confirmed relocation."
        if "office" in comment or "expand" in comment or "scale" in comment:
            return "Manual close reading indicates foreign-office expansion, not a headquarters move."
        return "Manual close reading rejected this as a target case under the research definition."
    if human_validation not in HUMAN_TRUE_LABELS:
        return ""

    if master_review_match and normalize_label(master_review_match.validation_label) == "true":
        return "The local pipeline already contains this as a true case, so it was not missed in the final outputs."
    if stage_matches["model3"] or final_review_match or master_review_match:
        final_label = normalize_label(
            (stage_matches["model3"].validation_label if stage_matches["model3"] else "")
            or (master_review_match.validation_label if master_review_match else "")
        )
        if final_label == "false":
            return "The firm reached late-stage validation, but the local pipeline classified it as a non-target case."
        if final_label == "unclear":
            return "The firm reached late-stage validation, but the local pipeline kept it as uncertain rather than true."
        return "The firm reached late-stage outputs, but the final labels differ across workflows."
    if stage_matches["model2"] and not stage_matches["model3"]:
        return "The firm survived discovery and enrichment but appears to have failed strict Model 3 validation."
    if stage_matches["model1"] and not stage_matches["model2"]:
        return "The firm was surfaced early but appears to have been lost during Model 2 enrichment/reconciliation."
    if stage_matches["dedup"] and not stage_matches["model1"]:
        return "The firm survived discovery deduplication but appears to have been dropped by recall-oriented Model 1."
    if stage_matches["discovery"] and not stage_matches["dedup"]:
        return "The firm appeared in discovery but not after deduplication, suggesting known-firm exclusion or duplicate collapsing."
    return "The firm does not appear in the local staged outputs, so it was likely missed at discovery or excluded before local pipeline ingestion."


def combined_comment(row: dict[str, Any]) -> str:
    parts = [
        safe_text(row.get("comment_final")),
        safe_text(row.get("additional_comment")),
        safe_text(row.get("relocation")),
    ]
    return " ".join(part.casefold() for part in parts if part)


def normalize_label(value: Any) -> str:
    return safe_text(value).strip().lower()


def matched_name(match: MatchRecord | None) -> str:
    return match.matched_name if match else ""


def matched_path(match: MatchRecord | None) -> str:
    return match.path if match else ""


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], path: Path, csv_output_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row["status_bucket"] for row in rows)
    pipeline_counts = Counter(row["pipeline_outcome"] for row in rows)

    manual_true_rows = [row for row in rows if row["status_bucket"] == "manual_true"]
    true_subcounts = Counter(row["pipeline_outcome"] for row in manual_true_rows)

    missing_true = [row for row in manual_true_rows if row["pipeline_outcome"] == "never_seen_in_local_pipeline_outputs"]
    disagree_true = [row for row in manual_true_rows if row["pipeline_outcome"] == "pipeline_surfaced_but_final_label_differs"]

    lines = [
        "# Results LLM Prompting Audit",
        "",
        "## Overview",
        f"- Total workbook rows audited: {len(rows)}",
        f"- Output CSV: `{csv_output_path.as_posix()}`",
        "",
        "## Manual Outcome Counts",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(status_counts.items()))
    lines.extend(["", "## Pipeline Outcome Counts"])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(pipeline_counts.items()))
    lines.extend(["", "## Manual True Cases"])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(true_subcounts.items()))

    if missing_true:
        lines.extend(["", "## Manual True Cases Never Seen Locally"])
        lines.extend(f"- {row['firm']}" for row in missing_true)

    if disagree_true:
        lines.extend(["", "## Manual True Cases With Different Local Final Labels"])
        lines.extend(
            f"- {row['firm']} | master_review_label={row['master_review_label'] or '<none>'} | likely_reason={row['likely_reason']}"
            for row in disagree_true
        )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
