"""Model 3 strict validation from Model 2 enriched candidate JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from export_final_review import export_final_review
from src.founding_eligibility import has_strong_post_1998_evidence, normalize_year
from src.io import save_jsonl
from src.normalization import normalize_company_name
from src.openai_batch import build_batch_request_item, extract_json_output_payload, run_responses_batch
from src.openai_costs import build_cost_record, cost_log_path, sum_cost_records

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

    tqdm.write = print  # type: ignore[attr-defined]

DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "2026-04-30-model3-v3"
DEFAULT_MASTER_VALIDATED_PATH = Path("data/cumulative/model3_validated_master_all_tracks.jsonl")
DEFAULT_MASTER_REVIEW_PATH = Path("data/cumulative/final_review_master_all_tracks.csv")
DEFAULT_TRACK_MASTER_VALIDATED_PATHS = {
    "in_denmark": Path("data/cumulative/model3_validated_master_in_denmark.jsonl"),
    "abroad_danish_founders": Path("data/cumulative/model3_validated_master_abroad_danish_founders.jsonl"),
}
DEFAULT_TRACK_MASTER_REVIEW_PATHS = {
    "in_denmark": Path("data/cumulative/final_review_master_in_denmark.csv"),
    "abroad_danish_founders": Path("data/cumulative/final_review_master_abroad_danish_founders.csv"),
}

SYSTEM_PROMPT = """You are performing final strict validation for a research pipeline.

Question:
Does this candidate meet the research definition?

The input record will specify an origin_track.

Strict rules:
- validation_label=false if founding_year < 1999
- validation_label=unclear if founding_year is unknown and no strong 1999-or-later evidence exists
- validation_label=true only if founding_year >= 1999 and the track-specific criteria are met
- for in_denmark:
  - foreign office or subsidiary alone is not enough
  - Danish founder abroad alone is not enough
  - acquisition-only is not enough
  - regional HQ or service-center HQ is not enough unless it reflects the focal firm's main executive HQ or main operations
  - legal redomiciling is not automatically executive HQ relocation
  - true requires evidence that the firm itself was founded, based, or incorporated in Denmark and later moved its own executive HQ, main office, or main operations abroad
- for abroad_danish_founders:
  - true requires evidence that the firm was founded abroad and that at least one founder appears Danish from source-backed evidence
  - true is not allowed if the record instead looks founded in Denmark
  - relocation is optional context, not a requirement

Return only JSON matching the requested schema."""


def load_openai_api_key(dotenv_path: Path = Path(".env")) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    if dotenv_path.exists():
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != "OPENAI_API_KEY":
                continue
            parsed = value.strip().strip("'").strip('"')
            if parsed:
                os.environ["OPENAI_API_KEY"] = parsed
                return parsed

    raise RuntimeError("OPENAI_API_KEY is required in the environment or a local .env file.")


def load_enriched_candidates(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_validated_candidates(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_user_prompt(record: dict[str, Any]) -> str:
    payload = {
        "origin_track": str(record.get("origin_track") or "in_denmark"),
        "task": "Apply the final conservative validation gate to this enriched candidate.",
        "candidate_input": record,
        "output_constraints": {
            "one_record_only": True,
            "strict_definition": True,
            "preserve_source_urls": True,
            "founding_year_rule": {
                "false_if_founding_year_before_1999": True,
                "unclear_if_founding_year_unknown_without_strong_1999_or_later_evidence": True,
                "true_requires_founding_year_1999_or_later": True,
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def validation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["record"],
        "properties": {
            "record": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "validation_label",
                    "validation_reason",
                    "exclusion_reason",
                    "evidence_summary",
                    "needs_human_review",
                ],
                "properties": {
                    "validation_label": {"type": "string", "enum": ["true", "false", "unclear"]},
                    "validation_reason": {"type": "string"},
                    "exclusion_reason": {"type": ["string", "null"]},
                    "evidence_summary": {"type": "string"},
                    "needs_human_review": {"type": "boolean"},
                },
            }
        },
    }


def build_response_request_body(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "model_3_validation",
                "strict": True,
                "schema": validation_json_schema(),
            }
        },
    }


def validate_candidate(record: dict[str, Any], model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    load_openai_api_key()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Model 3 validation requires the openai package. Install dependencies with "
            "`pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    response = client.responses.create(**build_response_request_body(record, model_name))
    parsed = json.loads(response.output_text)
    return (
        parse_validation_record(record, parsed),
        build_cost_record(
            stage="model3",
            request_kind="validation",
            raw_response=response,
            requested_model=model_name,
            metadata={"firm_name": str(record.get("firm_name") or "").strip()},
        ),
    )


def parse_validation_record(record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    validated = payload.get("record") or {}
    label = str(validated.get("validation_label") or "unclear")
    if label not in {"true", "false", "unclear"}:
        label = "unclear"

    founding_year = normalize_year(record.get("founding_year"))
    origin_track = _normalize_origin_track(record.get("origin_track"))
    strong_post_1998_evidence = has_strong_post_1998_evidence(
        record.get("founding_evidence"),
        record.get("founder_danish_context"),
        record.get("uncertainty_note"),
        record.get("status_today_context"),
        record.get("relocation_context"),
    )

    validation_reason = str(validated.get("validation_reason") or "")
    exclusion_reason = validated.get("exclusion_reason")

    if founding_year is not None and founding_year < 1999:
        label = "false"
        exclusion_reason = "Founded before 1999."
        validation_reason = _append_reason(
            validation_reason,
            "Founding year is before 1999, so the candidate is outside scope.",
        )
    elif founding_year is None and not strong_post_1998_evidence:
        label = "unclear"
        validation_reason = _append_reason(
            validation_reason,
            "Founding year is unknown and the record lacks strong evidence that the firm was founded in 1999 or later.",
        )
    elif founding_year is None and label == "true":
        label = "unclear"
        validation_reason = _append_reason(
            validation_reason,
            "True is not allowed when founding year is unknown.",
        )
    elif origin_track == "abroad_danish_founders":
        founder_link = _tri_state(record.get("danish_founders_abroad"))
        founded_in_denmark = _tri_state(record.get("founded_in_denmark"))
        if founded_in_denmark == "true":
            label = "false"
            exclusion_reason = "The record appears founded in Denmark rather than abroad."
            validation_reason = _append_reason(
                validation_reason,
                "This founder-abroad track excludes firms that instead appear founded in Denmark.",
            )
        elif founder_link != "true":
            label = "unclear" if founder_link == "uncertain" else "false"
            if founder_link == "false":
                exclusion_reason = "No sufficient source-backed Danish founder link."
            validation_reason = _append_reason(
                validation_reason,
                "The founder-abroad track requires a source-backed Danish founder link.",
            )
    else:
        founded_in_denmark = _tri_state(record.get("founded_in_denmark"))
        if founded_in_denmark == "false":
            label = "false"
            validation_reason = _append_reason(
                validation_reason,
                "The Denmark-founded track requires source-backed founding in Denmark.",
            )

    return {
        **record,
        "validation_label": label,
        "validation_reason": validation_reason,
        "exclusion_reason": exclusion_reason,
        "evidence_summary": str(validated.get("evidence_summary") or ""),
        "needs_human_review": bool(validated.get("needs_human_review")),
        "prompt_version_model3": PROMPT_VERSION,
    }


def _append_reason(existing: str, added: str) -> str:
    existing = existing.strip()
    if not existing:
        return added
    if added in existing:
        return existing
    return f"{existing} {added}"


def _normalize_origin_track(value: Any) -> str:
    if value == "abroad_danish_founders":
        return "abroad_danish_founders"
    return "in_denmark"


def _tri_state(value: Any) -> str:
    if value in {"true", "false", "uncertain"}:
        return str(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "uncertain"


def save_validated_records(records: list[dict[str, Any]], output_path: Path) -> None:
    save_jsonl(records, output_path)


def update_master_validated_dataset(
    records: list[dict[str, Any]],
    master_validated_path: Path,
    master_review_path: Path,
    track_master_validated_paths: dict[str, Path] | None = None,
    track_master_review_paths: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for existing in load_validated_candidates(master_validated_path):
        normalized = normalize_company_name(existing.get("firm_name"))
        if normalized:
            merged[normalized] = existing
    for record in records:
        normalized = normalize_company_name(record.get("firm_name"))
        if normalized:
            merged[normalized] = record
    merged_records = sorted(merged.values(), key=lambda row: str(row.get("firm_name") or "").casefold())
    save_jsonl(merged_records, master_validated_path)
    export_final_review(merged_records, master_review_path)
    _update_track_specific_masters(
        merged_records,
        track_master_validated_paths=track_master_validated_paths or DEFAULT_TRACK_MASTER_VALIDATED_PATHS,
        track_master_review_paths=track_master_review_paths or DEFAULT_TRACK_MASTER_REVIEW_PATHS,
    )
    return merged_records


def _update_track_specific_masters(
    records: list[dict[str, Any]],
    track_master_validated_paths: dict[str, Path],
    track_master_review_paths: dict[str, Path],
) -> None:
    for origin_track, validated_path in track_master_validated_paths.items():
        filtered = [
            record
            for record in records
            if _normalize_origin_track(record.get("origin_track")) == origin_track
        ]
        save_jsonl(filtered, validated_path)
        review_path = track_master_review_paths.get(origin_track)
        if review_path is not None:
            export_final_review(filtered, review_path)


def run_model_3(
    input_path: Path,
    output_path: Path,
    model_name: str,
    limit: int | None = None,
    batch: bool = False,
    master_validated_path: Path = DEFAULT_MASTER_VALIDATED_PATH,
    master_review_path: Path = DEFAULT_MASTER_REVIEW_PATH,
) -> list[dict[str, Any]]:
    records_in = load_enriched_candidates(input_path)
    if limit is not None:
        records_in = records_in[:limit]
    records_out: list[dict[str, Any]] = []
    cost_records: list[dict[str, Any]] = []
    if batch and records_in:
        load_openai_api_key()
        from openai import OpenAI

        client = OpenAI()
        request_items = [
            build_batch_request_item(
                custom_id=f"model3-{index}",
                body=build_response_request_body(record, model_name),
            )
            for index, record in enumerate(records_in)
        ]
        responses_by_custom_id, _batch_payload = run_responses_batch(client=client, request_items=request_items)
        for index, record in enumerate(records_in):
            response_body = responses_by_custom_id[f"model3-{index}"]
            parsed = extract_json_output_payload(response_body)
            records_out.append(parse_validation_record(record, parsed))
            cost_records.append(
                build_cost_record(
                    stage="model3",
                    request_kind="validation_batch",
                    raw_response=response_body,
                    requested_model=model_name,
                    metadata={"firm_name": str(record.get("firm_name") or "").strip(), "batch": True},
                )
            )
    else:
        for record in tqdm(records_in, total=len(records_in), desc="Model 3", unit="firm"):
            validated_record, cost_record = validate_candidate(record=record, model_name=model_name)
            records_out.append(validated_record)
            cost_records.append(cost_record)
    save_validated_records(records_out, output_path)
    merged_records = update_master_validated_dataset(
        records_out,
        master_validated_path=master_validated_path,
        master_review_path=master_review_path,
    )
    save_jsonl(cost_records, cost_log_path(output_path))
    totals = sum_cost_records(cost_records)
    tqdm.write(f"model_3_input_records={len(records_in)}")
    tqdm.write(f"estimated_cost_usd={totals['estimated_cost_usd']:.6f}")
    tqdm.write(f"api_cost_log_path={cost_log_path(output_path)}")
    tqdm.write(f"master_validated_records={len(merged_records)}")
    tqdm.write(f"master_validated_path={master_validated_path}")
    tqdm.write(f"master_review_path={master_review_path}")
    for origin_track, track_path in DEFAULT_TRACK_MASTER_VALIDATED_PATHS.items():
        tqdm.write(f"track_master_validated_path[{origin_track}]={track_path}")
    for origin_track, track_path in DEFAULT_TRACK_MASTER_REVIEW_PATHS.items():
        tqdm.write(f"track_master_review_path[{origin_track}]={track_path}")
    return records_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Model 3 strict validation from Model 2 enriched JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Path to Model 2 enriched JSONL input.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write validated JSONL output.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for small debugging runs.")
    parser.add_argument("--batch", action="store_true", help="Submit requests through the OpenAI Batch API.")
    parser.add_argument(
        "--master-validated-output",
        type=Path,
        default=DEFAULT_MASTER_VALIDATED_PATH,
        help="Cumulative master validated JSONL path updated after each run.",
    )
    parser.add_argument(
        "--master-review-output",
        type=Path,
        default=DEFAULT_MASTER_REVIEW_PATH,
        help="Cumulative master review CSV path regenerated after each run.",
    )
    args = parser.parse_args()

    records = run_model_3(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        limit=args.limit,
        batch=args.batch,
        master_validated_path=args.master_validated_output,
        master_review_path=args.master_review_output,
    )
    print(f"prompt_version={PROMPT_VERSION}")
    print(f"validated_records={len(records)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
