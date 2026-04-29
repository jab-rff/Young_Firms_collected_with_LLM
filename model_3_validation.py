"""Model 3 strict validation from Model 2 enriched candidate JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.founding_eligibility import has_strong_post_1999_evidence, normalize_year
from src.io import save_jsonl

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

    tqdm.write = print  # type: ignore[attr-defined]

DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "2026-04-28-model3-v2"

SYSTEM_PROMPT = """You are performing final strict validation for a research pipeline.

Question:
Does this candidate meet the research definition?

Strict rules:
- validation_label=false if founding_year < 2000
- validation_label=unclear if founding_year is unknown and no strong post-1999 evidence exists
- validation_label=true only if founding_year >= 2000 and the relocation criteria are met
- foreign office or subsidiary alone is not enough
- Danish founder abroad alone is not enough
- acquisition-only is not enough
- regional HQ or service-center HQ is not enough unless it reflects the focal firm's main executive HQ or main operations
- legal redomiciling is not automatically executive HQ relocation
- true requires evidence that the firm itself was founded, based, or incorporated in Denmark and later moved its own executive HQ, main office, or main operations abroad

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


def build_user_prompt(record: dict[str, Any]) -> str:
    payload = {
        "task": "Apply the final conservative validation gate to this enriched candidate.",
        "candidate_input": record,
        "output_constraints": {
            "one_record_only": True,
            "strict_definition": True,
            "preserve_source_urls": True,
            "founding_year_rule": {
                "false_if_founding_year_before_2000": True,
                "unclear_if_founding_year_unknown_without_strong_post_1999_evidence": True,
                "true_requires_founding_year_2000_or_later": True,
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


def validate_candidate(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    load_openai_api_key()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Model 3 validation requires the openai package. Install dependencies with "
            "`pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "model_3_validation",
                "strict": True,
                "schema": validation_json_schema(),
            }
        },
    )
    parsed = json.loads(response.output_text)
    return parse_validation_record(record, parsed)


def parse_validation_record(record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    validated = payload.get("record") or {}
    label = str(validated.get("validation_label") or "unclear")
    if label not in {"true", "false", "unclear"}:
        label = "unclear"

    founding_year = normalize_year(record.get("founding_year"))
    strong_post_1999_evidence = has_strong_post_1999_evidence(
        record.get("founding_evidence"),
        record.get("uncertainty_note"),
        record.get("confidence_note"),
        record.get("reasoning"),
        record.get("status_today_context"),
        record.get("relocation_context"),
    )

    validation_reason = str(validated.get("validation_reason") or "")
    exclusion_reason = validated.get("exclusion_reason")

    if founding_year is not None and founding_year < 2000:
        label = "false"
        exclusion_reason = "Founded before 2000."
        validation_reason = _append_reason(
            validation_reason,
            "Founding year is before 2000, so the candidate is outside scope.",
        )
    elif founding_year is None and not strong_post_1999_evidence:
        label = "unclear"
        validation_reason = _append_reason(
            validation_reason,
            "Founding year is unknown and the record lacks strong evidence that the firm was founded after 1999.",
        )
    elif founding_year is None and label == "true":
        label = "unclear"
        validation_reason = _append_reason(
            validation_reason,
            "True is not allowed when founding year is unknown.",
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


def save_validated_records(records: list[dict[str, Any]], output_path: Path) -> None:
    save_jsonl(records, output_path)


def run_model_3(
    input_path: Path,
    output_path: Path,
    model_name: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records_in = load_enriched_candidates(input_path)
    if limit is not None:
        records_in = records_in[:limit]
    records_out: list[dict[str, Any]] = []
    for record in tqdm(records_in, total=len(records_in), desc="Model 3", unit="firm"):
        records_out.append(validate_candidate(record=record, model_name=model_name))
    save_validated_records(records_out, output_path)
    tqdm.write(f"model_3_input_records={len(records_in)}")
    return records_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Model 3 strict validation from Model 2 enriched JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Path to Model 2 enriched JSONL input.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write validated JSONL output.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for small debugging runs.")
    args = parser.parse_args()

    records = run_model_3(input_path=args.input, output_path=args.output, model_name=args.model, limit=args.limit)
    print(f"prompt_version={PROMPT_VERSION}")
    print(f"validated_records={len(records)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
