"""LLM-driven snowball discovery from a known seed list.

Example:
python snowball_discovery.py ^
  --known preliminary_data_28_04.csv ^
  --output data/discovery/snowball_round_001.jsonl ^
  --round 1 ^
  --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.seed_list import build_core_relocation_names, build_exclusion_list, load_seed_firms
from src.founding_eligibility import is_post_1999_eligible_or_supported
from src.io import save_jsonl

DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "2026-04-29-snowball-v3"
DEFAULT_PROMPT_PATH = Path("prompts/snowball_discovery_prompt.txt")

SECTOR_BUCKETS = [
    "biotech",
    "SaaS/software",
    "fintech",
    "gaming",
    "medtech",
    "logistics",
    "cleantech",
    "retail/consumer",
    "hospitality",
    "design",
]

DESTINATION_BUCKETS = [
    "US",
    "UK",
    "Germany",
    "Sweden",
    "Switzerland",
    "Netherlands",
    "Ireland",
    "Portugal",
    "Italy",
    "Spain",
    "Singapore",
    "Canada",
    "China",
    "UAE",
    "South Africa",
]

MECHANISM_BUCKETS = [
    "IPO or listing-related relocation",
    "Delaware parent or UK parent after Danish founding",
    "principal executive offices moved abroad",
    "group headquarters moved abroad",
    "executive team or leadership moved abroad",
    "acquisition-linked relocation where the operating company continued",
    "founder relocation narrative after Denmark founding",
    "stealth or scaleup expansion that became a foreign HQ move",
]

SOURCE_STYLE_BUCKETS = [
    "Danish business media",
    "foreign media covering Danish-founded firms",
    "investor filings and annual reports",
    "company about pages and archived timeline pages",
]

CITY_PAIR_BUCKETS = [
    "Copenhagen to London",
    "Copenhagen to New York",
    "Copenhagen to San Francisco",
    "Aarhus to London",
    "Aarhus to New York",
    "Denmark to Cambridge MA",
]

SYSTEM_PROMPT = """You are a recall-first snowball discovery assistant for a company research pipeline.

Your task is to find additional, lesser-known firms that may have been founded in Denmark and may later have moved their own executive headquarters, main office, executive leadership base, or main operations abroad.

Rules:
- prioritize obscure or long-tail firms over famous repeated examples
- exclude every known firm provided by the user
- return only firms founded in 2000 or later
- do not return firms clearly founded before 2000
- if founding year is unknown, keep only firms whose source-backed evidence strongly suggests a young firm or startup founded after 1999
- return only source-backed candidates
- do not return Danish-founder-abroad cases unless the firm itself appears founded, based, or incorporated in Denmark
- do not treat foreign office openings alone as headquarters relocation
- do not treat acquisition-only stories as enough unless the Danish-origin firm appears to continue operating with its own HQ or main operations abroad
- search for relocation mechanisms beyond the literal phrase "moved headquarters", including principal executive offices, global headquarters, group headquarters, parent-company restructuring, IPO or listing re-domiciliation, executive-team relocation, and operating-company continuation after M&A
- pay attention to former names, legal entity names, rebrands, parent/subsidiary naming variants, and slash/formerly aliases
- prefer Danish business media, foreign reporting, investor filings, archived company pages, and registry-style sources over low-signal listicles
- keep uncertain but plausible candidates
- do not fabricate URLs, dates, or locations
- return only JSON matching the requested schema"""


@dataclass(frozen=True)
class DiscoveryBucket:
    bucket_type: str
    bucket_value: str

    @property
    def bucket_id(self) -> str:
        return f"{self.bucket_type}:{self.bucket_value}"


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


def build_discovery_buckets() -> list[DiscoveryBucket]:
    buckets: list[DiscoveryBucket] = []
    for value in SECTOR_BUCKETS:
        buckets.append(DiscoveryBucket(bucket_type="sector", bucket_value=value))
    for value in DESTINATION_BUCKETS:
        buckets.append(DiscoveryBucket(bucket_type="destination", bucket_value=value))
    for value in MECHANISM_BUCKETS:
        buckets.append(DiscoveryBucket(bucket_type="mechanism", bucket_value=value))
    for value in SOURCE_STYLE_BUCKETS:
        buckets.append(DiscoveryBucket(bucket_type="source_style", bucket_value=value))
    for value in CITY_PAIR_BUCKETS:
        buckets.append(DiscoveryBucket(bucket_type="city_pair", bucket_value=value))
    return buckets


def load_prompt_template(path: Path = DEFAULT_PROMPT_PATH) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return (
        "Find additional Danish-founded firms that may have moved their own executive headquarters, "
        "main office, main operations, or leadership abroad. Exclude all known firms listed below. "
        "Prioritize obscure or less-known firms. Return only source-backed candidates founded in 2000 or later. "
        "Search beyond literal headquarters-move wording and pay attention to principal executive offices, global or group HQ language, IPO or parent-company restructurings, founder or leadership relocation narratives, and former-name or legal-entity aliases."
    )


def snowball_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "firm_name",
                        "discovery_bucket",
                        "sector_if_known",
                        "possible_founding_location",
                        "possible_founding_year",
                        "possible_abroad_hq_location",
                        "possible_move_year",
                        "why_candidate",
                        "founding_denmark_evidence",
                        "abroad_hq_evidence",
                        "ma_context",
                        "source_urls",
                        "uncertainty_note",
                    ],
                    "properties": {
                        "firm_name": {"type": "string"},
                        "discovery_bucket": {"type": "string"},
                        "sector_if_known": {"type": ["string", "null"]},
                        "possible_founding_location": {"type": ["string", "null"]},
                        "possible_founding_year": {"type": ["integer", "null"]},
                        "possible_abroad_hq_location": {"type": ["string", "null"]},
                        "possible_move_year": {"type": ["integer", "null"]},
                        "why_candidate": {"type": "string"},
                        "founding_denmark_evidence": {"type": "string"},
                        "abroad_hq_evidence": {"type": "string"},
                        "ma_context": {"type": ["string", "null"]},
                        "source_urls": {"type": "array", "items": {"type": "string"}},
                        "uncertainty_note": {"type": "string"},
                    },
                },
            }
        },
    }


def build_discovery_prompt(
    bucket: DiscoveryBucket,
    exclusion_names: list[str],
    core_relocation_names: list[str],
    prompt_template: str,
) -> str:
    payload = {
        "task": prompt_template,
        "bucket": {
            "bucket_type": bucket.bucket_type,
            "bucket_value": bucket.bucket_value,
            "discovery_bucket": bucket.bucket_id,
        },
        "research_target": {
            "target_case": "firms founded in Denmark that may later have moved their own HQ or main operations abroad",
            "non_target_case": "abroad (Danish founders) without the firm itself being founded or based in Denmark",
            "founding_year_rule": "founding_year >= 2000; unknown founding year is allowed only with strong evidence of a post-1999 startup or young firm",
            "relocation_mechanisms_to_search": [
                "principal executive offices moved abroad",
                "global or group headquarters moved abroad",
                "IPO or listing-related re-domiciliation with executive relocation",
                "creation of a foreign parent or holding company after Danish founding",
                "acquisition-linked relocation where the operating company still exists abroad",
                "founder or executive relocation narrative tied to the firm's HQ",
            ],
            "name_handling": [
                "look for former names",
                "look for rebrands",
                "look for legal entity names",
                "look for slash-separated or formerly-style aliases",
            ],
            "core_known_cases_already_covered": core_relocation_names,
        },
        "requirements": {
            "prioritize": [
                "obscure firms",
                "long-tail firms",
                "source-backed candidates",
                "cases outside the usual famous examples",
                "Danish-language and foreign-language reporting when relevant",
                "city-pair and destination-specific relocation narratives",
            ],
            "avoid": [
                "any known firm in the exclusion list",
                "pure acquisition-only cases without continued operating-company evidence abroad",
                "foreign office openings alone",
                "Danish-founder-abroad cases unless the company itself was founded or based in Denmark",
                "firms clearly founded before 2000",
            ],
            "prefer_sources": [
                "Danish business media",
                "foreign business media",
                "investor filings",
                "company about pages",
                "archived company pages",
                "registry-style sources",
            ],
            "return_only_candidates_with_real_sources": True,
        },
        "output_fields": [
            "firm_name",
            "discovery_bucket",
            "sector_if_known",
            "possible_founding_location",
            "possible_founding_year",
            "possible_abroad_hq_location",
            "possible_move_year",
            "why_candidate",
            "founding_denmark_evidence",
            "abroad_hq_evidence",
            "ma_context",
            "source_urls",
            "uncertainty_note",
        ],
        "known_firm_exclusions": exclusion_names,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_followup_discovery_prompt(
    bucket: DiscoveryBucket,
    exclusion_names: list[str],
    core_relocation_names: list[str],
    prompt_template: str,
    already_found_names: list[str],
) -> str:
    payload = {
        "task": (
            "Find additional firms beyond the already-found names from the first pass for this same bucket. "
            f"{prompt_template}"
        ),
        "bucket": {
            "bucket_type": bucket.bucket_type,
            "bucket_value": bucket.bucket_value,
            "discovery_bucket": bucket.bucket_id,
        },
        "research_target": {
            "target_case": "firms founded in Denmark that may later have moved their own HQ or main operations abroad",
            "non_target_case": "abroad (Danish founders) without the firm itself being founded or based in Denmark",
            "founding_year_rule": "founding_year >= 2000; unknown founding year is allowed only with strong evidence of a post-1999 startup or young firm",
            "followup_instruction": "Return only additional firms not already found in the first pass for this bucket.",
            "core_known_cases_already_covered": core_relocation_names,
        },
        "requirements": {
            "prioritize": [
                "additional obscure firms",
                "long-tail firms not already named",
                "source-backed candidates",
                "cases outside the usual famous examples",
            ],
            "avoid": [
                "any known firm in the exclusion list",
                "any firm already found in the first pass for this bucket",
                "pure acquisition-only cases without continued operating-company evidence abroad",
                "foreign office openings alone",
                "Danish-founder-abroad cases unless the company itself was founded or based in Denmark",
                "firms clearly founded before 2000",
            ],
            "return_only_candidates_with_real_sources": True,
        },
        "output_fields": [
            "firm_name",
            "discovery_bucket",
            "sector_if_known",
            "possible_founding_location",
            "possible_founding_year",
            "possible_abroad_hq_location",
            "possible_move_year",
            "why_candidate",
            "founding_denmark_evidence",
            "abroad_hq_evidence",
            "ma_context",
            "source_urls",
            "uncertainty_note",
        ],
        "known_firm_exclusions": exclusion_names,
        "already_found_firms_this_bucket": already_found_names,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_openai_snowball_discovery(user_prompt: str, model: str) -> tuple[dict[str, Any], Any]:
    load_openai_api_key()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Snowball discovery requires the openai package. Install dependencies with "
            "`pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        text={
            "format": {
                "type": "json_schema",
                "name": "snowball_discovery",
                "strict": True,
                "schema": snowball_json_schema(),
            }
        },
    )
    return json.loads(response.output_text), response


def serialize_openai_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    return {"response_repr": repr(response)}


def parse_discovery_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in payload.get("candidates") or []:
        founding_year = candidate.get("possible_founding_year")
        record = {
            "firm_name": str(candidate.get("firm_name") or "").strip(),
            "discovery_bucket": str(candidate.get("discovery_bucket") or "").strip(),
            "sector_if_known": candidate.get("sector_if_known"),
            "possible_founding_location": candidate.get("possible_founding_location"),
            "possible_founding_year": founding_year,
            "possible_abroad_hq_location": candidate.get("possible_abroad_hq_location"),
            "possible_move_year": candidate.get("possible_move_year"),
            "why_candidate": str(candidate.get("why_candidate") or "").strip(),
            "founding_denmark_evidence": str(candidate.get("founding_denmark_evidence") or "").strip(),
            "abroad_hq_evidence": str(candidate.get("abroad_hq_evidence") or "").strip(),
            "ma_context": candidate.get("ma_context"),
            "source_urls": [str(url).strip() for url in candidate.get("source_urls") or [] if str(url).strip()],
            "uncertainty_note": str(candidate.get("uncertainty_note") or "").strip(),
        }
        if is_post_1999_eligible_or_supported(
            founding_year,
            record["why_candidate"],
            record["founding_denmark_evidence"],
            record["abroad_hq_evidence"],
            record["uncertainty_note"],
            record["ma_context"],
        ):
            records.append(record)
    return records


def build_bucket_run_record(
    bucket: DiscoveryBucket,
    round_number: int,
    model: str,
    prompt_text: str,
    raw_response: Any,
    parsed_payload: dict[str, Any],
    pass_type: str = "initial",
) -> dict[str, Any]:
    return {
        "round": round_number,
        "model": model,
        "bucket_type": bucket.bucket_type,
        "bucket_value": bucket.bucket_value,
        "discovery_bucket": bucket.bucket_id,
        "pass_type": pass_type,
        "prompt_version": PROMPT_VERSION,
        "prompt_text": prompt_text,
        "raw_response": serialize_openai_response(raw_response),
        "parsed_output": parsed_payload,
    }


def build_candidate_output_records(
    bucket: DiscoveryBucket,
    round_number: int,
    model: str,
    prompt_text: str,
    parsed_candidates: list[dict[str, Any]],
    raw_response: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in parsed_candidates:
        discovery_bucket = candidate["discovery_bucket"] or bucket.bucket_id
        records.append(
            {
                **candidate,
                "discovery_bucket": discovery_bucket,
                "bucket_type": bucket.bucket_type,
                "bucket_value": bucket.bucket_value,
                "round": round_number,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "prompt_text": prompt_text,
                "raw_response": serialize_openai_response(raw_response),
            }
        )
    return records


def run_snowball_discovery(
    known_path: Path,
    output_path: Path,
    round_number: int,
    model: str,
    max_buckets: int | None = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    followup_pass: bool = True,
) -> list[dict[str, Any]]:
    known_firms = load_seed_firms(known_path)
    exclusion_names = build_exclusion_list(known_firms)
    core_relocation_names = build_core_relocation_names(known_firms)
    prompt_template = load_prompt_template(prompt_path)
    buckets = build_discovery_buckets()
    if max_buckets is not None:
        buckets = buckets[:max_buckets]

    all_candidate_records: list[dict[str, Any]] = []
    bucket_run_records: list[dict[str, Any]] = []

    for position, bucket in enumerate(buckets, start=1):
        print(f"[{position}/{len(buckets)}] bucket={bucket.bucket_id}")
        prompt_text = build_discovery_prompt(
            bucket=bucket,
            exclusion_names=exclusion_names,
            core_relocation_names=core_relocation_names,
            prompt_template=prompt_template,
        )
        parsed_payload, raw_response = call_openai_snowball_discovery(user_prompt=prompt_text, model=model)
        parsed_candidates = parse_discovery_candidates(parsed_payload)
        bucket_run_records.append(
            build_bucket_run_record(
                bucket=bucket,
                round_number=round_number,
                model=model,
                prompt_text=prompt_text,
                raw_response=raw_response,
                parsed_payload=parsed_payload,
                pass_type="initial",
            )
        )
        all_candidate_records.extend(
            build_candidate_output_records(
                bucket=bucket,
                round_number=round_number,
                model=model,
                prompt_text=prompt_text,
                parsed_candidates=parsed_candidates,
                raw_response=raw_response,
            )
        )

        if followup_pass and parsed_candidates:
            already_found_names = list(
                dict.fromkeys(candidate["firm_name"] for candidate in parsed_candidates if candidate["firm_name"])
            )
            followup_prompt_text = build_followup_discovery_prompt(
                bucket=bucket,
                exclusion_names=exclusion_names,
                core_relocation_names=core_relocation_names,
                prompt_template=prompt_template,
                already_found_names=already_found_names,
            )
            followup_payload, followup_response = call_openai_snowball_discovery(
                user_prompt=followup_prompt_text,
                model=model,
            )
            followup_candidates = parse_discovery_candidates(followup_payload)
            bucket_run_records.append(
                build_bucket_run_record(
                    bucket=bucket,
                    round_number=round_number,
                    model=model,
                    prompt_text=followup_prompt_text,
                    raw_response=followup_response,
                    parsed_payload=followup_payload,
                    pass_type="followup",
                )
            )
            all_candidate_records.extend(
                build_candidate_output_records(
                    bucket=bucket,
                    round_number=round_number,
                    model=model,
                    prompt_text=followup_prompt_text,
                    parsed_candidates=followup_candidates,
                    raw_response=followup_response,
                )
            )

    save_jsonl(all_candidate_records, output_path)
    bucket_runs_path = output_path.with_name(f"{output_path.stem}_bucket_runs{output_path.suffix}")
    save_jsonl(bucket_run_records, bucket_runs_path)
    return all_candidate_records


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM-driven snowball discovery from known firms.")
    parser.add_argument("--known", type=Path, required=True, help="Known seed CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Candidate discovery JSONL path.")
    parser.add_argument("--round", type=int, required=True, dest="round_number", help="Discovery round number.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI Responses model name.")
    parser.add_argument(
        "--max-buckets",
        type=int,
        default=None,
        help="Optional cap for cheap testing across the combined bucket list.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Optional custom prompt template path.",
    )
    parser.add_argument(
        "--no-followup-pass",
        action="store_false",
        dest="followup_pass",
        help="Disable the bounded second-pass 'find more' discovery prompt for each bucket.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_snowball_discovery(
        known_path=args.known,
        output_path=args.output,
        round_number=args.round_number,
        model=args.model,
        max_buckets=args.max_buckets,
        prompt_path=args.prompt_path,
        followup_pass=args.followup_pass,
    )


if __name__ == "__main__":
    main()
