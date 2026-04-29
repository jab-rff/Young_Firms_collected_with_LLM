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

from discovery_memory import load_known_firms, save_known_firms, update_known_firms_from_rows
from src.seed_list import (
    build_core_relocation_names,
    build_exclusion_list,
    load_seed_firms,
    select_discovery_prompt_firms,
)
from src.normalization import normalize_company_name
from src.founding_eligibility import is_post_1998_eligible_or_supported
from src.io import save_jsonl
from src.openai_costs import build_cost_record, cost_log_path, serialize_openai_response, sum_cost_records

DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "2026-04-29-snowball-v5"
DEFAULT_PROMPT_PATHS = {
    "in_denmark": Path("prompts/snowball_discovery_prompt.txt"),
    "abroad_danish_founders": Path("prompts/snowball_discovery_abroad_danish_founders_prompt.txt"),
}
DEFAULT_DISCOVERY_MEMORY_PATH = Path("data/memory/discovery_known_firms.jsonl")

ALLOWED_SIGNAL_TYPES = [
    "explicit_hq_move",
    "foreign_principal_office",
    "foreign_parent_or_redomiciliation",
    "acquisition_linked_continuation",
    "foreign_office_or_operations",
    "leadership_abroad",
    "other_plausible_signal",
]

ALLOWED_SIGNAL_STRENGTHS = ["strong", "medium", "weak"]

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

CROSS_BUCKETS = [
    "sector=biotech | destination=US",
    "sector=SaaS/software | destination=US",
    "sector=SaaS/software | destination=UK",
    "sector=fintech | destination=UK",
    "sector=medtech | destination=US",
    "sector=gaming | destination=US",
    "sector=design | destination=UK",
    "sector=retail/consumer | destination=Germany",
    "sector=biotech | mechanism=IPO or listing-related relocation",
    "sector=biotech | mechanism=Delaware parent or UK parent after Danish founding",
    "destination=US | mechanism=principal executive offices moved abroad",
    "destination=UK | mechanism=group headquarters moved abroad",
]

SYSTEM_PROMPT = """You are a recall-first snowball discovery assistant for a company research pipeline.

The user will provide an explicit origin track. Follow that track exactly.

Possible tracks:
- in_denmark: firms founded in Denmark
- abroad_danish_founders: firms founded abroad by Danish founders

Rules:
- discovery is for cheap lead generation, not final validation
- prioritize obscure or long-tail firms over famous repeated examples
- exclude every known firm provided by the user
- return only firms founded in 1999 or later
- do not return firms clearly founded before 1999
- if founding year is unknown, keep only firms whose source-backed evidence strongly suggests a young firm or startup founded in 1999 or later
- return only source-backed leads
- keep short_reason to 1-2 short sentences
- return only 1-3 source URLs
- do not fabricate URLs, dates, locations, founder nationality, or certainty
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
    for value in CROSS_BUCKETS:
        buckets.append(DiscoveryBucket(bucket_type="cross_bucket", bucket_value=value))
    return buckets


def load_prompt_template(origin_track: str = "in_denmark", path: Path | None = None) -> str:
    chosen_path = path or DEFAULT_PROMPT_PATHS[origin_track]
    if chosen_path.exists():
        return chosen_path.read_text(encoding="utf-8").strip()
    if origin_track == "in_denmark":
        return (
            "Find additional Danish-founded firms that may plausibly show a later foreign HQ, principal-office, parent, "
            "leadership, or operations signal. Exclude all known firms listed below. Prioritize obscure or less-known firms. "
            "Return only short, source-backed leads founded in 1999 or later. Search beyond literal headquarters-move wording "
            "and pay attention to principal executive offices, global or group HQ language, re-domiciliation, foreign parents, "
            "leadership relocation, strategically important foreign offices, and former-name or legal-entity aliases."
        )
    return (
        "Find additional firms founded abroad by Danish founders. Exclude all known firms listed below. Prioritize obscure "
        "or less-known firms. Return only short, source-backed leads founded in 1999 or later. Look for clear founder links "
        "to Denmark together with firm founding, incorporation, or early operations outside Denmark."
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
                        "possible_abroad_location",
                        "possible_move_year",
                        "signal_type",
                        "signal_strength",
                        "short_reason",
                        "source_urls",
                    ],
                    "properties": {
                        "firm_name": {"type": "string"},
                        "discovery_bucket": {"type": "string"},
                        "sector_if_known": {"type": ["string", "null"]},
                        "possible_founding_location": {"type": ["string", "null"]},
                        "possible_founding_year": {"type": ["integer", "null"]},
                        "possible_abroad_location": {"type": ["string", "null"]},
                        "possible_move_year": {"type": ["integer", "null"]},
                        "signal_type": {"type": "string", "enum": ALLOWED_SIGNAL_TYPES},
                        "signal_strength": {"type": "string", "enum": ALLOWED_SIGNAL_STRENGTHS},
                        "short_reason": {"type": "string"},
                        "source_urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 3,
                        },
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
        "bucket_guidance": {
            "instruction": "For cross_bucket prompts, satisfy all parts of the combined bucket together rather than treating them as separate alternatives."
        },
        "research_target": {
            "target_case": "firms founded in Denmark that may later have moved their own HQ or main operations abroad",
            "non_target_case": "abroad (Danish founders) without the firm itself being founded or based in Denmark",
            "founding_year_rule": "founding_year >= 1999; unknown founding year is allowed only with strong evidence of a 1999-or-later startup or young firm",
            "relocation_mechanisms_to_search": [
                "principal executive offices moved abroad",
                "global or group headquarters moved abroad",
                "IPO or listing-related re-domiciliation with executive relocation",
                "creation of a foreign parent or holding company after Danish founding",
                "acquisition-linked relocation where the operating company still exists abroad",
                "founder or executive relocation narrative tied to the firm's HQ",
                "strategically important foreign office or operations base that may have become the main office",
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
                "cheap, source-backed leads",
                "cases outside the usual famous examples",
                "Danish-language and foreign-language reporting when relevant",
                "city-pair and destination-specific relocation narratives",
            ],
            "avoid": [
                "any known firm in the exclusion list",
                "pure acquisition-only cases without continued operating-company evidence abroad",
                "ordinary foreign office openings alone",
                "Danish-founder-abroad cases unless the company itself was founded or based in Denmark",
                "firms clearly founded before 1999",
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
            "lead_threshold": "at least one real source for Danish origin and at least one real source for a later plausible foreign signal",
        },
        "signal_definitions": {
            "allowed_signal_type_values": ALLOWED_SIGNAL_TYPES,
            "allowed_signal_strength_values": ALLOWED_SIGNAL_STRENGTHS,
            "strong": "explicit HQ/headquarters/principal executive office/global HQ abroad",
            "medium": "foreign parent, re-domiciliation, IPO/listing migration, or main office abroad without fully proven physical HQ move",
            "weak": "acquisition, foreign office, leadership abroad, or operational expansion abroad only; plausible lead but not enough for final validation",
        },
        "output_fields": [
            "firm_name",
            "discovery_bucket",
            "sector_if_known",
            "possible_founding_location",
            "possible_founding_year",
            "possible_abroad_location",
            "possible_move_year",
            "signal_type",
            "signal_strength",
            "short_reason",
            "source_urls",
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
    followup_round: int,
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
            "founding_year_rule": "founding_year >= 1999; unknown founding year is allowed only with strong evidence of a 1999-or-later startup or young firm",
            "followup_instruction": f"Return only additional firms not already found in earlier passes for this bucket. This is follow-up round {followup_round}.",
            "core_known_cases_already_covered": core_relocation_names,
        },
        "requirements": {
            "prioritize": [
                "additional obscure firms",
                "long-tail firms not already named",
                "cheap, source-backed leads",
                "cases outside the usual famous examples",
            ],
            "avoid": [
                "any known firm in the exclusion list",
                "any firm already found in the first pass for this bucket",
                "pure acquisition-only cases without continued operating-company evidence abroad",
                "ordinary foreign office openings alone",
                "Danish-founder-abroad cases unless the company itself was founded or based in Denmark",
                "firms clearly founded before 1999",
            ],
            "return_only_candidates_with_real_sources": True,
        },
        "signal_definitions": {
            "allowed_signal_type_values": ALLOWED_SIGNAL_TYPES,
            "allowed_signal_strength_values": ALLOWED_SIGNAL_STRENGTHS,
        },
        "output_fields": [
            "firm_name",
            "discovery_bucket",
            "sector_if_known",
            "possible_founding_location",
            "possible_founding_year",
            "possible_abroad_location",
            "possible_move_year",
            "signal_type",
            "signal_strength",
            "short_reason",
            "source_urls",
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


def parse_discovery_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in payload.get("candidates") or []:
        founding_year = candidate.get("possible_founding_year")
        signal_type = _normalize_signal_type(candidate.get("signal_type"))
        signal_strength = _normalize_signal_strength(candidate.get("signal_strength"))
        record = {
            "firm_name": str(candidate.get("firm_name") or "").strip(),
            "discovery_bucket": str(candidate.get("discovery_bucket") or "").strip(),
            "sector_if_known": candidate.get("sector_if_known"),
            "possible_founding_location": candidate.get("possible_founding_location"),
            "possible_founding_year": founding_year,
            "possible_abroad_location": candidate.get("possible_abroad_location"),
            "possible_move_year": candidate.get("possible_move_year"),
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "short_reason": str(candidate.get("short_reason") or "").strip(),
            "source_urls": [str(url).strip() for url in candidate.get("source_urls") or [] if str(url).strip()][:3],
        }
        if is_post_1998_eligible_or_supported(
            founding_year,
            record["short_reason"],
            record["possible_founding_location"],
            record["possible_abroad_location"],
            record["signal_type"],
            record["signal_strength"],
        ):
            records.append(record)
    return records


def _normalize_signal_type(value: Any) -> str:
    text = str(value or "").strip()
    if text in ALLOWED_SIGNAL_TYPES:
        return text
    return "other_plausible_signal"


def _normalize_signal_strength(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ALLOWED_SIGNAL_STRENGTHS:
        return text
    return "weak"


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
    prompt_path: Path | None = None,
    followup_rounds: int = 3,
    memory_path: Path = DEFAULT_DISCOVERY_MEMORY_PATH,
) -> list[dict[str, Any]]:
    known_firms = load_seed_firms(known_path)
    prompt_seed_firms = select_discovery_prompt_firms(known_path, known_firms)
    seed_exclusion_names = build_exclusion_list(prompt_seed_firms)
    core_relocation_names = build_core_relocation_names(prompt_seed_firms)
    prompt_template = load_prompt_template(path=prompt_path)
    buckets = build_discovery_buckets()
    if max_buckets is not None:
        buckets = buckets[:max_buckets]

    all_candidate_records: list[dict[str, Any]] = []
    bucket_run_records: list[dict[str, Any]] = []
    cost_records: list[dict[str, Any]] = []
    memory_known_normalized = load_known_firms(memory_path)
    rolling_exclusion_names = list(seed_exclusion_names)
    seen_prompt_names = {normalize_company_name(name) for name in rolling_exclusion_names if normalize_company_name(name)}
    for record in update_known_firms_from_rows([], memory_path=memory_path):
        normalized = str(record.get("normalized_name") or "").strip()
        display_name = str(record.get("firm_name") or "").strip()
        if normalized and normalized in memory_known_normalized and display_name and normalized not in seen_prompt_names:
            rolling_exclusion_names.append(display_name)
            seen_prompt_names.add(normalized)

    for position, bucket in enumerate(buckets, start=1):
        print(f"[{position}/{len(buckets)}] bucket={bucket.bucket_id}")
        prompt_text = build_discovery_prompt(
            bucket=bucket,
            exclusion_names=rolling_exclusion_names,
            core_relocation_names=core_relocation_names,
            prompt_template=prompt_template,
        )
        print(f"  - pass=initial query={bucket.bucket_type}:{bucket.bucket_value}")
        parsed_payload, raw_response = call_openai_snowball_discovery(user_prompt=prompt_text, model=model)
        parsed_candidates = parse_discovery_candidates(parsed_payload)
        print(f"  - pass=initial leads={len(parsed_candidates)} bucket={bucket.bucket_id}")
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
        cost_records.append(
            build_cost_record(
                stage="discovery",
                request_kind="initial",
                raw_response=raw_response,
                requested_model=model,
                metadata={
                    "round": round_number,
                    "bucket_type": bucket.bucket_type,
                    "bucket_value": bucket.bucket_value,
                    "discovery_bucket": bucket.bucket_id,
                },
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

        already_found_names = list(
            dict.fromkeys(candidate["firm_name"] for candidate in parsed_candidates if candidate["firm_name"])
        )
        for followup_round in range(1, followup_rounds + 1):
            if not already_found_names:
                break
            followup_prompt_text = build_followup_discovery_prompt(
                bucket=bucket,
                exclusion_names=rolling_exclusion_names,
                core_relocation_names=core_relocation_names,
                prompt_template=prompt_template,
                already_found_names=already_found_names,
                followup_round=followup_round,
            )
            print(f"  - pass=followup_{followup_round} query={bucket.bucket_type}:{bucket.bucket_value}")
            followup_payload, followup_response = call_openai_snowball_discovery(
                user_prompt=followup_prompt_text,
                model=model,
            )
            followup_candidates = parse_discovery_candidates(followup_payload)
            print(f"  - pass=followup_{followup_round} leads={len(followup_candidates)} bucket={bucket.bucket_id}")
            bucket_run_records.append(
                build_bucket_run_record(
                    bucket=bucket,
                    round_number=round_number,
                    model=model,
                    prompt_text=followup_prompt_text,
                    raw_response=followup_response,
                    parsed_payload=followup_payload,
                    pass_type=f"followup_{followup_round}",
                )
            )
            cost_records.append(
                build_cost_record(
                    stage="discovery",
                    request_kind=f"followup_{followup_round}",
                    raw_response=followup_response,
                    requested_model=model,
                    metadata={
                        "round": round_number,
                        "bucket_type": bucket.bucket_type,
                        "bucket_value": bucket.bucket_value,
                        "discovery_bucket": bucket.bucket_id,
                    },
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
            new_names = [candidate["firm_name"] for candidate in followup_candidates if candidate["firm_name"]]
            for name in new_names:
                if name not in already_found_names:
                    already_found_names.append(name)
            if not new_names:
                break

        for candidate in parsed_candidates:
            normalized = normalize_company_name(candidate.get("firm_name"))
            display_name = str(candidate.get("firm_name") or "").strip()
            if normalized and display_name and normalized not in seen_prompt_names:
                rolling_exclusion_names.append(display_name)
                seen_prompt_names.add(normalized)
        for candidate in all_candidate_records:
            normalized = normalize_company_name(candidate.get("firm_name"))
            display_name = str(candidate.get("firm_name") or "").strip()
            if normalized and display_name and normalized not in seen_prompt_names:
                rolling_exclusion_names.append(display_name)
                seen_prompt_names.add(normalized)

    save_jsonl(all_candidate_records, output_path)
    bucket_runs_path = output_path.with_name(f"{output_path.stem}_bucket_runs{output_path.suffix}")
    save_jsonl(bucket_run_records, bucket_runs_path)
    save_jsonl(cost_records, cost_log_path(output_path))
    updated_memory = update_known_firms_from_rows(all_candidate_records, memory_path=memory_path)
    save_known_firms(updated_memory, memory_path)
    totals = sum_cost_records(cost_records)
    print(f"estimated_cost_usd={totals['estimated_cost_usd']:.6f}")
    print(f"api_cost_log_path={cost_log_path(output_path)}")
    print(f"discovery_memory_path={memory_path}")
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
        default=None,
        help="Optional custom prompt template path.",
    )
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=DEFAULT_DISCOVERY_MEMORY_PATH,
        help="Persistent cross-run discovery memory path used for prompt exclusions and cumulative tracking.",
    )
    parser.add_argument(
        "--followup-rounds",
        type=int,
        default=3,
        help="Number of bounded follow-up 'find more' rounds to run per bucket after the initial pass.",
    )
    parser.add_argument(
        "--no-followup-pass",
        action="store_const",
        const=0,
        dest="followup_rounds",
        help="Disable follow-up discovery passes. Deprecated alias for --followup-rounds 0.",
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
        followup_rounds=max(0, args.followup_rounds),
        memory_path=args.memory_path,
    )


if __name__ == "__main__":
    main()
