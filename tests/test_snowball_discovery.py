import csv
import json
from pathlib import Path

from snowball_discovery import (
    CITY_PAIR_BUCKETS,
    CROSS_BUCKETS,
    DiscoveryBucket,
    MECHANISM_BUCKETS,
    SOURCE_STYLE_BUCKETS,
    build_discovery_buckets,
    build_discovery_prompt,
    build_followup_discovery_prompt,
    parse_discovery_candidates,
    run_snowball_discovery,
)
from src.seed_list import build_core_relocation_names, build_exclusion_list, load_seed_firms, select_discovery_prompt_firms


def write_known_csv(path: Path) -> None:
    rows = [
        {
            "name": "Zendesk",
            "founding_origin": "in Denmark",
            "industry": "software",
        },
        {
            "name": "Unity",
            "founding_origin": "abroad (Danish founders)",
            "industry": "gaming",
        },
        {
            "name": "Endomondo ApS",
            "founding_origin": "in Denmark",
            "industry": "fitness",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "founding_origin", "industry"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_known_seed_csv_recognizes_both_founding_origin_categories(tmp_path: Path) -> None:
    path = tmp_path / "known.csv"
    write_known_csv(path)

    firms = load_seed_firms(path)

    assert [firm.name for firm in firms] == ["Zendesk", "Unity", "Endomondo ApS"]
    assert {firm.founding_origin for firm in firms} == {"in Denmark", "abroad (Danish founders)"}


def test_exclusion_list_contains_all_known_firm_names(tmp_path: Path) -> None:
    path = tmp_path / "known.csv"
    write_known_csv(path)

    firms = load_seed_firms(path)
    exclusions = build_exclusion_list(firms)

    assert exclusions == ["Endomondo ApS", "Unity", "Zendesk"]


def test_core_relocation_list_contains_only_in_denmark_names(tmp_path: Path) -> None:
    path = tmp_path / "known.csv"
    write_known_csv(path)

    firms = load_seed_firms(path)
    core_names = build_core_relocation_names(firms)

    assert core_names == ["Endomondo ApS", "Zendesk"]
    assert "Unity" not in core_names


def test_prompt_contains_exclusion_names(tmp_path: Path) -> None:
    path = tmp_path / "known.csv"
    write_known_csv(path)
    firms = load_seed_firms(path)

    prompt = build_discovery_prompt(
        bucket=DiscoveryBucket(bucket_type="sector", bucket_value="biotech"),
        exclusion_names=build_exclusion_list(firms),
        core_relocation_names=build_core_relocation_names(firms),
        prompt_template="Find additional firms.",
    )

    payload = json.loads(prompt)
    assert payload["known_firm_exclusions"] == ["Endomondo ApS", "Unity", "Zendesk"]
    assert payload["research_target"]["core_known_cases_already_covered"] == ["Endomondo ApS", "Zendesk"]
    assert "relocation_mechanisms_to_search" in payload["research_target"]
    assert "name_handling" in payload["research_target"]
    assert "prefer_sources" in payload["requirements"]
    assert payload["signal_definitions"]["allowed_signal_strength_values"] == ["strong", "medium", "weak"]
    assert "bucket_guidance" in payload


def test_build_discovery_buckets_includes_new_bucket_families() -> None:
    buckets = build_discovery_buckets()

    bucket_ids = {bucket.bucket_id for bucket in buckets}
    assert f"mechanism:{MECHANISM_BUCKETS[0]}" in bucket_ids
    assert f"source_style:{SOURCE_STYLE_BUCKETS[0]}" in bucket_ids
    assert f"city_pair:{CITY_PAIR_BUCKETS[0]}" in bucket_ids
    assert f"cross_bucket:{CROSS_BUCKETS[0]}" in bucket_ids


def test_followup_prompt_requests_additional_firms(tmp_path: Path) -> None:
    path = tmp_path / "known.csv"
    write_known_csv(path)
    firms = load_seed_firms(path)

    prompt = build_followup_discovery_prompt(
        bucket=DiscoveryBucket(bucket_type="sector", bucket_value="biotech"),
        exclusion_names=build_exclusion_list(firms),
        core_relocation_names=build_core_relocation_names(firms),
        prompt_template="Find additional firms.",
        already_found_names=["Allarity", "IO Biotech"],
        followup_round=2,
    )

    payload = json.loads(prompt)
    assert payload["already_found_firms_this_bucket"] == ["Allarity", "IO Biotech"]
    assert "Find additional firms" in payload["task"]
    assert "followup_instruction" in payload["research_target"]
    assert "follow-up round 2" in payload["research_target"]["followup_instruction"]


def test_discovery_prompt_exclusions_use_borsen_subset_for_29_04_file(tmp_path: Path) -> None:
    path = tmp_path / "preliminary_data_29_04.csv"
    rows = [
        {
            "name": "APM Terminals",
            "founding_origin": "in Denmark",
            "industry": "Transportation",
            "founded": "2001",
            "moved": "2004",
            "latest_hq_city": "The Hague",
            "latest_hq_country": "NL",
            "today_hq_city": "",
            "today_hq_country": "",
            "status_today": "Active",
            "employment_total": "",
            "employment_dk": "6",
            "acquiror": "",
            "acquiror_country": "",
            "deal_value_th_usd": "",
            "deal_year": "",
            "method": "Børsen",
        },
        {
            "name": "Cobalt",
            "founding_origin": "abroad (Danish founders)",
            "industry": "Tech",
            "founded": "2013",
            "moved": "",
            "latest_hq_city": "Buenos Aires",
            "latest_hq_country": "AR",
            "today_hq_city": "San Francisco",
            "today_hq_country": "US",
            "status_today": "Active",
            "employment_total": "548",
            "employment_dk": "0",
            "acquiror": "",
            "acquiror_country": "",
            "deal_value_th_usd": "",
            "deal_year": "",
            "method": "Børsen",
        },
        {
            "name": "Manual Firm",
            "founding_origin": "in Denmark",
            "industry": "Tech",
            "founded": "2015",
            "moved": "",
            "latest_hq_city": "",
            "latest_hq_country": "",
            "today_hq_city": "",
            "today_hq_country": "",
            "status_today": "Active",
            "employment_total": "",
            "employment_dk": "",
            "acquiror": "",
            "acquiror_country": "",
            "deal_value_th_usd": "",
            "deal_year": "",
            "method": "manual",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    firms = load_seed_firms(path)
    prompt_firms = select_discovery_prompt_firms(path, firms)
    prompt = build_discovery_prompt(
        bucket=DiscoveryBucket(bucket_type="sector", bucket_value="biotech"),
        exclusion_names=build_exclusion_list(prompt_firms),
        core_relocation_names=build_core_relocation_names(prompt_firms),
        prompt_template="Find additional firms.",
    )

    payload = json.loads(prompt)
    assert payload["known_firm_exclusions"] == ["APM Terminals", "Cobalt"]
    assert payload["research_target"]["core_known_cases_already_covered"] == ["APM Terminals"]


def test_parser_handles_empty_candidate_list() -> None:
    assert parse_discovery_candidates({"candidates": []}) == []


def test_parse_discovery_candidates_excludes_firms_founded_in_1998() -> None:
    payload = {
        "candidates": [
            {
                "firm_name": "OldCo",
                "discovery_bucket": "sector:biotech",
                "sector_if_known": "biotech",
                "possible_founding_location": "Copenhagen",
                "possible_founding_year": 1998,
                "possible_abroad_location": "Boston",
                "possible_move_year": 2010,
                "signal_type": "explicit_hq_move",
                "signal_strength": "strong",
                "short_reason": "Founded in Copenhagen in 1998 and later based in Boston.",
                "source_urls": ["https://example.com/oldco"],
            }
        ]
    }

    assert parse_discovery_candidates(payload) == []


def test_parse_discovery_candidates_keeps_firms_founded_in_1999() -> None:
    payload = {
        "candidates": [
            {
                "firm_name": "NewCo",
                "discovery_bucket": "sector:biotech",
                "sector_if_known": "biotech",
                "possible_founding_location": "Copenhagen",
                "possible_founding_year": 1999,
                "possible_abroad_location": "Boston",
                "possible_move_year": 2010,
                "signal_type": "explicit_hq_move",
                "signal_strength": "strong",
                "short_reason": "Founded in Copenhagen in 1999 and later based in Boston.",
                "source_urls": ["https://example.com/newco"],
            }
        ]
    }

    parsed = parse_discovery_candidates(payload)
    assert len(parsed) == 1
    assert parsed[0]["firm_name"] == "NewCo"


def test_parse_discovery_candidates_drops_unknown_year_without_young_firm_evidence() -> None:
    payload = {
        "candidates": [
            {
                "firm_name": "UnknownCo",
                "discovery_bucket": "sector:biotech",
                "sector_if_known": "biotech",
                "possible_founding_location": "Copenhagen",
                "possible_founding_year": None,
                "possible_abroad_location": "Boston",
                "possible_move_year": 2010,
                "signal_type": "foreign_office_or_operations",
                "signal_strength": "weak",
                "short_reason": "A Danish company with later Boston operations.",
                "source_urls": ["https://example.com/unknownco"],
            }
        ]
    }

    assert parse_discovery_candidates(payload) == []


def test_parse_discovery_candidates_allows_weak_leads() -> None:
    payload = {
        "candidates": [
            {
                "firm_name": "YoungLeadCo",
                "discovery_bucket": "destination:UK",
                "sector_if_known": "design",
                "possible_founding_location": "Aarhus",
                "possible_founding_year": 2004,
                "possible_abroad_location": "London",
                "possible_move_year": None,
                "signal_type": "leadership_abroad",
                "signal_strength": "weak",
                "short_reason": "Founded in Aarhus in 2004; later sources place leadership in London.",
                "source_urls": [
                    "https://example.com/1",
                    "https://example.com/2",
                    "https://example.com/3",
                    "https://example.com/4",
                ],
            }
        ]
    }

    parsed = parse_discovery_candidates(payload)
    assert len(parsed) == 1
    assert parsed[0]["signal_strength"] == "weak"
    assert parsed[0]["signal_type"] == "leadership_abroad"
    assert parsed[0]["source_urls"] == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]


def test_run_snowball_discovery_excludes_memory_and_prior_bucket_firms(tmp_path: Path, monkeypatch) -> None:
    known_path = tmp_path / "preliminary_data_29_04.csv"
    write_known_csv(known_path)
    output_path = tmp_path / "data" / "discovery" / "test.jsonl"
    memory_path = tmp_path / "data" / "memory" / "known.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        json.dumps(
            {
                "firm_name": "MemoryCo",
                "normalized_name": "memoryco",
                "first_seen_at": "2026-04-29T00:00:00Z",
                "last_seen_at": "2026-04-29T00:00:00Z",
                "times_seen": 1,
                "source_urls": [],
                "query_ids": [],
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "snowball_discovery.build_discovery_buckets",
        lambda: [
            DiscoveryBucket(bucket_type="sector", bucket_value="biotech"),
            DiscoveryBucket(bucket_type="sector", bucket_value="design"),
        ],
    )

    seen_prompts: list[dict[str, object]] = []

    def fake_call_openai_snowball_discovery(user_prompt: str, model: str):
        payload = json.loads(user_prompt)
        seen_prompts.append(payload)
        bucket_id = payload["bucket"]["discovery_bucket"]
        if bucket_id == "sector:biotech":
            return (
                {
                    "candidates": [
                        {
                            "firm_name": "BucketOneCo",
                            "discovery_bucket": bucket_id,
                            "sector_if_known": "biotech",
                            "possible_founding_location": "Copenhagen",
                            "possible_founding_year": 2005,
                            "possible_abroad_location": "Boston",
                            "possible_move_year": 2015,
                            "signal_type": "explicit_hq_move",
                            "signal_strength": "strong",
                            "short_reason": "Founded in Copenhagen in 2005 and later based in Boston.",
                            "source_urls": ["https://example.com/1"],
                        }
                    ]
                },
                {"id": f"resp-{len(seen_prompts)}", "model": "gpt-5-mini-2025-08-07", "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1}, "output": []},
            )
        return (
            {"candidates": []},
            {"id": f"resp-{len(seen_prompts)}", "model": "gpt-5-mini-2025-08-07", "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1}, "output": []},
        )

    monkeypatch.setattr("snowball_discovery.call_openai_snowball_discovery", fake_call_openai_snowball_discovery)

    run_snowball_discovery(
        known_path=known_path,
        output_path=output_path,
        round_number=1,
        model="gpt-5-mini",
        max_buckets=2,
        followup_rounds=0,
        memory_path=memory_path,
    )

    assert "MemoryCo" in seen_prompts[0]["known_firm_exclusions"]
    assert "BucketOneCo" in seen_prompts[1]["known_firm_exclusions"]
