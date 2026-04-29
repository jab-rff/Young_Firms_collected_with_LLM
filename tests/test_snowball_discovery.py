import csv
import json
from pathlib import Path

from snowball_discovery import (
    CITY_PAIR_BUCKETS,
    DiscoveryBucket,
    MECHANISM_BUCKETS,
    SOURCE_STYLE_BUCKETS,
    build_discovery_buckets,
    build_discovery_prompt,
    build_followup_discovery_prompt,
    parse_discovery_candidates,
)
from src.seed_list import build_core_relocation_names, build_exclusion_list, load_seed_firms


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


def test_build_discovery_buckets_includes_new_bucket_families() -> None:
    buckets = build_discovery_buckets()

    bucket_ids = {bucket.bucket_id for bucket in buckets}
    assert f"mechanism:{MECHANISM_BUCKETS[0]}" in bucket_ids
    assert f"source_style:{SOURCE_STYLE_BUCKETS[0]}" in bucket_ids
    assert f"city_pair:{CITY_PAIR_BUCKETS[0]}" in bucket_ids


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
    )

    payload = json.loads(prompt)
    assert payload["already_found_firms_this_bucket"] == ["Allarity", "IO Biotech"]
    assert "Find additional firms" in payload["task"]
    assert "followup_instruction" in payload["research_target"]


def test_parser_handles_empty_candidate_list() -> None:
    assert parse_discovery_candidates({"candidates": []}) == []


def test_parse_discovery_candidates_excludes_firms_founded_in_1999() -> None:
    payload = {
        "candidates": [
            {
                "firm_name": "OldCo",
                "discovery_bucket": "sector:biotech",
                "sector_if_known": "biotech",
                "possible_founding_location": "Copenhagen",
                "possible_founding_year": 1999,
                "possible_abroad_hq_location": "Boston",
                "possible_move_year": 2010,
                "why_candidate": "Possible relocation story.",
                "founding_denmark_evidence": "Founded in Copenhagen in 1999.",
                "abroad_hq_evidence": "Later based in Boston.",
                "ma_context": None,
                "source_urls": ["https://example.com/oldco"],
                "uncertainty_note": "",
            }
        ]
    }

    assert parse_discovery_candidates(payload) == []


def test_parse_discovery_candidates_keeps_firms_founded_in_2000() -> None:
    payload = {
        "candidates": [
            {
                "firm_name": "NewCo",
                "discovery_bucket": "sector:biotech",
                "sector_if_known": "biotech",
                "possible_founding_location": "Copenhagen",
                "possible_founding_year": 2000,
                "possible_abroad_hq_location": "Boston",
                "possible_move_year": 2010,
                "why_candidate": "Possible relocation story.",
                "founding_denmark_evidence": "Founded in Copenhagen in 2000.",
                "abroad_hq_evidence": "Later based in Boston.",
                "ma_context": None,
                "source_urls": ["https://example.com/newco"],
                "uncertainty_note": "",
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
                "possible_abroad_hq_location": "Boston",
                "possible_move_year": 2010,
                "why_candidate": "Possible relocation story.",
                "founding_denmark_evidence": "A Danish company with later Boston operations.",
                "abroad_hq_evidence": "Later based in Boston.",
                "ma_context": None,
                "source_urls": ["https://example.com/unknownco"],
                "uncertainty_note": "",
            }
        ]
    }

    assert parse_discovery_candidates(payload) == []
