import csv
import json
from pathlib import Path

from discovery_memory import (
    build_exclusion_prompt,
    create_followup_queries,
    load_known_firms,
    save_known_firms,
    update_known_firms_from_rows,
    update_known_firms_from_model1,
)
from retrieve_with_openai import load_queries


def test_load_known_firms_missing_file_returns_empty_set(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    assert load_known_firms(missing_path) == set()


def test_update_known_firms_from_model1_merges_duplicates(tmp_path: Path) -> None:
    model1_path = tmp_path / "model1.jsonl"
    rows = [
        {
            "firm_name": "Zendesk",
            "sources": ["https://example.com/zendesk-1"],
            "query_ids": ["q_001"],
        },
        {
            "firm_name": "Zendesk, Inc.",
            "sources": ["https://example.com/zendesk-2"],
            "query_ids": ["q_002"],
        },
        {
            "firm_name": "Coana",
            "sources": ["https://example.com/coana"],
            "query_ids": ["q_003"],
        },
    ]
    with model1_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    records = update_known_firms_from_model1(
        model1_path=model1_path,
        seen_at="2026-04-27T12:00:00Z",
    )

    assert len(records) == 2
    zendesk = next(record for record in records if record["normalized_name"] == "zendesk")
    assert zendesk["firm_name"] == "Zendesk"
    assert zendesk["times_seen"] == 2
    assert zendesk["first_seen_at"] == "2026-04-27T12:00:00Z"
    assert zendesk["last_seen_at"] == "2026-04-27T12:00:00Z"
    assert zendesk["source_urls"] == ["https://example.com/zendesk-1", "https://example.com/zendesk-2"]
    assert zendesk["query_ids"] == ["q_001", "q_002"]


def test_save_known_firms_and_load_known_firms_csv(tmp_path: Path) -> None:
    memory_path = tmp_path / "known_firms.csv"
    records = [
        {
            "firm_name": "Unity",
            "normalized_name": "unity",
            "first_seen_at": "2026-04-27T12:00:00Z",
            "last_seen_at": "2026-04-27T12:00:00Z",
            "times_seen": 3,
            "source_urls": ["https://example.com/unity"],
            "query_ids": ["q_004"],
            "notes": "",
        }
    ]

    save_known_firms(records, memory_path)

    with memory_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["firm_name"] == "Unity"
    assert load_known_firms(memory_path) == {"unity"}


def test_followup_queries_include_exclusion_wording_and_are_loadable(tmp_path: Path) -> None:
    memory_path = tmp_path / "known_firms.csv"
    output_path = tmp_path / "queries_followup_round_001.jsonl"
    save_known_firms(
        [
            {
                "firm_name": "Zendesk",
                "normalized_name": "zendesk",
                "first_seen_at": "2026-04-27T12:00:00Z",
                "last_seen_at": "2026-04-27T12:00:00Z",
                "times_seen": 5,
                "source_urls": ["https://example.com/zendesk"],
                "query_ids": ["q_001"],
                "notes": "",
            },
            {
                "firm_name": "Unity",
                "normalized_name": "unity",
                "first_seen_at": "2026-04-27T12:00:00Z",
                "last_seen_at": "2026-04-27T12:00:00Z",
                "times_seen": 4,
                "source_urls": ["https://example.com/unity"],
                "query_ids": ["q_002"],
                "notes": "",
            },
        ],
        memory_path,
    )

    queries = create_followup_queries(memory_path=memory_path, round_number=1)
    save_rows = [
        {
            "query_id": query.query_id,
            "query_text": query.query_text,
            "language": query.language,
            "family": query.family,
            "created_at": query.created_at,
        }
        for query in queries
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in save_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    assert any("Do not return these already-known firms: Zendesk, Unity." in query.query_text for query in queries)
    assert any(query.family == "followup_destination_country" for query in queries)
    assert any(query.family == "followup_mechanism" for query in queries)
    assert any(query.family == "followup_city_pair" for query in queries)
    assert any("Børsen" in query.query_text for query in queries)
    assert any("less-known firms" in query.query_text for query in queries)
    assert any("principal executive" in query.query_text.lower() or "globalt" in query.query_text.lower() for query in queries)

    loaded_queries = load_queries(output_path)
    assert len(loaded_queries) == len(queries)
    assert loaded_queries[0].query_id.startswith("q_followup_r001_")


def test_build_exclusion_prompt_caps_list_length() -> None:
    records = [{"firm_name": f"Firm {index}"} for index in range(150)]

    prompt = build_exclusion_prompt(records, limit=100)

    assert prompt.startswith("Do not return these already-known firms:")
    assert "Firm 99" in prompt
    assert "Firm 100" not in prompt


def test_update_known_firms_from_rows_uses_source_urls(tmp_path: Path) -> None:
    memory_path = tmp_path / "known_firms.jsonl"
    records = update_known_firms_from_rows(
        [
            {"firm_name": "LessKnownCo", "source_urls": ["https://example.com/1"]},
            {"firm_name": "LessKnownCo ApS", "source_urls": ["https://example.com/2"]},
        ],
        memory_path=memory_path,
        seen_at="2026-04-29T12:00:00Z",
    )

    assert len(records) == 1
    assert records[0]["normalized_name"] == "lessknownco"
    assert records[0]["times_seen"] == 2
    assert records[0]["source_urls"] == ["https://example.com/1", "https://example.com/2"]
