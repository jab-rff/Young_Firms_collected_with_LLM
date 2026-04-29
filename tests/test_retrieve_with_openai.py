import json
from pathlib import Path

import pytest

from retrieve_with_openai import build_retrieval_prompt, load_queries, parse_retrieved_items, save_retrieved_items
from src.data_models import Query, RetrievedItem


def test_load_queries_reads_utf8_jsonl(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    rows = [
        {
            "query_id": "q_001",
            "query_text": "danish startup moved headquarters",
            "language": "en",
            "family": "seed",
            "created_at": "2026-04-27T10:00:00Z",
        },
        {
            "query_id": "q_002",
            "query_text": "dansk virksomhed hovedkontor udlandet",
            "language": "da",
            "family": "exploratory",
            "created_at": "2026-04-27T10:01:00Z",
        },
    ]
    with queries_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    queries = load_queries(queries_path)

    assert queries == [
        Query(
            query_id="q_001",
            query_text="danish startup moved headquarters",
            language="en",
            family="seed",
            created_at="2026-04-27T10:00:00Z",
        ),
        Query(
            query_id="q_002",
            query_text="dansk virksomhed hovedkontor udlandet",
            language="da",
            family="exploratory",
            created_at="2026-04-27T10:01:00Z",
        ),
    ]


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_load_queries_reads_windows_jsonl_encodings(tmp_path: Path, encoding: str) -> None:
    queries_path = tmp_path / f"queries_{encoding}.jsonl"
    rows = [
        {
            "query_id": "q_003",
            "query_text": "founded in Denmark now headquartered in",
            "language": "en",
            "family": "implicit_foreign_hq",
            "created_at": "2026-04-27T10:02:00Z",
        }
    ]
    with queries_path.open("w", encoding=encoding) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    queries = load_queries(queries_path)

    assert queries == [
        Query(
            query_id="q_003",
            query_text="founded in Denmark now headquartered in",
            language="en",
            family="implicit_foreign_hq",
            created_at="2026-04-27T10:02:00Z",
        )
    ]


def test_load_queries_raises_clear_error_for_unsupported_encoding(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries_bad.jsonl"
    queries_path.write_bytes(b"\x81\x82\x83\x84")

    with pytest.raises(RuntimeError, match="Could not decode query file"):
        load_queries(queries_path)


def test_parse_retrieved_items_preserves_query_provenance() -> None:
    query = Query(
        query_id="q_003",
        query_text="sitecore denmark headquarters abroad",
        language="en",
        family="seed",
        created_at="2026-04-27T10:02:00Z",
    )
    payload = {
        "results": [
            {
                "title": "Sitecore opens new U.S. headquarters",
                "snippet": "Sitecore, founded in Denmark, expanded its executive presence in the United States.",
                "url": "https://example.com/sitecore-hq",
                "source_name": "Example News",
                "language": "en",
                "raw_text": "Sitecore was founded in Denmark and later described its U.S. headquarters strategy.",
                "focal_firm_names": ["Sitecore"],
                "non_focal_entities": ["United States"],
                "evidence_note": "The source links Sitecore to Denmark and later U.S. headquarters activity.",
            },
            {
                "title": "Podio company history",
                "snippet": "Podio later moved headquarters abroad.",
                "url": "https://example.com/podio-history",
                "source_name": "",
                "language": "",
                "raw_text": "",
                "focal_firm_names": ["Podio"],
                "non_focal_entities": [],
                "evidence_note": "The source suggests Podio later moved its headquarters abroad.",
            },
            {
                "title": "Duplicate Podio page",
                "snippet": "Duplicate source should be skipped.",
                "url": "https://example.com/podio-history",
                "source_name": "Duplicate Source",
                "language": "en",
                "raw_text": "Duplicate row.",
                "focal_firm_names": ["Podio"],
                "non_focal_entities": ["Duplicate Source"],
                "evidence_note": "Duplicate source.",
            },
            {
                "title": "Missing URL row",
                "snippet": "This should be dropped.",
                "url": "",
                "source_name": "No URL Source",
                "language": "en",
                "raw_text": "No URL available.",
                "focal_firm_names": ["Unknown"],
                "non_focal_entities": [],
                "evidence_note": "No usable source URL.",
            },
        ]
    }

    items = parse_retrieved_items(
        query=query,
        payload=payload,
        retrieved_at="2026-04-27T10:03:00Z",
    )

    assert len(items) == 2
    assert items[0].query_id == "q_003"
    assert items[0].query_text == "sitecore denmark headquarters abroad"
    assert items[0].source_name == "Example News"
    assert items[0].title == "Sitecore opens new U.S. headquarters"
    assert items[0].url == "https://example.com/sitecore-hq"
    assert items[0].language == "en"
    assert items[0].retrieved_at == "2026-04-27T10:03:00Z"
    assert "Sitecore" in items[0].raw_text
    assert "United States" not in items[0].raw_text

    assert items[1].query_id == "q_003"
    assert items[1].query_text == "sitecore denmark headquarters abroad"
    assert items[1].source_name == "example.com"
    assert items[1].title == "Podio company history"
    assert items[1].url == "https://example.com/podio-history"
    assert items[1].language == ""
    assert items[1].retrieved_at == "2026-04-27T10:03:00Z"
    assert "Podio" in items[1].raw_text
    assert items[0].retrieved_item_id != items[1].retrieved_item_id


def test_build_retrieval_prompt_reflects_recall_first_framing() -> None:
    query = Query(
        query_id="q_005",
        query_text="danish startups moved abroad",
        language="en",
        family="exploratory",
        created_at="2026-04-27T10:05:00Z",
    )

    prompt = json.loads(build_retrieval_prompt(query, limit=10))

    assert prompt["query"]["query_id"] == "q_005"
    assert prompt["requirements"]["max_results"] == 10
    assert "broad recall over precision" in prompt["requirements"]["focus"]
    assert "continued operation of the focal firm after the move" in prompt["requirements"]["focus"]
    instructions = prompt["instructions"]
    assert "Do not output scores, labels, or final classifications." in instructions
    assert "If no source URL is available, omit that result." in instructions
    assert "Keep one result per source URL." in instructions
    assert "Do not include labels such as 'firm name(s):', 'Danish evidence:', 'foreign headquarters:', or 'uncertainty:'." in instructions
    assert "Use focal_firm_names for candidate-firm names only." in instructions
    assert "Use non_focal_entities for people, acquirers, universities, roles, investors, and locations that are not the focal candidate firm." in instructions
    assert "Avoid pure acquisition announcements when no headquarters or main operations move is shown." in instructions
    assert "In evidence_note, explicitly mention when the evidence is acquisition-only and weak, but prefer not to return acquisition-only results unless no better results exist." in instructions
    assert any("Danish" in instruction or "Denmark" in instruction for instruction in instructions)


def test_parse_retrieved_items_excludes_non_focal_people_and_roles_from_raw_text() -> None:
    query = Query(
        query_id="q_006",
        query_text="coana socket acquisition",
        language="en",
        family="exploratory",
        created_at="2026-04-27T10:06:00Z",
    )
    payload = {
        "results": [
            {
                "title": "Coana acquired by Socket",
                "snippet": "The Danish security startup Coana was acquired by Socket.",
                "url": "https://example.com/coana",
                "source_name": "Example News",
                "language": "en",
                "raw_text": "Fallback text with Martin Torp.",
                "focal_firm_names": ["Coana"],
                "non_focal_entities": ["Socket", "Aarhus University", "Martin Torp"],
                "evidence_note": "The source ties Coana to Denmark and discusses Socket as the acquirer.",
            },
            {
                "title": "Issuu appoints CEO",
                "snippet": "Issuu later operated from the United States.",
                "url": "https://example.com/issuu",
                "source_name": "Example Source",
                "language": "en",
                "raw_text": "Fallback text with American CEO.",
                "focal_firm_names": ["Issuu"],
                "non_focal_entities": ["Joe Hyrkin", "American CEO"],
                "evidence_note": "The source mentions Issuu's U.S. leadership context after its Danish origins.",
            },
            {
                "title": "FJ Industries history",
                "snippet": "FJ Industries later expanded abroad.",
                "url": "https://example.com/fj-industries",
                "source_name": "Registry",
                "language": "en",
                "raw_text": "Fallback text with FJ Sintermetal AB.",
                "focal_firm_names": ["FJ Industries"],
                "non_focal_entities": ["FJ Sintermetal AB"],
                "evidence_note": "The source describes FJ Industries as the Danish-origin firm.",
            },
        ]
    }

    items = parse_retrieved_items(query=query, payload=payload, retrieved_at="2026-04-27T10:07:00Z")

    assert "Coana" in items[0].raw_text
    assert "Martin Torp" not in items[0].raw_text
    assert "Socket" not in items[0].raw_text
    assert "Issuu" in items[1].raw_text
    assert "American CEO" not in items[1].raw_text
    assert "Joe Hyrkin" not in items[1].raw_text
    assert "FJ Industries" in items[2].raw_text
    assert "FJ Sintermetal AB" not in items[2].raw_text


def test_save_retrieved_items_writes_expected_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "raw" / "retrieved.jsonl"
    items = [
        RetrievedItem(
            retrieved_item_id="ret_001",
            query_id="q_004",
            query_text="issuu palo alto",
            source_name="Example Source",
            title="Issuu moved headquarters",
            snippet="Issuu moved headquarters to Palo Alto.",
            url="https://example.com/issuu",
            language="en",
            retrieved_at="2026-04-27T10:04:00Z",
            raw_text="Issuu was founded in Copenhagen and later moved headquarters to Palo Alto.",
        )
    ]

    save_retrieved_items(items, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved == {
        "retrieved_item_id": "ret_001",
        "query_id": "q_004",
        "query_text": "issuu palo alto",
        "source_name": "Example Source",
        "title": "Issuu moved headquarters",
        "snippet": "Issuu moved headquarters to Palo Alto.",
        "url": "https://example.com/issuu",
        "language": "en",
        "retrieved_at": "2026-04-27T10:04:00Z",
        "raw_text": "Issuu was founded in Copenhagen and later moved headquarters to Palo Alto.",
    }
