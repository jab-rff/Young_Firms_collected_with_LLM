"""Retrieve broad web-search evidence with OpenAI and save RetrievedItem JSONL.

Command order:
1. python generate_queries.py --output data/queries_extended.jsonl --include-exploratory
2. python retrieve_with_openai.py --queries data/queries_extended.jsonl --output data/raw/openai_retrieved_test_001.jsonl --limit 10
3. python -m src.main --input data/raw/openai_retrieved_test_001.jsonl --output data/candidates/openai_test_001
4. python export_candidates.py --candidates data/candidates/openai_test_001/candidates.jsonl --mentions data/candidates/openai_test_001/mentions.jsonl --output data/candidates/openai_test_001/candidates_review.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.data_models import Query, RetrievedItem
from src.io import save_jsonl
from src.normalization import make_retrieved_item_id

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_LIMIT = 10


SYSTEM_PROMPT = """You are a recall-first web retrieval assistant for a company research pipeline.

Your job is to collect broad, source-backed web evidence for firms that may have been founded in Denmark and may later have moved their own executive headquarters, main office, executive base, or main operations abroad while continuing as an operating company.

This is a retrieval stage, not an adjudication, scoring, or classification stage:
- keep plausible and uncertain cases
- do not output scores, labels, rankings, or final judgments
- do not decide whether a firm truly qualifies
- preserve provenance for every result
- do not exclude acquisition or merger cases; preserve that context if present

Retrieval framing:
- candidate evidence can include explicit or indirect hints that a firm was founded, based, or incorporated in Denmark
- candidate evidence can include explicit or indirect hints that the firm later had its own headquarters, executive base, main office, or main operations abroad
- primary retrieval target is a Danish-origin firm that later appears to continue as an operating company after moving its own headquarters or main operations abroad
- a foreign sales office or subsidiary is not final proof of headquarters relocation, but it can still be kept as weak retrieval evidence if it appears near Danish founding context
- Danish founder abroad is not the same as founded in Denmark, but such cases may still be retrieved as weak evidence for later filtering
- acquisition or merger context is secondary and should only be kept when it co-occurs with possible headquarters, main office, or main operations relocation
- an acquired firm continuing as a separate subsidiary or brand with its own headquarters abroad may be kept as uncertain
- pure acquisition announcements, absorbed or discontinued firms, buyer-only expansion stories, and foreign office or service-center moves without evidence of the focal firm's own main headquarters or operations move should be de-prioritized or avoided

Write raw_text as natural, source-grounded evidence prose, not as labels, bullets, or field names.
Do not include labels such as "firm name(s):", "Danish evidence:", "foreign headquarters:", or "uncertainty:".
Instead, write one or two natural sentences that mention the firm and any relevant Denmark, foreign headquarters, operations, leadership, acquisition, merger, or caveat details found in the source.

For each result item:
- focal_firm_names should contain company names that may be the Danish-origin focal candidate firm
- non_focal_entities should contain people, acquirers, universities, research institutions, investors, roles, and locations that appear in the source but are not the focal candidate firm
- evidence_note should be a short natural-language explanation of why the source was retrieved

Focal entity rules:
- prefer the Danish-origin company as focal_firm_names
- if a source says "X acquired Y", usually Y is focal if Y is the Danish-origin firm; X should usually be non_focal unless X itself appears to be the Danish-origin firm
- include only the Danish-origin firm that may have moved its own headquarters or main operations abroad in focal_firm_names
- people, founders, CEOs, professors, and role phrases belong in non_focal_entities, not focal_firm_names
- universities and research institutions belong in non_focal_entities unless the query is explicitly about institutions
- locations and adjective phrases like US-based, Danish-founded, and Copenhagen-based are not focal names
- keep uncertain plausible focal firms; do not over-filter

Return search-result style evidence items only. Each item must map to one real source URL."""


def load_queries(path: Path) -> list[Query]:
    text = _read_text_with_supported_encodings(path)
    queries: list[Query] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        queries.append(
            Query(
                query_id=data["query_id"],
                query_text=data["query_text"],
                language=data["language"],
                family=data["family"],
                created_at=data["created_at"],
            )
        )
    return queries


def _read_text_with_supported_encodings(path: Path) -> str:
    last_error: UnicodeError | None = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return path.read_text(encoding=encoding).lstrip("\ufeff")
        except UnicodeError as exc:
            last_error = exc
    raise RuntimeError(
        f"Could not decode query file {path} with supported encodings: utf-8, utf-8-sig, utf-16."
    ) from last_error


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


def build_retrieval_prompt(query: Query, limit: int) -> str:
    payload = {
        "task": (
            "Collect broad web-search evidence for firms that may have been founded in Denmark "
            "and may later have moved their own executive headquarters or main operations abroad."
        ),
        "query": {
            "query_id": query.query_id,
            "query_text": query.query_text,
            "language": query.language,
            "family": query.family,
        },
        "requirements": {
            "max_results": limit,
            "focus": [
                "companies possibly founded in Denmark",
                "possible own executive headquarters or main operations abroad",
                "continued operation of the focal firm after the move",
                "broad recall over precision",
            ],
            "return_fields": [
                "title",
                "snippet",
                "url",
                "source_name",
                "language",
                "raw_text",
                "focal_firm_names",
                "non_focal_entities",
                "evidence_note",
            ],
        },
        "instructions": [
            "Use web search.",
            "Return up to max_results result items.",
            "Keep uncertain but plausible evidence.",
            "Do not decide whether a company truly qualifies.",
            "Do not output scores, labels, or final classifications.",
            "Preserve acquisition or merger context if present.",
            "Prioritize Danish-origin firms that appear to continue as operating companies after moving their own headquarters or main operations abroad.",
            "Treat acquisition or merger context as secondary unless it co-occurs with plausible evidence of the focal firm's own headquarters, main office, or main operations abroad.",
            "Avoid pure acquisition announcements when no headquarters or main operations move is shown.",
            "Avoid buyer-only expansion stories, absorbed or discontinued firms, and foreign office or service-center moves that do not indicate the focal firm's own main headquarters or operations move.",
            "Prefer company pages, registries, press releases, reliable startup databases, news articles, SEC or company filings, and archived pages.",
            "Do not fabricate URLs.",
            "If no source URL is available, omit that result.",
            "Keep one result per source URL.",
            "Do not deduplicate across different sources.",
            "Use snippet for a short search-result style description.",
            "Use raw_text for a brief natural-language evidence note grounded in the source.",
            "Write raw_text as one or two natural sentences, not as labels, bullet points, templates, or field names.",
            "Do not include labels such as 'firm name(s):', 'Danish evidence:', 'foreign headquarters:', or 'uncertainty:'.",
            "In raw_text, naturally mention firm names, Denmark-related founding or basing evidence, foreign headquarters or operations evidence, acquisition or merger context, and uncertainty if relevant.",
            "Use focal_firm_names for candidate-firm names only.",
            "Use non_focal_entities for people, acquirers, universities, roles, investors, and locations that are not the focal candidate firm.",
            "Use evidence_note as a short natural-language explanation of why the source was retrieved.",
            "In evidence_note, explicitly mention when the evidence is acquisition-only and weak, but prefer not to return acquisition-only results unless no better results exist.",
            "If language is unclear, return an empty string.",
            "If source_name is unclear, use the publication or domain name.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def retrieval_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "title",
                        "snippet",
                        "url",
                        "source_name",
                        "language",
                        "raw_text",
                        "focal_firm_names",
                        "non_focal_entities",
                        "evidence_note",
                    ],
                    "properties": {
                        "title": {"type": "string"},
                        "snippet": {"type": "string"},
                        "url": {"type": "string"},
                        "source_name": {"type": "string"},
                        "language": {"type": "string"},
                        "raw_text": {"type": "string"},
                        "focal_firm_names": {"type": "array", "items": {"type": "string"}},
                        "non_focal_entities": {"type": "array", "items": {"type": "string"}},
                        "evidence_note": {"type": "string"},
                    },
                },
            }
        },
    }


def call_openai_retrieval(query: Query, model: str, limit: int) -> dict[str, Any]:
    load_openai_api_key()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI retrieval requires the openai package. Install dependencies with "
            "`pip install -e .` and set OPENAI_API_KEY."
        ) from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_retrieval_prompt(query, limit)},
        ],
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        text={
            "format": {
                "type": "json_schema",
                "name": "retrieval_results",
                "strict": True,
                "schema": retrieval_json_schema(),
            }
        },
    )
    return json.loads(response.output_text)


def parse_retrieved_items(
    query: Query,
    payload: dict[str, Any],
    retrieved_at: str | None = None,
) -> list[RetrievedItem]:
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    items: list[RetrievedItem] = []
    seen_urls: set[str] = set()

    for index, row in enumerate(payload.get("results", []), start=1):
        title = str(row.get("title") or "").strip()
        snippet = str(row.get("snippet") or "").strip()
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        source_name = str(row.get("source_name") or "").strip() or infer_source_name(url)
        language = str(row.get("language") or "").strip()
        evidence_note = str(row.get("evidence_note") or "").strip()
        focal_firm_names = [str(name).strip() for name in row.get("focal_firm_names", []) if str(name).strip()]
        non_focal_entities = [str(name).strip() for name in row.get("non_focal_entities", []) if str(name).strip()]
        raw_text = _build_raw_text(
            title=_strip_non_focal_entities(title, non_focal_entities),
            snippet=_strip_non_focal_entities(snippet, non_focal_entities),
            evidence_note=_strip_non_focal_entities(evidence_note, non_focal_entities),
            focal_firm_names=focal_firm_names,
            fallback_raw_text=str(row.get("raw_text") or "").strip(),
        )
        raw_text = _strip_non_focal_entities(raw_text, non_focal_entities)
        retrieved_item_id = make_retrieved_item_id(
            row_number=index,
            query_id=query.query_id,
            url=url,
            title=title,
        )
        items.append(
            RetrievedItem(
                retrieved_item_id=retrieved_item_id,
                query_id=query.query_id,
                query_text=query.query_text,
                source_name=source_name,
                title=title,
                snippet=snippet,
                url=url,
                language=language,
                retrieved_at=timestamp,
                raw_text=raw_text,
            )
        )
    return items


def infer_source_name(url: str) -> str:
    if not url:
        return ""
    hostname = urlparse(url).netloc.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def _build_raw_text(
    title: str,
    snippet: str,
    evidence_note: str,
    focal_firm_names: list[str],
    fallback_raw_text: str,
) -> str:
    parts: list[str] = []
    if title:
        parts.append(title)
    if snippet and snippet not in parts:
        parts.append(snippet)
    if evidence_note and evidence_note not in parts:
        parts.append(evidence_note)
    if focal_firm_names:
        focal_text = ", ".join(dict.fromkeys(focal_firm_names))
        if focal_text and focal_text not in parts:
            parts.append(focal_text)
    if not parts and fallback_raw_text:
        parts.append(fallback_raw_text)
    return " ".join(parts).strip()


def _strip_non_focal_entities(text: str, non_focal_entities: list[str]) -> str:
    cleaned = text
    for entity in sorted(non_focal_entities, key=len, reverse=True):
        if not entity:
            continue
        patterns = [
            rf"\b(?:US-based|U\.S\.-based|US headquartered|U\.S\. headquartered)\s+{re.escape(entity)}(?:['’]s)?\b",
            rf"\b{re.escape(entity)}(?:['’]s)?\b",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip(" \t\r\n,.;:")


def retrieve_queries(
    queries_path: Path,
    output_path: Path,
    model: str,
    limit: int,
) -> list[RetrievedItem]:
    queries = load_queries(queries_path)
    all_items: list[RetrievedItem] = []

    for position, query in enumerate(queries, start=1):
        payload = call_openai_retrieval(query=query, model=model, limit=limit)
        items = parse_retrieved_items(query=query, payload=payload)
        all_items.extend(items)
        print(f"[{position}/{len(queries)}] query_id={query.query_id} results={len(items)}")

    save_retrieved_items(all_items, output_path)
    return all_items


def save_retrieved_items(items: list[RetrievedItem], output_path: Path) -> None:
    rows = [asdict(item) for item in items]
    save_jsonl(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve raw web-search evidence with OpenAI web search.")
    parser.add_argument(
        "--queries",
        required=True,
        type=Path,
        help="Path to a query JSONL file, e.g. data/queries_extended.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write raw RetrievedItem JSONL output.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use for retrieval (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of retrieval items to request per query (default: {DEFAULT_LIMIT})",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be greater than 0")

    items = retrieve_queries(
        queries_path=args.queries,
        output_path=args.output,
        model=args.model,
        limit=args.limit,
    )
    print(f"saved_items={len(items)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
