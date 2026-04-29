"""Lightweight candidate memory and follow-up discovery query generation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_models import Query
from src.io import save_jsonl
from src.normalization import normalize_company_name, strip_trailing_legal_suffixes

_MEMORY_FIELDS = [
    "firm_name",
    "normalized_name",
    "first_seen_at",
    "last_seen_at",
    "times_seen",
    "source_urls",
    "query_ids",
    "notes",
]
_DEFAULT_NOTES = ""
_MAX_EXCLUSION_FIRMS = 100


def load_known_firms(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record["normalized_name"] for record in _load_memory_records(path) if record["normalized_name"]}


def save_known_firms(records: list[dict[str, Any]], path: Path) -> None:
    normalized_records = [_normalize_memory_record(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".jsonl":
        save_jsonl(normalized_records, path)
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MEMORY_FIELDS)
        writer.writeheader()
        for record in normalized_records:
            row = dict(record)
            row["source_urls"] = json.dumps(record["source_urls"], ensure_ascii=False)
            row["query_ids"] = json.dumps(record["query_ids"], ensure_ascii=False)
            writer.writerow(row)


def update_known_firms_from_model1(
    model1_path: Path,
    memory_path: Path | None = None,
    seen_at: str | None = None,
) -> list[dict[str, Any]]:
    records_by_name: dict[str, dict[str, Any]] = {}
    if memory_path is not None and memory_path.exists():
        for record in _load_memory_records(memory_path):
            records_by_name[record["normalized_name"]] = record

    timestamp = seen_at or datetime.now(timezone.utc).isoformat()
    for candidate in _load_jsonl_rows(model1_path):
        firm_name = str(candidate.get("firm_name") or "").strip()
        normalized_name = normalize_company_name(firm_name)
        if not normalized_name:
            continue
        display_name = strip_trailing_legal_suffixes(firm_name) or firm_name
        sources = _string_list(candidate.get("sources"))
        query_ids = _string_list(candidate.get("query_ids"))
        existing = records_by_name.get(normalized_name)
        if existing is None:
            records_by_name[normalized_name] = {
                "firm_name": display_name,
                "normalized_name": normalized_name,
                "first_seen_at": timestamp,
                "last_seen_at": timestamp,
                "times_seen": 1,
                "source_urls": sources,
                "query_ids": query_ids,
                "notes": _DEFAULT_NOTES,
            }
            continue

        existing["firm_name"] = _preferred_display_name(existing["firm_name"], display_name)
        existing["last_seen_at"] = timestamp
        existing["times_seen"] = int(existing.get("times_seen", 0)) + 1
        existing["source_urls"] = _merge_unique(existing.get("source_urls"), sources)
        existing["query_ids"] = _merge_unique(existing.get("query_ids"), query_ids)

    return sorted(records_by_name.values(), key=lambda record: record["normalized_name"])


def build_exclusion_prompt(records_or_names: list[dict[str, Any]] | set[str] | list[str], limit: int = _MAX_EXCLUSION_FIRMS) -> str:
    names: list[str] = []
    if isinstance(records_or_names, set):
        names = sorted(name for name in records_or_names if name)
    else:
        for item in records_or_names:
            if isinstance(item, dict):
                name = str(item.get("firm_name") or "").strip()
            else:
                name = str(item).strip()
            if name:
                names.append(name)

    unique_names = list(dict.fromkeys(names))
    if not unique_names:
        return ""
    capped_names = unique_names[:limit]
    return "Do not return these already-known firms: " + ", ".join(capped_names) + "."


def create_followup_queries(
    memory_path: Path,
    round_number: int,
    max_exclusions: int = _MAX_EXCLUSION_FIRMS,
) -> list[Query]:
    records = _load_memory_records(memory_path) if memory_path.exists() else []
    exclusion_text = build_exclusion_prompt(records, limit=max_exclusions)
    created_at = datetime.now(timezone.utc).isoformat()

    query_specs = [
        ("en", "followup_destination_country", "Additional Danish-founded firms with main headquarters moved to the UK"),
        ("en", "followup_destination_country", "Additional Danish-founded firms with main office moved to Germany"),
        ("en", "followup_destination_country", "Additional Danish-founded firms with executive base moved to Sweden"),
        ("en", "followup_destination_country", "Additional Danish-founded firms with headquarters moved to the Netherlands"),
        ("en", "followup_destination_country", "Additional Danish-founded firms with main operations moved to Switzerland"),
        ("en", "followup_destination_country", "Additional Danish-founded firms with headquarters moved to Ireland"),
        ("en", "followup_destination_country", "Additional Danish-founded firms with headquarters moved to the US"),
        ("en", "followup_destination_country", "Additional Danish-founded firms with executive headquarters moved to Singapore"),
        ("en", "followup_sector", "Less-known Danish-founded software firms now headquartered abroad"),
        ("en", "followup_sector", "Less-known Danish-founded biotech firms with main operations abroad"),
        ("en", "followup_sector", "Less-known Danish-founded fintech firms now based abroad"),
        ("en", "followup_sector", "Less-known Danish-founded gaming firms with headquarters abroad"),
        ("en", "followup_sector", "Less-known Danish-founded medtech firms with executive base abroad"),
        ("en", "followup_sector", "Less-known Danish-founded logistics firms with main office abroad"),
        ("en", "followup_sector", "Less-known Danish-founded cleantech firms now headquartered abroad"),
        ("en", "followup_sector", "Less-known Danish-founded retail or consumer firms now headquartered abroad"),
        ("en", "followup_sector", "Less-known Danish-founded hospitality firms with headquarters abroad"),
        ("en", "followup_sector", "Less-known Danish-founded design firms with headquarters abroad"),
        ("en", "followup_mechanism", "Additional Danish-founded firms whose principal executive offices later moved abroad"),
        ("en", "followup_mechanism", "Additional Danish-founded firms whose group headquarters later moved abroad"),
        ("en", "followup_mechanism", "Additional Danish-founded firms that created a Delaware or UK parent and later ran the group from abroad"),
        ("en", "followup_mechanism", "Additional Danish-founded firms with IPO or listing-related relocation abroad"),
        ("en", "followup_mechanism", "Additional Danish-founded firms with acquisition-linked relocation where the operating company continued abroad"),
        ("en", "followup_mechanism", "Additional Danish-founded firms where founders or executives relocated the company base abroad"),
        ("en", "followup_city_pair", "Danish-founded firms that moved from Copenhagen to London"),
        ("en", "followup_city_pair", "Danish-founded firms that moved from Copenhagen to New York"),
        ("en", "followup_city_pair", "Danish-founded firms that moved from Copenhagen to San Francisco"),
        ("en", "followup_city_pair", "Danish-founded firms that moved from Aarhus to London or New York"),
        ("en", "followup_source_style", "Danish-founded relocation firms mentioned in Danish business media or archived company pages"),
        ("en", "followup_source_style", "Danish-founded relocation firms found in investor filings or annual reports"),
        ("en", "followup_long_tail", "Find additional obscure Danish-founded firms that later moved headquarters abroad"),
        ("en", "followup_non_us_destinations", "Additional Danish-founded firms now headquartered outside the US in Europe or Asia"),
        ("da", "followup_danish_sources", "Børsen dansk virksomhed grundlagt i Danmark senere hovedkontor i udlandet"),
        ("da", "followup_danish_sources", "Finans dansk startup hovedkontor flyttet til udlandet"),
        ("da", "followup_danish_sources", "ITWatch dansk softwarevirksomhed nu baseret i USA eller London"),
        ("da", "followup_danish_sources", "Version2 dansk techvirksomhed flyttede hovedkvarter til udlandet"),
        ("da", "followup_danish_sources", "Trendsonline dansk startup hovedkontor i udlandet"),
        ("da", "followup_danish_sources", "TechSavvy dansk virksomhed ledelse flyttet til udlandet"),
        ("da", "followup_danish_sources", "Børsen dansk virksomhed principal executive office i udlandet"),
        ("da", "followup_danish_sources", "Finans dansk virksomhed globalt hovedkontor i London eller New York"),
        ("da", "followup_danish_sources", "Version2 dansk virksomhed holdingselskab i UK eller Delaware efter grundlæggelse i Danmark"),
        ("da", "followup_destination_country", "Yderligere danske virksomheder med hovedkontor flyttet til Storbritannien"),
        ("da", "followup_destination_country", "Yderligere danske virksomheder med hovedkontor flyttet til Tyskland"),
        ("da", "followup_destination_country", "Yderligere danske virksomheder med hovedkontor flyttet til Sverige"),
        ("da", "followup_destination_country", "Yderligere danske virksomheder med hovedkontor flyttet til Holland"),
        ("da", "followup_destination_country", "Yderligere danske virksomheder med hovedkontor flyttet til Schweiz"),
        ("da", "followup_destination_country", "Yderligere danske virksomheder med hovedkontor flyttet til Irland"),
        ("da", "followup_destination_country", "Yderligere danske virksomheder med hovedkontor flyttet til USA"),
        ("da", "followup_destination_country", "Yderligere danske virksomheder med hovedkontor flyttet til Singapore"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte softwarevirksomheder med hovedkvarter i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte biotekvirksomheder med hoveddrift i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte fintech virksomheder nu baseret i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte gaming virksomheder med hovedkontor i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte medtech virksomheder med ledelse i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte logistikvirksomheder med hovedkontor i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte cleantech virksomheder nu baseret i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte retail eller consumer virksomheder med hovedkvarter i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte hospitality virksomheder med hovedkvarter i udlandet"),
        ("da", "followup_sector", "Mindre kendte dansk grundlagte designvirksomheder med hovedkvarter i udlandet"),
        ("da", "followup_mechanism", "Yderligere danske virksomheder hvor principal executive office senere blev flyttet til udlandet"),
        ("da", "followup_mechanism", "Yderligere danske virksomheder med globalt eller koncernhovedkontor flyttet til udlandet"),
        ("da", "followup_mechanism", "Yderligere danske virksomheder med UK eller Delaware holdingselskab efter grundlæggelse i Danmark"),
        ("da", "followup_mechanism", "Yderligere danske virksomheder med IPO eller børsnotering knyttet til flytning af hovedkontor"),
        ("da", "followup_mechanism", "Yderligere danske virksomheder hvor opkøb eller fusion blev fulgt af flytning af ledelse eller hovedkontor"),
        ("da", "followup_mechanism", "Yderligere danske virksomheder hvor grundlæggere eller ledelse flyttede virksomhedens base til udlandet"),
        ("da", "followup_city_pair", "Danske virksomheder flyttet fra København til London"),
        ("da", "followup_city_pair", "Danske virksomheder flyttet fra København til New York"),
        ("da", "followup_city_pair", "Danske virksomheder flyttet fra København til San Francisco"),
        ("da", "followup_city_pair", "Danske virksomheder flyttet fra Aarhus til London eller New York"),
        ("da", "followup_long_tail", "Find yderligere mindre kendte danske virksomheder som flyttede hovedkvarter til udlandet"),
        ("da", "followup_non_us_destinations", "Yderligere danske virksomheder med hovedkontor flyttet uden for USA"),
    ]

    queries: list[Query] = []
    for index, (language, family, prompt_text) in enumerate(query_specs, start=1):
        query_text = prompt_text
        if exclusion_text:
            query_text = f"{prompt_text}. {exclusion_text} Prioritize additional less-known firms."
        query_id = f"q_followup_r{round_number:03d}_{language}_{index:03d}"
        queries.append(
            Query(
                query_id=query_id,
                query_text=query_text,
                language=language,
                family=family,
                created_at=created_at,
            )
        )
    return queries


def export_followup_queries(output_path: Path, memory_path: Path, round_number: int) -> list[Query]:
    queries = create_followup_queries(memory_path=memory_path, round_number=round_number)
    save_jsonl(queries, output_path)
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate follow-up discovery queries with known-firm exclusions.")
    parser.add_argument("--memory", required=True, type=Path, help="Path to known firm memory CSV or JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write follow-up query JSONL.")
    parser.add_argument("--round", required=True, type=int, help="Discovery round number.")
    args = parser.parse_args()

    queries = export_followup_queries(output_path=args.output, memory_path=args.memory, round_number=args.round)
    print(f"followup_queries={len(queries)}")
    print(f"output_path={args.output}")


def _load_memory_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return [_normalize_memory_record(row) for row in _load_jsonl_rows(path)]

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                _normalize_memory_record(
                    {
                        "firm_name": row.get("firm_name", ""),
                        "normalized_name": row.get("normalized_name", ""),
                        "first_seen_at": row.get("first_seen_at", ""),
                        "last_seen_at": row.get("last_seen_at", ""),
                        "times_seen": row.get("times_seen", 0),
                        "source_urls": _decode_list_field(row.get("source_urls", "")),
                        "query_ids": _decode_list_field(row.get("query_ids", "")),
                        "notes": row.get("notes", ""),
                    }
                )
            )
    return records


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_memory_record(record: dict[str, Any]) -> dict[str, Any]:
    firm_name = str(record.get("firm_name") or "").strip()
    normalized_name = str(record.get("normalized_name") or normalize_company_name(firm_name)).strip()
    return {
        "firm_name": strip_trailing_legal_suffixes(firm_name) or firm_name,
        "normalized_name": normalized_name,
        "first_seen_at": str(record.get("first_seen_at") or ""),
        "last_seen_at": str(record.get("last_seen_at") or ""),
        "times_seen": int(record.get("times_seen") or 0),
        "source_urls": _string_list(record.get("source_urls")),
        "query_ids": _string_list(record.get("query_ids")),
        "notes": str(record.get("notes") or ""),
    }


def _decode_list_field(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    return _string_list(parsed)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        items = []
    else:
        text = str(value).strip()
        items = [text] if text else []
    return list(dict.fromkeys(items))


def _merge_unique(left: Any, right: Any) -> list[str]:
    return list(dict.fromkeys(_string_list(left) + _string_list(right)))


def _preferred_display_name(current: str, new_value: str) -> str:
    current_clean = strip_trailing_legal_suffixes(current) or current
    new_clean = strip_trailing_legal_suffixes(new_value) or new_value
    if not current_clean:
        return new_clean
    if not new_clean:
        return current_clean
    return min((current_clean, new_clean), key=lambda value: (len(value), value.casefold()))


if __name__ == "__main__":
    main()
