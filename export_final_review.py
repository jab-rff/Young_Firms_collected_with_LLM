"""Export final validated candidates to a manual-review CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def prepare_review_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "firm_name": str(record.get("firm_name") or ""),
        "first_legal_entity_name": str(record.get("first_legal_entity_name") or ""),
        "validation_label": str(record.get("validation_label") or ""),
        "needs_human_review": "true" if record.get("needs_human_review") else "false",
        "founded_in_denmark": str(record.get("founded_in_denmark") or ""),
        "founding_year": _stringify(record.get("founding_year")),
        "founding_city": str(record.get("founding_city") or ""),
        "founding_country_iso": str(record.get("founding_country_iso") or ""),
        "moved_hq_abroad": str(record.get("moved_hq_abroad") or ""),
        "move_year": _stringify(record.get("move_year")),
        "moved_to_city": str(record.get("moved_to_city") or ""),
        "moved_to_country_iso": str(record.get("moved_to_country_iso") or ""),
        "hq_today_city": str(record.get("hq_today_city") or ""),
        "hq_today_country_iso": str(record.get("hq_today_country_iso") or ""),
        "status_today": str(record.get("status_today") or ""),
        "confidence": str(record.get("confidence") or ""),
        "validation_reason": str(record.get("validation_reason") or ""),
        "exclusion_reason": str(record.get("exclusion_reason") or ""),
        "evidence_summary": str(record.get("evidence_summary") or ""),
        "founding_evidence": str(record.get("founding_evidence") or ""),
        "relocation_evidence": str(record.get("relocation_evidence") or ""),
        "ma_evidence": str(record.get("ma_evidence") or ""),
        "relocation_context": str(record.get("relocation_context") or ""),
        "ma_context": str(record.get("ma_context") or ""),
        "uncertainty_note": str(record.get("uncertainty_note") or ""),
        "sources_founding": " | ".join(str(value) for value in record.get("sources_founding") or []),
        "sources_relocation": " | ".join(str(value) for value in record.get("sources_relocation") or []),
        "sources_ma": " | ".join(str(value) for value in record.get("sources_ma") or []),
        "sources_status_today": " | ".join(str(value) for value in record.get("sources_status_today") or []),
    }


def export_final_review(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = list(prepare_review_row(records[0]).keys()) if records else [
        "firm_name",
        "first_legal_entity_name",
        "validation_label",
        "needs_human_review",
        "founded_in_denmark",
        "founding_year",
        "founding_city",
        "founding_country_iso",
        "moved_hq_abroad",
        "move_year",
        "moved_to_city",
        "moved_to_country_iso",
        "hq_today_city",
        "hq_today_country_iso",
        "status_today",
        "confidence",
        "validation_reason",
        "exclusion_reason",
        "evidence_summary",
        "founding_evidence",
        "relocation_evidence",
        "ma_evidence",
        "relocation_context",
        "ma_context",
        "uncertainty_note",
        "sources_founding",
        "sources_relocation",
        "sources_ma",
        "sources_status_today",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(prepare_review_row(record))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final validated candidates to review CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Path to Model 3 validated JSONL input.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write final review CSV output.")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    export_final_review(records, args.output)
    print(f"review_rows={len(records)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
