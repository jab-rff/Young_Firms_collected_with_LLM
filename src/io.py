"""Local transparent file IO for the recall pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

from src.data_models import RetrievedItem
from src.normalization import make_retrieved_item_id

T = TypeVar("T")


def load_retrieved_items(path: Path) -> list[RetrievedItem]:
    """Load local retrieved records from JSONL without altering raw text."""
    items: list[RetrievedItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            retrieved_item_id = data.get("retrieved_item_id") or make_retrieved_item_id(
                row_number=row_number,
                query_id=data["query_id"],
                url=data.get("url", ""),
                title=data.get("title", ""),
            )
            items.append(
                RetrievedItem(
                    retrieved_item_id=retrieved_item_id,
                    query_id=data["query_id"],
                    query_text=data["query_text"],
                    source_name=data["source_name"],
                    title=data.get("title", ""),
                    snippet=data.get("snippet", ""),
                    url=data.get("url", ""),
                    language=data.get("language", ""),
                    retrieved_at=data["retrieved_at"],
                    raw_text=data.get("raw_text", ""),
                )
            )
    return items


def dataclass_to_dict(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    raise TypeError(f"Expected dataclass instance, got {type(record)!r}")


def save_jsonl(records: Iterable[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            data = dataclass_to_dict(record) if is_dataclass(record) else record
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_parquet(records: Iterable[Any], path: Path) -> None:
    """Write real parquet via pyarrow.

    pyarrow is intentionally the only parquet dependency. It is imported here so
    non-parquet stages remain runnable without optional IO dependencies.
    """
    rows = [dataclass_to_dict(record) if is_dataclass(record) else record for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Writing parquet requires pyarrow. Install project dependencies with "
            "`pip install -e .` before running the pipeline."
        ) from exc

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
