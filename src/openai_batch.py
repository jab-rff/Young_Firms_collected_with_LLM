"""Helpers for running OpenAI Responses API requests through the Batch API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def run_responses_batch(
    *,
    client: Any,
    request_items: list[dict[str, Any]],
    completion_window: str = "24h",
    poll_interval_seconds: int = 10,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not request_items:
        return {}, {}

    temp_path = _write_batch_input_file(request_items)
    try:
        with temp_path.open("rb") as handle:
            uploaded_file = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/responses",
            completion_window=completion_window,
        )

        while True:
            batch = client.batches.retrieve(batch.id)
            if batch.status in TERMINAL_BATCH_STATUSES:
                break
            time.sleep(poll_interval_seconds)

        batch_payload = _serialize(batch)
        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} did not complete successfully: {batch.status}")
        output_file_id = getattr(batch, "output_file_id", None) or batch_payload.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Batch {batch.id} completed without an output_file_id.")
        content = client.files.content(output_file_id)
        responses_by_custom_id = parse_batch_output_content(_file_content_to_text(content))
        return responses_by_custom_id, batch_payload
    finally:
        temp_path.unlink(missing_ok=True)


def build_batch_request_item(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def parse_batch_output_content(text: str) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        custom_id = str(item.get("custom_id") or "").strip()
        if not custom_id:
            continue
        error = item.get("error")
        if error:
            raise RuntimeError(f"Batch request {custom_id} failed: {error}")
        response = item.get("response") or {}
        status_code = int(response.get("status_code") or 0)
        if status_code >= 400:
            raise RuntimeError(f"Batch request {custom_id} returned status {status_code}.")
        body = response.get("body") or {}
        parsed[custom_id] = body
    return parsed


def _write_batch_input_file(request_items: list[dict[str, Any]]) -> Path:
    with NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False, suffix=".jsonl") as handle:
        for item in request_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return Path(handle.name)


def _serialize(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"repr": repr(value)}


def _file_content_to_text(content: Any) -> str:
    text = getattr(content, "text", None)
    if callable(text):
        return text()
    if isinstance(text, str):
        return text
    binary = getattr(content, "content", None)
    if isinstance(binary, bytes):
        return binary.decode("utf-8")
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return str(content)
