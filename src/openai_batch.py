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
    network_retry_attempts: int = 5,
    network_retry_delay_seconds: int = 5,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not request_items:
        return {}, {}

    temp_path = _write_batch_input_file(request_items)
    try:
        with temp_path.open("rb") as handle:
            uploaded_file = _retry_api_call(
                lambda: client.files.create(file=handle, purpose="batch"),
                attempts=network_retry_attempts,
                delay_seconds=network_retry_delay_seconds,
                operation_name="files.create(batch)",
            )
        batch = _retry_api_call(
            lambda: client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/responses",
                completion_window=completion_window,
            ),
            attempts=network_retry_attempts,
            delay_seconds=network_retry_delay_seconds,
            operation_name="batches.create",
        )
        print(f"batch_id={batch.id}")

        while True:
            batch = _retry_api_call(
                lambda: client.batches.retrieve(batch.id),
                attempts=network_retry_attempts,
                delay_seconds=network_retry_delay_seconds,
                operation_name=f"batches.retrieve({batch.id})",
            )
            if batch.status in TERMINAL_BATCH_STATUSES:
                break
            time.sleep(poll_interval_seconds)

        batch_payload = _serialize(batch)
        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} did not complete successfully: {batch.status}")
        output_file_id = getattr(batch, "output_file_id", None) or batch_payload.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Batch {batch.id} completed without an output_file_id.")
        content = _retry_api_call(
            lambda: client.files.content(output_file_id),
            attempts=network_retry_attempts,
            delay_seconds=network_retry_delay_seconds,
            operation_name=f"files.content({output_file_id})",
        )
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


def extract_json_output_payload(response_body: dict[str, Any]) -> dict[str, Any]:
    output_text = str(response_body.get("output_text") or "").strip()
    if output_text:
        return json.loads(output_text)

    for item in response_body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content") or []:
            if content_item.get("type") != "output_text":
                continue
            text = str(content_item.get("text") or "").strip()
            if text:
                return json.loads(text)

    raise ValueError("Batch response body did not contain JSON output_text content.")


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


def _retry_api_call(
    func: Any,
    *,
    attempts: int,
    delay_seconds: int,
    operation_name: str,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= max(1, attempts) or not _is_retryable_network_error(exc):
                break
            print(f"retrying {operation_name}: attempt={attempt + 1}/{attempts}")
            time.sleep(delay_seconds)
    if last_exc is None:
        raise RuntimeError(f"{operation_name} failed without an exception.")
    raise last_exc


def _is_retryable_network_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    retry_markers = [
        "connection error",
        "apiconnectionerror",
        "connecterror",
        "getaddrinfo failed",
        "temporarily unavailable",
        "timed out",
        "timeout",
    ]
    return any(marker in text for marker in retry_markers)
