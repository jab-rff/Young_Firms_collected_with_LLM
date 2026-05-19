import pytest

from src.openai_batch import extract_json_output_payload


def test_extract_json_output_payload_reads_flat_output_text() -> None:
    payload = extract_json_output_payload({"output_text": '{"record": {"ok": true}}'})
    assert payload == {"record": {"ok": True}}


def test_extract_json_output_payload_reads_nested_message_output() -> None:
    payload = extract_json_output_payload(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"record": {"ok": true}}',
                        }
                    ],
                }
            ]
        }
    )
    assert payload == {"record": {"ok": True}}


def test_extract_json_output_payload_raises_when_missing_text() -> None:
    with pytest.raises(ValueError, match="did not contain JSON output_text content"):
        extract_json_output_payload({"output": []})
