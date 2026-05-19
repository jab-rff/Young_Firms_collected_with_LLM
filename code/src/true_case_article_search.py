"""Shared alias matching helpers for true-case article stage scripts."""

from __future__ import annotations

import re
from typing import Any


def build_alias_search_index(firm_specs: list[dict[str, Any]]) -> dict[str, Any]:
    alias_to_firms: dict[str, set[str]] = {}
    for spec in firm_specs:
        for alias in spec.get("aliases_display", []):
            key = alias.casefold()
            alias_to_firms.setdefault(key, set()).add(spec["firm"])

    aliases = sorted(alias_to_firms, key=lambda value: (-len(value), value))
    if not aliases:
        pattern = re.compile(r"$^")
    else:
        escaped = [re.escape(alias) for alias in aliases]
        pattern = re.compile(rf"(?<!\w)(?:{'|'.join(escaped)})(?!\w)", flags=re.IGNORECASE)

    return {
        "pattern": pattern,
        "alias_to_firms": alias_to_firms,
    }


def match_firms_in_text(text: str, search_index: dict[str, Any]) -> set[str]:
    matched_firms: set[str] = set()
    for match in search_index["pattern"].finditer(text):
        alias = match.group(0).casefold()
        matched_firms.update(search_index["alias_to_firms"].get(alias, set()))
    return matched_firms
