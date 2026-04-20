"""Simple deterministic deduplication."""


def deduplicate_candidates(candidates: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result
