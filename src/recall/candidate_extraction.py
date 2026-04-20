"""Minimal candidate extraction helpers."""


def extract_candidates(lines: list[str]) -> list[str]:
    """Return non-empty candidate strings."""
    return [line.strip() for line in lines if line.strip()]
