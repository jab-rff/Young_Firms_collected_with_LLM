"""Helpers for generating broad discovery queries."""


def generate_queries(seed_terms: list[str]) -> list[str]:
    """Return normalized query strings from seed terms."""
    return [term.strip() for term in seed_terms if term.strip()]
