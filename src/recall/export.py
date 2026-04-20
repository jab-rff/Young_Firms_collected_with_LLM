"""Helpers for exporting review-ready candidate lists."""

from pathlib import Path


def export_candidates(candidates: list[str], output_path: Path) -> None:
    """Write candidates one per line."""
    output_path.write_text("\n".join(candidates) + ("\n" if candidates else ""), encoding="utf-8")
