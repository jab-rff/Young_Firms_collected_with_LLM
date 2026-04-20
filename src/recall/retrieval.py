"""Local retrieval placeholders for raw source inputs."""

from pathlib import Path


def list_raw_inputs(raw_dir: Path) -> list[Path]:
    """Return files available for local retrieval."""
    return sorted(path for path in raw_dir.iterdir() if path.is_file())
