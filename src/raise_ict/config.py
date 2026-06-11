"""Configuration helpers for RAISE-ICT scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty files."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def repo_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a path relative to the repository root or a supplied root."""
    base = Path(root) if root is not None else Path.cwd()
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate

