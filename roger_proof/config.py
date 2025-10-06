"""Configuration helpers for Level 2 runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    """Load a configuration file in JSON format."""

    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported configuration format: {path}")
    text = path.read_text(encoding="utf-8")
    return json.loads(text)
