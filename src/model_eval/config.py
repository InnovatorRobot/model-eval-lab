"""Repository paths and configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

# src/model_eval/config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG_PATH = REPO_ROOT / "config" / "models.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
