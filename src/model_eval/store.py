"""Persist benchmark results to disk so the dashboard (and reruns) can read them.

Results live as JSON under ``results/results.json``. Saving MERGES by
(task, model, backend): re-running one model updates just that row while older
comparisons stay available, so the dashboard always sees the latest of each.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import Result
from .config import REPO_ROOT

RESULTS_PATH = REPO_ROOT / "results" / "results.json"


def _key(record: dict) -> tuple:
    """Identity of a result row for merge purposes."""
    return (record.get("task"), record.get("model"), record.get("backend"))


def load_results(path: Path = RESULTS_PATH) -> list[dict]:
    """Return the stored result records, or an empty list if none exist yet."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both the wrapped payload and a bare list, for forward safety.
    if isinstance(data, dict):
        return data.get("results", [])
    return data


def save_results(results, path: Path = RESULTS_PATH, merge: bool = True) -> Path:
    """Write `results` (a list of `Result` or dicts) to `path`.

    When `merge` is True (default) existing rows are kept and updated by
    (task, model, backend); when False the file is overwritten.
    """
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_records = [r.to_dict() if isinstance(r, Result) else dict(r) for r in results]
    for rec in new_records:
        rec.setdefault("timestamp", stamp)

    existing = load_results(path) if merge else []
    by_key = {_key(r): r for r in existing}
    for rec in new_records:
        by_key[_key(rec)] = rec

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"saved_at": stamp, "results": list(by_key.values())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
