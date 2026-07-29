"""Parse the `models:` config into ModelSpec units and iterate/select them."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

DEFAULT_BACKEND = "pytorch"


@dataclass(frozen=True)
class ModelSpec:
    """One benchmarkable unit: a model on a task, running on a backend."""

    task: str
    model: str
    backend: str = DEFAULT_BACKEND


def _entry_specs(task: str, entry) -> list[ModelSpec]:
    """Expand one YAML entry (str or {model, backend(s)}) into ModelSpecs."""
    if isinstance(entry, str):
        return [ModelSpec(task, entry)]
    if isinstance(entry, dict) and entry.get("model"):
        backends = entry.get("backends") or [entry.get("backend", DEFAULT_BACKEND)]
        return [ModelSpec(task, entry["model"], b) for b in backends]
    return []


def _matches(name: str, selectors: list[str]) -> bool:
    low = name.lower()
    return any(s == low or s in low for s in selectors)


def iter_models(config: dict, only: list[str] | None = None) -> Iterator[ModelSpec]:
    """Yield a ModelSpec per (model, backend), optionally filtered by `only`."""
    selectors = [s.lower() for s in only] if only else None
    for group in config["models"]:
        for task, entries in group.items():
            for entry in entries:
                for spec in _entry_specs(task, entry):
                    if selectors is None or _matches(spec.model, selectors):
                        yield spec
