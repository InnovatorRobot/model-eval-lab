"""Load model inputs and references from local CSVs or Hugging Face datasets."""

from __future__ import annotations

import csv
from pathlib import Path

from .config import REPO_ROOT
from .tasks import TaskSpec

# Input columns are picked per modality so a `text` reference (captions, ASR
# transcriptions) is never mistaken for the input of an image/audio task.
_INPUT_COLUMNS = {
    "text": ("input", "text", "sentence", "prompt"),
    "image": ("image", "input", "path", "file", "img"),
    "audio": ("audio", "input", "path", "file"),
}
_REFERENCE_COLUMNS = (
    "label",
    "reference",
    "target",
    "caption",
    "transcription",
    "text",
    "sentence",
)


def resolve_input(value: str, modality: str) -> str:
    """Text passes through; image/audio become repo-relative file paths or URLs."""
    if modality == "text":
        return value
    text = str(value)
    if text.startswith(("http://", "https://")):
        return text
    path = Path(text)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _pick(columns, candidates) -> str | None:
    available = set(columns or [])
    return next((c for c in candidates if c in available), None)


def _select_columns(columns, modality: str) -> tuple[str | None, str | None]:
    input_col = _pick(columns, _INPUT_COLUMNS.get(modality, _INPUT_COLUMNS["text"]))
    ref_col = _pick(columns, [c for c in _REFERENCE_COLUMNS if c != input_col])
    return input_col, ref_col


def _load_csv(config: dict, task: TaskSpec) -> tuple[list, list[str]]:
    rel_dir = config.get("data_dir", {}).get(task.name)
    csv_path = REPO_ROOT / rel_dir / "data.csv" if rel_dir else None
    if not csv_path or not csv_path.exists():
        return [], []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_col, ref_col = _select_columns(reader.fieldnames, task.modality)
        if not input_col or not ref_col:
            return [], []
        rows = [(resolve_input(r[input_col], task.modality), r[ref_col]) for r in reader]
    inputs, refs = map(list, zip(*rows)) if rows else ([], [])
    return list(inputs), list(refs)


def _hf_input(value, modality: str):
    if modality == "audio" and isinstance(value, dict) and "array" in value:
        return {"raw": value["array"], "sampling_rate": value["sampling_rate"]}
    return value


def _hf_reference(value, label_map, feature) -> str:
    if label_map and value in label_map:
        return str(label_map[value])
    names = getattr(feature, "names", None)
    if isinstance(value, int) and names and 0 <= value < len(names):
        return str(names[value])
    return str(value)


def _load_hf(config: dict, task: TaskSpec) -> tuple[list, list[str]]:
    spec = config.get("hf_datasets", {}).get(task.name)
    if not spec:
        return [], []
    from datasets import load_dataset

    dataset = load_dataset(spec["name"], spec.get("config"), split=spec.get("split", "test"))
    auto_in, auto_ref = _select_columns(dataset.column_names, task.modality)
    input_col = spec.get("input_column") or auto_in
    ref_col = spec.get("reference_column") or auto_ref
    if not input_col or not ref_col:
        return [], []
    label_map, feature = spec.get("label_map"), dataset.features.get(ref_col)
    inputs = [_hf_input(row[input_col], task.modality) for row in dataset]
    refs = [_hf_reference(row[ref_col], label_map, feature) for row in dataset]
    return inputs, refs


def load_eval_data(config: dict, task: TaskSpec) -> tuple[list, list[str]]:
    """Eval (inputs, references): a configured HF dataset wins over the local CSV."""
    inputs, refs = _load_hf(config, task)
    return (inputs, refs) if inputs else _load_csv(config, task)


def sample_inputs(config: dict, task: TaskSpec) -> list:
    """Inputs for the latency pass: sample_inputs override, else eval data, else texts."""
    override = config.get("sample_inputs", {}).get(task.name)
    if override:
        return [resolve_input(v, task.modality) for v in override]
    inputs, _ = load_eval_data(config, task)
    if inputs:
        return inputs
    if task.modality == "text" and "sample_texts" in config:
        return list(config["sample_texts"])
    return []
