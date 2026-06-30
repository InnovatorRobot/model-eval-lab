"""Input resolution and dataset / sample loading.

Everything that turns the YAML config + CSV files into concrete model inputs
lives here, so the model-running code stays focused on inference.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .config import REPO_ROOT
from .tasks import input_modality

# Candidate CSV/dataset column names. Inputs are chosen per modality so that a
# reference column such as `text` (ASR transcription, captions) is never
# mistaken for the input of an image/audio task.
_INPUT_COLUMNS_BY_MODALITY = {
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
    """Resolve one raw input value for the given modality.

    Text is returned unchanged. Image/audio values are treated as file paths
    (resolved relative to the repo root) or passed through if they are URLs,
    so the pipeline can load them directly.
    """
    if modality == "text":
        return value
    text = str(value)
    if text.startswith(("http://", "https://")):
        return text
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _pick_column(fieldnames, candidates) -> str | None:
    """Return the first candidate column that exists in the CSV header."""
    available = set(fieldnames or [])
    for name in candidates:
        if name in available:
            return name
    return None


def _select_columns(fieldnames, modality: str) -> tuple[str | None, str | None]:
    """Choose (input_col, reference_col) for a modality from a dataset header.

    The input column is picked from modality-specific candidates; the reference
    column is then picked from the remaining candidates (so the input column is
    never also used as the reference).
    """
    input_candidates = _INPUT_COLUMNS_BY_MODALITY.get(modality, _INPUT_COLUMNS_BY_MODALITY["text"])
    input_col = _pick_column(fieldnames, input_candidates)
    ref_candidates = [c for c in _REFERENCE_COLUMNS if c != input_col]
    ref_col = _pick_column(fieldnames, ref_candidates)
    return input_col, ref_col


def load_dataset_rows(config: dict, task: str) -> tuple[list[str], list[str]]:
    """Load (inputs, references) for a task from its `data_dir/data.csv`.

    Inputs are resolved for the task's modality (file paths for image/audio).
    References are class labels (classification) or reference text (generation).
    Returns empty lists if the task has no dataset, so the benchmark can still
    run the task-agnostic latency/memory metrics.
    """
    data_dirs = config.get("data_dir", {})
    rel_dir = data_dirs.get(task)
    if not rel_dir:
        return [], []

    csv_path = REPO_ROOT / rel_dir / "data.csv"
    if not csv_path.exists():
        return [], []

    modality = input_modality(task)
    inputs: list[str] = []
    references: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_col, ref_col = _select_columns(reader.fieldnames, modality)
        if input_col is None or ref_col is None:
            return [], []
        for row in reader:
            inputs.append(resolve_input(row[input_col], modality))
            references.append(row[ref_col])
    return inputs, references


def _resolve_hf_input(value, modality: str):
    """Adapt one Hugging Face dataset cell into something a pipeline accepts.

    Text passes through as a string and images pass through as PIL objects.
    Audio cells arrive as ``{"array", "sampling_rate", "path"}``; transformers
    pipelines expect ``{"raw", "sampling_rate"}``, so we remap those keys.
    """
    if modality == "audio" and isinstance(value, dict) and "array" in value:
        return {"raw": value["array"], "sampling_rate": value["sampling_rate"]}
    return value


def _resolve_reference(value, label_map: dict | None, feature) -> str:
    """Turn a raw reference cell into a comparable string label/text.

    Integer class labels are mapped to names via an explicit ``label_map`` or
    the dataset's ``ClassLabel`` feature names when available.
    """
    if label_map and value in label_map:
        return str(label_map[value])
    if isinstance(value, int):
        names = getattr(feature, "names", None)
        if names and 0 <= value < len(names):
            return str(names[value])
    return str(value)


def load_hf_dataset_rows(config: dict, task: str) -> tuple[list, list[str]]:
    """Load (inputs, references) for a task from a Hugging Face Hub dataset.

    Config shape (under a top-level ``hf_datasets`` mapping):
        hf_datasets:
          sentiment-analysis:
            name: glue
            config: sst2          # optional dataset config/subset
            split: validation[:50]
            input_column: sentence     # optional; auto-detected otherwise
            reference_column: label    # optional; auto-detected otherwise
            label_map: {0: negative, 1: positive}   # optional

    Returns empty lists if the task has no `hf_datasets` entry.
    """
    specs = config.get("hf_datasets", {})
    spec = specs.get(task)
    if not spec:
        return [], []

    from datasets import load_dataset

    dataset = load_dataset(
        spec["name"],
        spec.get("config"),
        split=spec.get("split", "test"),
    )

    columns = dataset.column_names
    modality = input_modality(task)
    auto_input, auto_ref = _select_columns(columns, modality)
    input_col = spec.get("input_column") or auto_input
    ref_col = spec.get("reference_column") or auto_ref
    if input_col is None or ref_col is None:
        return [], []

    label_map = spec.get("label_map")
    feature = dataset.features.get(ref_col)

    inputs: list = []
    references: list[str] = []
    for row in dataset:
        inputs.append(_resolve_hf_input(row[input_col], modality))
        references.append(_resolve_reference(row[ref_col], label_map, feature))
    return inputs, references


def load_eval_data(config: dict, task: str) -> tuple[list, list[str]]:
    """Load (inputs, references) for a task, preferring a configured HF dataset.

    Falls back to the local ``data_dir/<task>/data.csv`` when no Hugging Face
    dataset is configured for the task.
    """
    inputs, references = load_hf_dataset_rows(config, task)
    if inputs:
        return inputs, references
    return load_dataset_rows(config, task)


def sample_inputs_for(config: dict, task: str) -> list[str]:
    """Pick the inputs used for the latency/memory pass.

    Preference order: a per-task ``sample_inputs`` override, then the task's
    eval dataset (local CSV or HF), then the legacy global ``sample_texts``
    (text tasks only).
    """
    modality = input_modality(task)
    overrides = config.get("sample_inputs", {})
    if task in overrides:
        return [resolve_input(v, modality) for v in overrides[task]]

    dataset_inputs, _ = load_eval_data(config, task)
    if dataset_inputs:
        return dataset_inputs

    if modality == "text" and "sample_texts" in config:
        return list(config["sample_texts"])

    return []


def iter_models(config: dict):
    """Yield (task, model_name) pairs from the task-grouped YAML.

    Config shape:
        models:
          - sentiment-analysis:
              - model-a
              - model-b
    """
    for group in config["models"]:
        for task, model_names in group.items():
            for model_name in model_names:
                yield task, model_name
