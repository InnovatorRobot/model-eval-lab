"""Model loading, memory measurement, and prediction.

This module is the only place that touches `transformers`/`huggingface_hub`,
keeping the heavy ML dependency surface in one spot.
"""

from __future__ import annotations

import psutil
from huggingface_hub import model_info
from transformers import pipeline

from .metrics import normalize_label
from .tasks import is_generation_task


def detect_task(model_name: str) -> str:
    """Auto-detect a model's task (its Hub `pipeline_tag`).

    Falls back to letting transformers infer the task from the model's
    architecture if the Hub doesn't declare a pipeline_tag.
    """
    try:
        tag = model_info(model_name).pipeline_tag
        if tag:
            return tag
    except Exception:
        pass
    # Fallback: build a default pipeline (no task given) and read what it chose.
    return pipeline(model=model_name).task


def build_pipeline_with_memory(model_name: str, task: str):
    """Build the pipeline ONCE and measure how much RSS memory it adds.

    We snapshot the process's resident memory (RSS) before and after loading
    the model. The delta approximates the model's in-memory footprint.

    `task` is taken from the YAML config (the group the model is listed under),
    so we don't need a Hub round-trip just to build the pipeline.

    Caveat (note this in NOTES.md): all models load into the SAME process, and
    Python rarely returns freed memory to the OS, so this is an approximation,
    not a clean per-model number. A subprocess-per-model would be more precise.
    """
    process = psutil.Process()
    rss_before = process.memory_info().rss
    clf = pipeline(task, model=model_name)
    rss_after = process.memory_info().rss
    memory_mb = (rss_after - rss_before) / (1024 * 1024)
    return clf, memory_mb


def _first(pred):
    """Pipelines may return a single dict or a top-k list; take the top item."""
    return pred[0] if isinstance(pred, list) else pred


def _extract_label(pred) -> str:
    """Read a normalized class label from one classification prediction."""
    return normalize_label(_first(pred)["label"])


def _extract_text(pred) -> str:
    """Read generated text from one generation prediction (caption / ASR)."""
    item = _first(pred)
    for key in ("generated_text", "text"):
        if key in item:
            return item[key]
    return str(item)


def predict(clf, inputs: list, task: str) -> list[str]:
    """Run the pipeline over `inputs` and return one string per input.

    Returns normalized class labels for classification tasks, or generated
    text for generation tasks.
    """
    raw = clf(inputs)
    extract = _extract_text if is_generation_task(task) else _extract_label
    return [extract(r) for r in raw]
