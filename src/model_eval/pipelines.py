"""Model execution: build a pipeline on a backend and run predictions.

The only module that imports transformers/optimum.
"""

from __future__ import annotations

import psutil
from transformers import pipeline

from .metrics import normalize_label
from .models import DEFAULT_BACKEND

# transformers renamed some task ids across versions (e.g. image-to-text).
_TASK_ALIASES = {"image-to-text": "image-text-to-text"}


def _pipeline_task(task: str) -> str:
    try:
        from transformers.pipelines import PIPELINE_REGISTRY

        supported = set(PIPELINE_REGISTRY.get_supported_tasks())
    except Exception:
        supported = set()
    alias = _TASK_ALIASES.get(task)
    return alias if supported and task not in supported and alias in supported else task


def _pytorch(model: str, task: str):
    return pipeline(_pipeline_task(task), model=model)


def _onnxruntime(model: str, task: str):
    try:
        from optimum.pipelines import pipeline as ort_pipeline
    except ImportError as exc:
        raise RuntimeError("onnxruntime backend needs: pip install 'optimum[onnxruntime]'") from exc
    return ort_pipeline(_pipeline_task(task), model=model, accelerator="ort")


def _openvino(model: str, task: str):
    try:
        from optimum.intel.pipelines import pipeline as ov_pipeline
    except ImportError as exc:
        raise RuntimeError("openvino backend needs: pip install 'optimum[openvino]'") from exc
    return ov_pipeline(_pipeline_task(task), model=model)


# backend name -> builder. Add a runtime here; the rest of the engine is agnostic.
BACKENDS = {"pytorch": _pytorch, "onnxruntime": _onnxruntime, "openvino": _openvino}


def build_pipeline(model: str, task: str, backend: str = DEFAULT_BACKEND):
    """Build the pipeline once and return (pipeline, added RSS in MB)."""
    builder = BACKENDS.get(backend)
    if builder is None:
        raise ValueError(f"Unknown backend '{backend}'. Known: {sorted(BACKENDS)}")
    process = psutil.Process()
    before = process.memory_info().rss
    clf = builder(model, task)
    memory_mb = (process.memory_info().rss - before) / (1024 * 1024)
    return clf, memory_mb


def _first(pred):
    return pred[0] if isinstance(pred, list) else pred


def _label(pred) -> str:
    return normalize_label(_first(pred)["label"])


def _text(pred) -> str:
    item = _first(pred)
    return next((item[k] for k in ("generated_text", "text") if k in item), str(item))


# task kind -> how to read one prediction into a comparable string.
_EXTRACT = {"classification": _label, "generation": _text}


def predict(clf, inputs: list, kind: str) -> list[str]:
    extract = _EXTRACT.get(kind, _label)
    return [extract(r) for r in clf(inputs)]
