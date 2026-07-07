"""Model loading, memory measurement, and prediction.

This module is the only place that touches `transformers`/`huggingface_hub`,
keeping the heavy ML dependency surface in one spot.
"""

from __future__ import annotations

import psutil
from huggingface_hub import model_info
from transformers import pipeline

from .data import DEFAULT_BACKEND
from .metrics import normalize_label
from .tasks import is_generation_task

# Some pipeline task identifiers were renamed across transformers versions.
# Map our stable task names to alternatives, and pick whichever the installed
# transformers actually supports at build time.
_PIPELINE_TASK_ALIASES = {
    # transformers v5 renamed image captioning's task identifier.
    "image-to-text": "image-text-to-text",
}


def _resolve_pipeline_task(task: str) -> str:
    """Map our task name to one the installed transformers supports."""
    try:
        from transformers.pipelines import PIPELINE_REGISTRY

        supported = set(PIPELINE_REGISTRY.get_supported_tasks())
    except Exception:
        supported = set()

    if supported and task not in supported:
        alias = _PIPELINE_TASK_ALIASES.get(task)
        if alias and alias in supported:
            return alias
    return task


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


def _build_pytorch(model_name: str, task: str):
    """Standard transformers pipeline on the PyTorch runtime."""
    return pipeline(_resolve_pipeline_task(task), model=model_name)


def _build_onnxruntime(model_name: str, task: str):
    """ONNX Runtime pipeline via Optimum (exports the model to ONNX on load)."""
    try:
        from optimum.pipelines import pipeline as ort_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime backend requires Optimum: " "pip install 'optimum[onnxruntime]'"
        ) from exc
    return ort_pipeline(
        _resolve_pipeline_task(task),
        model=model_name,
        accelerator="ort",
    )


def _build_openvino(model_name: str, task: str):
    """OpenVINO pipeline via Optimum Intel (exports to OpenVINO IR on load)."""
    try:
        from optimum.intel import OVModelForFeatureExtraction  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "openvino backend requires Optimum Intel: " "pip install 'optimum[openvino]'"
        ) from exc
    # Optimum Intel picks the right OVModelFor<Task> class when export=True.
    from optimum.intel.pipelines import pipeline as ov_pipeline

    return ov_pipeline(_resolve_pipeline_task(task), model=model_name)


# Backend name -> builder. Add new runtimes here; the rest of the engine is
# backend-agnostic (it just gets a callable pipeline back).
_BACKEND_BUILDERS = {
    "pytorch": _build_pytorch,
    "onnxruntime": _build_onnxruntime,
    "openvino": _build_openvino,
}


def build_pipeline_with_memory(model_name: str, task: str, backend: str = DEFAULT_BACKEND):
    """Build the pipeline ONCE on `backend` and measure the RSS it adds.

    We snapshot the process's resident memory (RSS) before and after loading
    the model. The delta approximates the model's in-memory footprint.

    `task` is taken from the YAML config (the group the model is listed under),
    so we don't need a Hub round-trip just to build the pipeline. `backend`
    selects the runtime (pytorch / onnxruntime / openvino).

    Caveat: with the `inline` launcher all models load into the SAME process,
    and Python rarely returns freed memory to the OS, so this is an
    approximation. Use the `process` launcher (see benchmark.run_benchmark) for
    a clean per-model baseline via a fresh subprocess.
    """
    builder = _BACKEND_BUILDERS.get(backend)
    if builder is None:
        raise ValueError(
            f"Unknown backend '{backend}'. Known backends: {sorted(_BACKEND_BUILDERS)}"
        )

    process = psutil.Process()
    rss_before = process.memory_info().rss
    clf = builder(model_name, task)
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
