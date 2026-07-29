"""model_eval: compare models on quality and speed, config-driven.

Modules: config (paths/YAML) · tasks (modality+kind) · models (specs/selection) ·
data (input/dataset loading) · pipelines (backends + predict) · metrics (by kind) ·
benchmark (run + Result) · report (console) · store (persist results).
"""

from __future__ import annotations

from .benchmark import Result, measure_latency, run_benchmark
from .config import load_config
from .models import ModelSpec, iter_models
from .report import print_report
from .store import RESULTS_PATH, load_results, save_results
from .tasks import TaskSpec, resolve_task

__all__ = [
    "Result",
    "run_benchmark",
    "measure_latency",
    "ModelSpec",
    "iter_models",
    "TaskSpec",
    "resolve_task",
    "load_config",
    "print_report",
    "load_results",
    "save_results",
    "RESULTS_PATH",
]
