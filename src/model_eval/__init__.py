"""model_eval: a multi-modal model comparison toolkit.

The package is split into focused modules so a future web dashboard can import
the engine (`run_benchmark` -> list of `Result`) without dragging in the console
reporting layer:

    config.py     paths + YAML loading
    tasks.py      task taxonomy (input modality, generation vs. classification)
    data.py       input resolution, dataset/sample loading, model iteration
    pipelines.py  model loading, memory measurement, prediction
    metrics.py    classification + generation metric computation
    benchmark.py  Result dataclass + run_benchmark orchestration
    report.py     console reporting
"""

from __future__ import annotations

from .benchmark import Result, measure_latency, run_benchmark
from .config import load_config
from .report import print_report

__all__ = [
    "Result",
    "measure_latency",
    "run_benchmark",
    "load_config",
    "print_report",
]
