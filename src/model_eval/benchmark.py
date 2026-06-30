"""Benchmark orchestration.

`run_benchmark` is the engine entrypoint: it returns a list of `Result` rows
that a console reporter (see report.py) or a web dashboard can render. `Result`
is a plain dataclass with a `to_dict` helper so it serializes cleanly to JSON.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field

from .config import load_config
from .data import iter_models, load_eval_data, sample_inputs_for
from .metrics import compute_classification_metrics, compute_generation_metrics
from .pipelines import build_pipeline_with_memory, predict
from .tasks import is_generation_task


@dataclass
class Result:
    """One model's measurements.

    `metrics` holds whatever task-specific scores were computed (e.g.
    ``accuracy``/``f1`` for classification, ``wer``/``bleu``/``rougeL`` for
    generation). Keeping it a dict lets a single report table span modalities.
    """

    model: str
    task: str
    avg_latency_ms: float = 0.0
    model_memory_mb: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable representation (handy for a web dashboard)."""
        return asdict(self)


def measure_latency(clf, inputs: list, runs: int = 20) -> float:
    """Average per-inference latency in MILLISECONDS.

    `inputs` may be text strings, image paths, or audio paths -- whatever the
    pipeline accepts for this task. Steps:
      - Do a warm-up:           clf(inputs[0])
      - For `runs` iterations, for each input, time a single clf(input) call.
      - Return the mean of all timings, converted to milliseconds.
    """
    clf(inputs[0])  # warm-up
    timings: list[float] = []  # seconds per single inference

    for _ in range(runs):
        for item in inputs:
            start = time.perf_counter()
            clf(item)
            timings.append(time.perf_counter() - start)

    return statistics.mean(timings) * 1000.0


def run_benchmark(config: dict | None = None) -> list[Result]:
    """Run every configured model and collect a `Result` per model.

    Pass a `config` dict to benchmark a custom selection (e.g. from a web
    request); otherwise the on-disk config is loaded.
    """
    config = config if config is not None else load_config()
    runs = config.get("latency_runs", 20)

    results: list[Result] = []
    for task, model_name in iter_models(config):
        print(f"\nLoading {model_name}  (task: {task}) ...")

        inputs = sample_inputs_for(config, task)
        if not inputs:
            print(f"  skipped: no sample inputs configured for task '{task}'.")
            continue

        clf, memory_mb = build_pipeline_with_memory(model_name, task)
        latency_ms = measure_latency(clf, inputs, runs=runs)

        metrics: dict[str, float] = {}
        eval_inputs, eval_refs = load_eval_data(config, task)
        if eval_inputs:
            preds = predict(clf, eval_inputs, task)
            if is_generation_task(task):
                metrics = compute_generation_metrics(preds, eval_refs)
            else:
                metrics = compute_classification_metrics(preds, eval_refs)

        results.append(
            Result(
                model=model_name,
                task=task,
                avg_latency_ms=latency_ms,
                model_memory_mb=memory_mb,
                metrics=metrics,
            )
        )
    return results
