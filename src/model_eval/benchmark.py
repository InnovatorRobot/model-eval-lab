"""Benchmark orchestration.

`run_benchmark` is the engine entrypoint: it returns a list of `Result` rows
that a console reporter (see report.py) or a web dashboard can render. `Result`
is a plain dataclass with a `to_dict` helper so it serializes cleanly to JSON.
"""

from __future__ import annotations

import multiprocessing as mp
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
    backend: str = "pytorch"
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


def _benchmark_one(config: dict, spec, runs: int) -> Result | None:
    """Build, time, and score a single model spec. Returns None if unrunnable.

    This is the unit of work shared by both launchers: the `inline` launcher
    calls it directly; the `process` launcher runs it in a fresh subprocess so
    memory is measured against a clean baseline and crashes stay isolated.
    """
    task, model_name, backend = spec.task, spec.model, spec.backend

    inputs = sample_inputs_for(config, task)
    if not inputs:
        print(f"  skipped: no sample inputs configured for task '{task}'.")
        return None

    clf, memory_mb = build_pipeline_with_memory(model_name, task, backend)
    latency_ms = measure_latency(clf, inputs, runs=runs)

    metrics: dict[str, float] = {}
    eval_inputs, eval_refs = load_eval_data(config, task)
    if eval_inputs:
        preds = predict(clf, eval_inputs, task)
        if is_generation_task(task):
            metrics = compute_generation_metrics(preds, eval_refs)
        else:
            metrics = compute_classification_metrics(preds, eval_refs)

    return Result(
        model=model_name,
        task=task,
        backend=backend,
        avg_latency_ms=latency_ms,
        model_memory_mb=memory_mb,
        metrics=metrics,
    )


def _subprocess_worker(config: dict, spec, runs: int, queue) -> None:
    """Child-process entrypoint: run one spec and ship the outcome back."""
    try:
        result = _benchmark_one(config, spec, runs)
        queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001 - reported to the parent as a skip
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _benchmark_one_in_subprocess(config: dict, spec, runs: int):
    """Run `_benchmark_one` in a fresh spawned process.

    Returns (result, error_message). A clean interpreter per model gives an
    accurate memory baseline, and a hard crash (e.g. segfault) surfaces as a
    non-zero exit code instead of taking down the whole benchmark.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_subprocess_worker, args=(config, spec, runs, queue))
    proc.start()
    proc.join()

    if not queue.empty():
        status, payload = queue.get()
        if status == "ok":
            return payload, None
        return None, payload

    # Nothing on the queue: the child died before reporting (crash/OOM).
    return None, f"process exited with code {proc.exitcode} before returning a result"


def run_benchmark(
    config: dict | None = None,
    models: list[str] | None = None,
    launcher: str | None = None,
) -> list[Result]:
    """Run the configured models and collect a `Result` per model.

    Pass a `config` dict to benchmark a custom selection (e.g. from a web
    request); otherwise the on-disk config is loaded.

    Pass `models` to compare only a chosen subset -- each entry is matched
    against configured model names by exact name or case-insensitive substring
    (e.g. ``["distilbert", "cardiffnlp"]``). Omit it to run every model.

    `launcher` selects how each model runs:
      * ``inline``  (default) - in this process; fast, shared memory space.
      * ``process`` - each model in a fresh spawned subprocess, giving a clean
        memory baseline and isolating crashes from the rest of the run.
    It falls back to the config's ``launcher`` key, then ``inline``.
    """
    config = config if config is not None else load_config()
    runs = config.get("latency_runs", 20)
    launcher = launcher or config.get("launcher", "inline")

    pairs = list(iter_models(config, only=models))
    if models and not pairs:
        print(f"No configured models matched selection: {models}")
        return []

    results: list[Result] = []
    for spec in pairs:
        print(f"\nLoading {spec.model} [{spec.backend}]  (task: {spec.task}) ...")

        if launcher == "process":
            result, error = _benchmark_one_in_subprocess(config, spec, runs)
            if error:
                print(f"  skipped: {error}")
        else:
            try:
                result = _benchmark_one(config, spec, runs)
            except Exception as exc:  # noqa: BLE001 - keep comparing the others
                print(f"  skipped: {type(exc).__name__}: {exc}")
                result = None

        if result is not None:
            results.append(result)

    return results
