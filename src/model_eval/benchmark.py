"""Run benchmarks: build, time, and score each model spec, inline or per-process."""

from __future__ import annotations

import multiprocessing as mp
import statistics
import time
from dataclasses import asdict, dataclass, field

from .config import load_config
from .data import load_eval_data, sample_inputs
from .metrics import compute_metrics
from .models import iter_models
from .pipelines import build_pipeline, predict
from .tasks import resolve_task


@dataclass
class Result:
    model: str
    task: str
    backend: str = "pytorch"
    avg_latency_ms: float = 0.0
    model_memory_mb: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def measure_latency(clf, inputs: list, runs: int = 20) -> float:
    """Mean single-inference latency in ms, after one warm-up call."""
    clf(inputs[0])
    timings: list[float] = []
    for _ in range(runs):
        for item in inputs:
            start = time.perf_counter()
            clf(item)
            timings.append(time.perf_counter() - start)
    return statistics.mean(timings) * 1000.0


def _run_one(config: dict, spec, runs: int) -> Result:
    task = resolve_task(spec.task, config)
    inputs = sample_inputs(config, task)
    if not inputs:
        raise RuntimeError(f"no sample inputs for task '{task.name}'")

    clf, memory_mb = build_pipeline(spec.model, task.name, spec.backend)
    latency_ms = measure_latency(clf, inputs, runs)

    eval_inputs, refs = load_eval_data(config, task)
    metrics = (
        compute_metrics(task.kind, predict(clf, eval_inputs, task.kind), refs)
        if eval_inputs
        else {}
    )
    return Result(spec.model, spec.task, spec.backend, latency_ms, memory_mb, metrics)


def _worker(config, spec, runs, queue) -> None:
    try:
        queue.put(("ok", _run_one(config, spec, runs)))
    except Exception as exc:  # noqa: BLE001 - reported to the parent as a skip
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_in_process(config, spec, runs):
    """Run one spec in a fresh spawned process: clean memory baseline + crash isolation."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(config, spec, runs, queue))
    proc.start()
    proc.join()
    if not queue.empty():
        status, payload = queue.get()
        return (payload, None) if status == "ok" else (None, payload)
    return None, f"process exited with code {proc.exitcode}"


def run_benchmark(
    config: dict | None = None,
    models: list[str] | None = None,
    launcher: str | None = None,
) -> list[Result]:
    """Benchmark configured models. `models` filters by name/substring; `launcher`
    is 'inline' (default) or 'process' (isolated subprocess per model)."""
    config = config if config is not None else load_config()
    runs = config.get("latency_runs", 20)
    launcher = launcher or config.get("launcher", "inline")

    specs = list(iter_models(config, only=models))
    if models and not specs:
        print(f"No configured models matched: {models}")
        return []

    results: list[Result] = []
    for spec in specs:
        print(f"\nLoading {spec.model} [{spec.backend}]  (task: {spec.task}) ...")
        if launcher == "process":
            result, error = _run_in_process(config, spec, runs)
        else:
            try:
                result, error = _run_one(config, spec, runs), None
            except Exception as exc:  # noqa: BLE001 - keep comparing the others
                result, error = None, f"{type(exc).__name__}: {exc}"
        if error:
            print(f"  skipped: {error}")
        elif result:
            results.append(result)
    return results
