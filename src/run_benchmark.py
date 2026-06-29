"""Generic model benchmark.

Reads a list of models from config/models.yaml, auto-detects each model's
task from the Hugging Face Hub, groups models that share a task, and measures
TASK-AGNOSTIC metrics (latency + memory) for every model. Models in the same
group are directly comparable.

Run with:  .venv/bin/python src/run_benchmark.py

Accuracy / F1 (task-SPECIFIC metrics) are added in the next slice, because
they require labeled data per task plus label normalization.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import psutil
import yaml
from huggingface_hub import model_info
from transformers import pipeline

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"


@dataclass
class Result:
    """One model's measurements. We'll add accuracy/f1 fields next slice."""

    model: str
    task: str
    avg_latency_ms: float = 0.0
    model_memory_mb: float = 0.0


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def measure_memory_and_build(model_name: str, task: str):
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


def measure_latency(clf, texts: list[str], runs: int = 20) -> float:
    """Average per-inference latency in MILLISECONDS.

    TODO (your turn): fix the two classic bugs from your first attempt.
      1. WARM-UP: the very first inference includes lazy setup and is far
         slower than steady state. Do ONE throwaway call before timing.
      2. TIMER: use time.perf_counter() (high-resolution, monotonic), NOT
         time.time(). Time each call individually and average all of them.

    Steps:
      - Do a warm-up:           clf(texts[0])
      - For `runs` iterations, for each text in `texts`:
            start = time.perf_counter()
            clf(text)
            record (time.perf_counter() - start)
      - Return the mean of all recorded timings, converted to milliseconds
        (multiply seconds by 1000).
    """
    timings: list[float] = []  # seconds per single inference

    # --- replace the body below with your implementation ---
    raise NotImplementedError("Implement measure_latency (see the TODO above).")
    # return statistics.mean(timings) * 1000.0


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


def run_benchmark() -> list[Result]:
    config = load_config()
    texts = config["sample_texts"]
    runs = config.get("latency_runs", 20)

    results: list[Result] = []
    for task, model_name in iter_models(config):
        print(f"\nLoading {model_name}  (task: {task}) ...")
        clf, memory_mb = measure_memory_and_build(model_name, task)
        latency_ms = measure_latency(clf, texts, runs=runs)
        results.append(
            Result(
                model=model_name,
                task=task,
                avg_latency_ms=latency_ms,
                model_memory_mb=memory_mb,
            )
        )
    return results


def print_report(results: list[Result]) -> None:
    """Print results grouped by task, so only comparable models sit together."""
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        groups[r.task].append(r)

    for task, rows in groups.items():
        print(f"\n===== task: {task}  ({len(rows)} model(s)) =====")
        print(f"{'model':55} {'latency (ms)':>14} {'memory (MB)':>13}")
        print("-" * 84)
        for r in sorted(rows, key=lambda x: x.avg_latency_ms):
            print(f"{r.model:55} {r.avg_latency_ms:14.2f} {r.model_memory_mb:13.1f}")


def main() -> None:
    results = run_benchmark()
    print_report(results)


if __name__ == "__main__":
    main()
