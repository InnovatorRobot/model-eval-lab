"""Console reporting for benchmark results.

This is intentionally separate from the engine (`benchmark.run_benchmark`) so a
web dashboard can consume the same `Result` list without the text-table code.
"""

from __future__ import annotations

from collections import defaultdict

from .benchmark import Result
from .tasks import is_generation_task

# Stable, readable ordering for any metric columns we encounter.
_METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "wer", "bleu", "rougeL"]


def _order_key(name: str) -> tuple[int, str]:
    index = _METRIC_ORDER.index(name) if name in _METRIC_ORDER else len(_METRIC_ORDER)
    return (index, name)


def print_report(results: list[Result]) -> None:
    """Print results grouped by task, so only comparable models sit together."""
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        groups[r.task].append(r)

    for task, rows in groups.items():
        metric_names = sorted({name for r in rows for name in r.metrics}, key=_order_key)

        header = f"{'model':55} {'latency (ms)':>14} {'memory (MB)':>13}"
        header += "".join(f"{name:>9}" for name in metric_names)
        print(f"\n===== task: {task}  ({len(rows)} model(s)) =====")
        print(header)
        print("-" * len(header))

        # Generation tasks rank by WER (lower better); others by latency.
        sort_key = (
            (lambda x: x.metrics.get("wer", float("inf")))
            if is_generation_task(task)
            else (lambda x: x.avg_latency_ms)
        )
        for r in sorted(rows, key=sort_key):
            line = f"{r.model:55} {r.avg_latency_ms:14.2f} {r.model_memory_mb:13.1f}"
            line += "".join(
                f"{r.metrics[name]:9.3f}" if name in r.metrics else f"{'n/a':>9}"
                for name in metric_names
            )
            print(line)
