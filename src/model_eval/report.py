"""Console table of results, grouped by task."""

from __future__ import annotations

from collections import defaultdict

from .benchmark import Result

_METRIC_ORDER = ("accuracy", "precision", "recall", "f1", "wer", "bleu", "rougeL")


def _order(name: str) -> tuple[int, str]:
    return (_METRIC_ORDER.index(name) if name in _METRIC_ORDER else len(_METRIC_ORDER), name)


def print_report(results: list[Result]) -> None:
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        groups[r.task].append(r)

    for task, rows in groups.items():
        metrics = sorted({m for r in rows for m in r.metrics}, key=_order)
        header = f"{'model':45} {'backend':12} {'latency (ms)':>14} {'memory (MB)':>13}"
        header += "".join(f"{m:>9}" for m in metrics)
        print(f"\n===== task: {task}  ({len(rows)} model(s)) =====")
        print(header)
        print("-" * len(header))

        # Generation tasks (WER present) rank by WER; others by latency.
        rank = (
            (lambda r: r.metrics.get("wer", float("inf")))
            if "wer" in metrics
            else (lambda r: r.avg_latency_ms)
        )
        for r in sorted(rows, key=rank):
            line = f"{r.model:45} {r.backend:12} {r.avg_latency_ms:14.2f} {r.model_memory_mb:13.1f}"
            line += "".join(
                f"{r.metrics[m]:9.3f}" if m in r.metrics else f"{'n/a':>9}" for m in metrics
            )
            print(line)
