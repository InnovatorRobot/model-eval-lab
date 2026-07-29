"""Metrics registered per task kind; each returns a plain dict[str, float]."""

from __future__ import annotations

import statistics
from collections.abc import Callable

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

Metric = Callable[[list[str], list[str]], dict[str, float]]
_REGISTRY: dict[str, Metric] = {}


def register(kind: str) -> Callable[[Metric], Metric]:
    def decorate(fn: Metric) -> Metric:
        _REGISTRY[kind] = fn
        return fn

    return decorate


def normalize_label(label: str) -> str:
    return label.strip().lower()


def compute_metrics(kind: str, preds: list[str], refs: list[str]) -> dict[str, float]:
    fn = _REGISTRY.get(kind)
    return fn(preds, refs) if fn else {}


@register("classification")
def _classification(preds: list[str], refs: list[str]) -> dict[str, float]:
    gold = [normalize_label(r) for r in refs]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@register("generation")
def _generation(preds: list[str], refs: list[str]) -> dict[str, float]:
    import jiwer
    import sacrebleu
    from rouge_score import rouge_scorer

    hyps = [p.strip().lower() for p in preds]
    golds = [r.strip().lower() for r in refs]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = statistics.mean(scorer.score(g, h)["rougeL"].fmeasure for g, h in zip(golds, hyps))
    return {
        "wer": float(jiwer.wer(golds, hyps)),
        "bleu": float(sacrebleu.corpus_bleu(hyps, [golds]).score),
        "rougeL": float(rouge_l),
    }
