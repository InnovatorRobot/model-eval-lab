"""Metric computation and label normalization.

`compute_classification_metrics` and `compute_generation_metrics` both return a
plain ``dict[str, float]`` so the benchmark can treat metrics uniformly across
modalities (and a web layer can serialize them directly).
"""

from __future__ import annotations

import statistics

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def normalize_label(label: str) -> str:
    """Normalize a label so predictions and gold labels are comparable.

    Different models phrase the same class differently (e.g. ``POSITIVE`` vs
    ``positive``), so we lowercase and strip surrounding whitespace.
    """
    return label.strip().lower()


def compute_classification_metrics(preds: list[str], gold: list[str]) -> dict[str, float]:
    """accuracy + macro precision/recall/F1 for classification tasks.

    `preds` are expected to be already normalized (see `normalize_label`).
    """
    gold = [normalize_label(label) for label in gold]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_generation_metrics(preds: list[str], refs: list[str]) -> dict[str, float]:
    """WER (lower is better), BLEU (0-100), and ROUGE-L F1 for generation tasks.

    Heavy metric libraries are imported lazily so text/vision-only runs stay
    light.
    """
    import jiwer
    import sacrebleu
    from rouge_score import rouge_scorer

    hyps = [p.strip().lower() for p in preds]
    golds = [r.strip().lower() for r in refs]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = statistics.mean(
        scorer.score(gold, hyp)["rougeL"].fmeasure for gold, hyp in zip(golds, hyps)
    )

    return {
        "wer": float(jiwer.wer(golds, hyps)),
        "bleu": float(sacrebleu.corpus_bleu(hyps, [golds]).score),
        "rougeL": float(rouge_l),
    }
