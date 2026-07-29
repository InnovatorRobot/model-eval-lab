"""Model-Eval-Lab dashboard: pick models, see the quality vs. speed tradeoff.

Run with:  .venv/bin/streamlit run app.py

It reads results/results.json (written by `python src/run_benchmark.py`) and
lets you compare models head-to-head in a table, a latency-vs-quality scatter,
and a one-line plain-English verdict. No config, no ML knowledge required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the `model_eval` package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from model_eval import RESULTS_PATH, load_results  # noqa: E402

# Quality metrics where a HIGHER value is better vs. LOWER is better.
_HIGHER_BETTER = ("accuracy", "f1", "precision", "recall", "bleu", "rougeL")
_LOWER_BETTER = ("wer",)
# Preference order when auto-picking the headline quality metric.
_QUALITY_PRIORITY = ("accuracy", "f1", "rougeL", "bleu", "wer")


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten result records (metrics dict -> columns) into a DataFrame."""
    rows = []
    for rec in records:
        row = {
            "model": rec.get("model"),
            "backend": rec.get("backend", "pytorch"),
            "task": rec.get("task"),
            "latency_ms": rec.get("avg_latency_ms"),
            "memory_mb": rec.get("model_memory_mb"),
        }
        row.update(rec.get("metrics", {}))
        row["label"] = f"{row['model']} [{row['backend']}]"
        rows.append(row)
    return pd.DataFrame(rows)


def pick_quality_metric(df: pd.DataFrame) -> str | None:
    """Choose a sensible default y-axis metric from those present."""
    for name in _QUALITY_PRIORITY:
        if name in df.columns and df[name].notna().any():
            return name
    return None


def verdict(df: pd.DataFrame, metric: str | None) -> str:
    """One-line recommendation comparing the selected models."""
    if len(df) < 2:
        return "Select at least two models to see a recommendation."

    fastest = df.loc[df["latency_ms"].idxmin()]

    if metric and metric in df.columns and df[metric].notna().any():
        better = df[metric].idxmax() if metric in _HIGHER_BETTER else df[metric].idxmin()
        best_quality = df.loc[better]
    else:
        best_quality = fastest

    if best_quality["label"] == fastest["label"]:
        return (
            f"**{fastest['label']}** wins outright — it is both the fastest and "
            f"the best on {metric or 'the available metric'}."
        )

    speed_ratio = best_quality["latency_ms"] / fastest["latency_ms"]
    quality_gap = ""
    if metric:
        diff = abs(best_quality[metric] - fastest[metric])
        unit = "" if metric in ("wer", "bleu") else "-point"
        quality_gap = f" (a {diff:.3f}{unit} edge)"

    return (
        f"**Tradeoff:** {best_quality['label']} is best on {metric}{quality_gap}, "
        f"but {fastest['label']} is **{speed_ratio:.1f}× faster** "
        f"({fastest['latency_ms']:.1f} ms vs {best_quality['latency_ms']:.1f} ms). "
        f"Pick {fastest['label']} when latency matters; {best_quality['label']} when quality does."
    )


def main() -> None:
    st.set_page_config(page_title="Model-Eval-Lab", page_icon="⚖️", layout="wide")
    st.title("⚖️ Model-Eval-Lab")
    st.caption("Compare models on quality **and** speed. Pick two, get a verdict.")

    records = load_results()
    if not records:
        st.info(
            f"No results yet. Run a benchmark first, e.g.\n\n"
            f"```\npython src/run_benchmark.py distilbert cardiffnlp\n```\n\n"
            f"Results are read from `{RESULTS_PATH}`."
        )
        return

    df = to_frame(records)

    # --- Sidebar controls ---
    tasks = sorted(df["task"].dropna().unique())
    task = st.sidebar.selectbox("Task", tasks)
    task_df = df[df["task"] == task]

    labels = sorted(task_df["label"].unique())
    chosen = st.sidebar.multiselect(
        "Models to compare", labels, default=labels[: min(2, len(labels))]
    )

    metric_cols = [c for c in (*_HIGHER_BETTER, *_LOWER_BETTER) if c in task_df.columns]
    default_metric = pick_quality_metric(task_df)
    metric = st.sidebar.selectbox(
        "Quality metric (Y axis)",
        metric_cols or ["(none available)"],
        index=(
            (metric_cols.index(default_metric) if default_metric in metric_cols else 0)
            if metric_cols
            else 0
        ),
    )
    if metric not in task_df.columns:
        metric = None

    sel = task_df[task_df["label"].isin(chosen)]
    if sel.empty:
        st.warning("Select at least one model in the sidebar.")
        return

    # --- Verdict ---
    st.subheader("Verdict")
    st.markdown(verdict(sel, metric))

    # --- Comparison table ---
    st.subheader("Comparison")
    display_cols = ["label", "latency_ms", "memory_mb", *metric_cols]
    st.dataframe(
        sel[display_cols].rename(columns={"label": "model"}).set_index("model"),
        use_container_width=True,
    )

    # --- Scatter: latency (lower-left is better) vs quality ---
    if metric:
        st.subheader(f"Latency vs. {metric}")
        note = "lower is better" if metric in _LOWER_BETTER else "higher is better"
        st.caption(f"X: latency ms (lower is better) · Y: {metric} ({note}) · size: memory")
        st.scatter_chart(
            sel,
            x="latency_ms",
            y=metric,
            color="label",
            size="memory_mb",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
