"""CLI entrypoint for the multi-modal model benchmark.

The engine lives in the `model_eval` package; this file just wires the pieces
together for command-line use.

Examples:
  # Run every model in the config
  .venv/bin/python src/run_benchmark.py

  # Compare only two specific models (matched by name or substring)
  .venv/bin/python src/run_benchmark.py distilbert cardiffnlp

  # List the models the config knows about
  .venv/bin/python src/run_benchmark.py --list
"""

from __future__ import annotations

import argparse

from model_eval import load_config, print_report, run_benchmark
from model_eval.data import iter_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark and compare models. With no MODEL arguments, every model "
            "in config/models.yaml is run; otherwise only the named models are "
            "compared."
        )
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Model names (or case-insensitive substrings) to compare. "
        "Omit to run all configured models.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the configured models grouped by task, then exit.",
    )
    parser.add_argument(
        "--launcher",
        choices=["inline", "process"],
        default=None,
        help="How to run each model: 'inline' (this process) or 'process' "
        "(a fresh subprocess per model for accurate memory + crash isolation). "
        "Defaults to the config's 'launcher' key, else 'inline'.",
    )
    args = parser.parse_args()

    if args.list:
        for spec in iter_models(load_config()):
            print(f"{spec.task:32} {spec.backend:12} {spec.model}")
        return

    results = run_benchmark(models=args.models or None, launcher=args.launcher)
    print_report(results)


if __name__ == "__main__":
    main()
