"""CLI entrypoint for the multi-modal model benchmark.

The engine lives in the `model_eval` package; this file just wires the pieces
together for command-line use.

Run with:  .venv/bin/python src/run_benchmark.py
"""

from __future__ import annotations

from model_eval import print_report, run_benchmark


def main() -> None:
    results = run_benchmark()
    print_report(results)


if __name__ == "__main__":
    main()
