"""Phase 1: Run two sentiment-analysis models on the same sentences.

Goal: prove we can load a Hugging Face model and get predictions.
Run it with:  .venv/bin/python src/run_models.py
"""

from transformers import pipeline

# The two models we'll compare for the whole project.
# Both do sentiment analysis, but they were trained differently,
# so later phases will reveal real tradeoffs between them.
MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]

# The same input for every model, so comparisons are fair.
TEXTS = [
    "I loved this movie!",
    "What a waste of time.",
]


def main() -> None:
    for name in MODELS:
        clf = pipeline("sentiment-analysis", model=name)

        predictions = clf(TEXTS)

        print(f"\n=== {name} ===")
        for text, pred in zip(TEXTS, predictions):
            print(f"  {text!r} -> {pred}")


if __name__ == "__main__":
    main()
