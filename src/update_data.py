from typing import Any

import pandas as pd


def update_sentiment_analysis_data(data: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    """Update the data dict with the predictions from a sentiment-analysis model.

    The predictions are a list of dicts, one per input text. Each dict has
    keys like 'label' and 'score'. We add these to the data dict under the
    'predictions' key.
    """
    data["predictions"] = predictions
    with open("data/sentiment-analysis/data.csv", "a", encoding="utf-8") as f:
        df = pd.DataFrame(predictions)
        df.to_csv(f, header=f.tell() == 0, index=False)  # Write header only if file is empty
