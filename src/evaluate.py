"""
Evaluation metrics, and a script that scores the baselines.

Metrics (all in dollars unless noted):
  MAE        mean absolute error per category per month - "on average we're off by $X"
  Total MAE  error on the person's TOTAL monthly spending
  WAPE       total absolute error / total actual spending, in % (scale-free)

Run from the project root:
    python -m src.evaluate
"""

import argparse

import numpy as np
import pandas as pd

from .baselines import BASELINES
from .data import CATEGORIES, load_splits


def metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = np.abs(pred - actual)
    return {
        "MAE": float(err.mean()),
        "Total MAE": float(np.abs(pred.sum(1) - actual.sum(1)).mean()),
        "WAPE %": float(100 * err.sum() / max(actual.sum(), 1e-9)),
    }


def per_category_mae(pred: np.ndarray, actual: np.ndarray) -> pd.Series:
    return pd.Series(np.abs(pred - actual).mean(axis=0), index=CATEGORIES)


def score_all(predictors: dict, splits: dict, names=("test", "original")) -> pd.DataFrame:
    """predictors: name -> function(Windows) -> predictions. Returns a tidy results table."""
    rows = []
    for split in names:
        w = splits[split]
        for name, fn in predictors.items():
            rows.append({"split": split, "method": name, **metrics(fn(w), w.target)})
    return pd.DataFrame(rows)


def predictors_for(model=None) -> dict:
    """All baselines, plus the neural network if one is given."""
    preds = dict(BASELINES)
    if model is not None:
        from .train import predict_windows
        preds["Neural network"] = lambda w: predict_windows(model, w)
    return preds


def category_table(splits, model=None, split="test") -> pd.DataFrame:
    """Per-category MAE for the best baseline vs. the model on one split."""
    w = splits[split]
    table = pd.DataFrame({"avg actual": w.target.mean(axis=0),
                          "Full-history avg": per_category_mae(BASELINES["Full-history average"](w), w.target),
                          "Full-history median": per_category_mae(BASELINES["Full-history median"](w), w.target)},
                         index=CATEGORIES)
    if model is not None:
        from .train import predict_windows
        table["Neural network"] = per_category_mae(predict_windows(model, w), w.target)
    return table


def report(splits, model=None):
    pd.set_option("display.width", 120)
    print("Overall (lower is better)")
    print(score_all(predictors_for(model), splits).round(2).to_string(index=False))
    for split in ("test", "original"):
        print(f"\nMAE per category ($) on '{split}'")
        print(category_table(splits, model, split).round(1).to_string())


def main():
    parser = argparse.ArgumentParser(description="Score the baselines (and the saved model, if any).")
    parser.add_argument("--model", default="models/spending_model.pt")
    parser.add_argument("--history", type=int, default=None,
                        help="months of history (default: the saved model's, else 6)")
    args = parser.parse_args()

    model, history = None, args.history
    from pathlib import Path
    if Path(args.model).exists():
        from .train import load_model
        model, config, _ = load_model(args.model)
        history = config.history
    splits = load_splits(history=history or 6)
    for name, w in splits.items():
        print(f"{name:>9}: {len(w):6,} examples from {w.meta['sample_id'].nunique()} people")
    print()
    report(splits, model)


if __name__ == "__main__":
    main()
