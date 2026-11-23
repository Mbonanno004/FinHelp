"""
How accurate is a model? Metrics, model comparison and per-person accuracy checks.

Metrics (computed the same way everywhere in the project and the app):
  MAE         mean absolute error: average dollars off, per category per month
              = mean over months and categories of |predicted - actual|
  Total MAE   average dollars off on a person's TOTAL monthly spending
  WAPE %      weighted absolute percentage error
              = sum of |predicted - actual| / sum of actual spending x 100
  Accuracy %  = 100 - WAPE   (floored at 0). "How much of the spending was predicted right."

Run from the project root to compare every available model:
    python -m src.evaluate
"""

import numpy as np
import pandas as pd

from .data import CATEGORIES, Windows, make_windows

ACCURACY_EXPLANATION = """
**How accuracy is calculated**

1. **Hide a month, then predict it.** For every month that has enough earlier history,
   the model only sees the months *before* it (for example the 6 months before) and predicts
   that month's spending in each of the 11 categories. It never sees the month it is predicting.
2. **Compare with what really happened.** For every category in every month we take the
   miss: `|predicted − actual|`.
3. **Summarize the misses:**
   - **Average miss per category (MAE)** = the average of all those misses, in dollars.
   - **Error rate (WAPE)** = all misses added up ÷ all actual spending added up.
   - **Accuracy** = 100% − error rate. 85% accuracy means the misses add up to 15% of
     what was actually spent.
4. **Compare with a simple method.** The same check is run on the *3-month average*
   (next month = average of the last 3 months). A model is only useful if it beats that.

This is called **backtesting**, also known as walk-forward validation. For the dataset-wide
numbers, it is done on **test people the models never saw during training**, so the scores
are honest.
"""


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = np.abs(pred - actual)
    wape = float(100 * err.sum() / max(actual.sum(), 1e-9))
    return {
        "Accuracy %": max(0.0, 100.0 - wape),
        "MAE": float(err.mean()),
        "Total MAE": float(np.abs(pred.sum(1) - actual.sum(1)).mean()),
        "WAPE %": wape,
    }


def compare(predictors, splits: dict, names=("test", "original")) -> pd.DataFrame:
    """Score every predictor on each named split. Returns one row per (split, model)."""
    rows = []
    for split in names:
        w = splits.get(split)
        if w is None:
            continue
        for p in predictors:
            rows.append({"split": split, "model": p.name, "kind": p.kind,
                         **metrics(p.predict(w), w.target)})
    return pd.DataFrame(rows)


def category_errors(predictors, w: Windows) -> pd.DataFrame:
    """Average dollars off per category, one column per model."""
    table = pd.DataFrame({"Avg actual": w.target.mean(axis=0)}, index=CATEGORIES)
    for p in predictors:
        table[p.name] = np.abs(p.predict(w) - w.target).mean(axis=0)
    return table


# --------------------------------------------------------------------------------------
# Per-person backtest
# --------------------------------------------------------------------------------------

def backtest(predictors, monthly_person: pd.DataFrame):
    """
    Walk forward through ONE person's history: for every month that has enough earlier
    months for all predictors, predict it from the earlier months only.

    Returns (per_month, summary):
      per_month  Month, Actual total, and each model's predicted total
      summary    one row per model with Accuracy %, MAE, Total MAE, WAPE %
    Raises ValueError if the person doesn't have enough months.
    """
    history = max(p.history for p in predictors)
    w = make_windows(monthly_person, history=history)
    per_month = pd.DataFrame({"Month": w.meta["target_month"], "Actual": w.target.sum(axis=1)})
    rows = []
    for p in predictors:
        pred = p.predict(w)
        per_month[p.name] = pred.sum(axis=1)
        rows.append({"model": p.name, "kind": p.kind, "months checked": len(w),
                     **metrics(pred, w.target)})
    return per_month, pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------

def report(splits, predictors):
    pd.set_option("display.width", 140)
    print("Overall (Accuracy: higher is better; errors: lower is better)")
    print(compare(predictors, splits).round(2).to_string(index=False))
    for split in ("test", "original"):
        if splits.get(split) is not None:
            print(f"\nAverage $ off per category on '{split}'")
            print(category_errors(predictors, splits[split]).round(1).to_string())


def main():
    from .data import load_splits
    from .zoo import all_predictors

    predictors, untrained, errors = all_predictors()
    for name in untrained:
        print(f"(skipping '{name}': not trained yet)")
    for name, err in errors.items():
        print(f"(could not load custom model '{name}': {err})")
    history = max(p.history for p in predictors)
    splits = load_splits(history=history)
    for name, w in splits.items():
        if w is not None:
            print(f"{name:>9}: {len(w):6,} examples from {w.meta['sample_id'].nunique()} people")
    print()
    report(splits, predictors)


if __name__ == "__main__":
    main()
