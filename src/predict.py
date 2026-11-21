"""
Forecast next month for ONE person, given their monthly table (from data.to_monthly).
Used by the Budget Planner page of the app.
"""

import numpy as np
import pandas as pd
import torch

from .baselines import BASELINES
from .data import CATEGORIES, HISTORY_FEATURES, Windows


def person_window(monthly_person: pd.DataFrame, history: int) -> Windows:
    """Build a single example from the person's most recent `history` months."""
    person = monthly_person.sort_values("month")
    if len(person) < history:
        raise ValueError(f"Need at least {history} months of history, got {len(person)}.")
    recent = person.tail(history)
    next_month = recent["month"].iloc[-1] + 1
    return Windows(
        history=recent[HISTORY_FEATURES].to_numpy(dtype=np.float32)[None],
        month=np.array([next_month.month], dtype=np.int64),
        target=np.zeros((1, len(CATEGORIES)), dtype=np.float32),   # unknown future
        meta=pd.DataFrame({"sample_id": [person["sample_id"].iloc[0]],
                           "target_month": [str(next_month)]}),
    )


def forecast(model, monthly_person: pd.DataFrame, history: int) -> pd.DataFrame:
    """
    Returns a table: Category, Neural network, 3-month average, and the forecast month
    in the attribute .attrs['month'].
    """
    w = person_window(monthly_person, history)
    nn_pred = model.predict(torch.from_numpy(w.history), torch.from_numpy(w.month)).numpy()[0]
    avg = BASELINES["3-month average"](w)[0]
    out = pd.DataFrame({"Category": CATEGORIES, "Neural network": nn_pred,
                        "3-month average": avg})
    out.attrs["month"] = w.meta["target_month"].iloc[0]
    return out
