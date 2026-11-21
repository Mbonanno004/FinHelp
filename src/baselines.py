"""
Simple non-learning predictors. The neural network has to beat these to be worth using.

Each function takes a Windows object (see data.py) and returns predictions shaped
(N, number_of_categories), in dollars.
"""

import numpy as np

from .data import CATEGORIES, Windows

N_CAT = len(CATEGORIES)


def last_month(w: Windows) -> np.ndarray:
    """Next month = exactly what was spent last month."""
    return w.history[:, -1, :N_CAT]


def moving_average(w: Windows, k: int = 3) -> np.ndarray:
    """Next month = average of the last k months (the main baseline)."""
    return w.history[:, -k:, :N_CAT].mean(axis=1)


def history_average(w: Windows) -> np.ndarray:
    """Next month = average of all available history months."""
    return w.history[:, :, :N_CAT].mean(axis=1)


def history_median(w: Windows) -> np.ndarray:
    """Next month = the median month in the history. Ignores one-off spikes (a trip,
    a tax bill), which lowers MAE on irregular categories - a fair comparison for a model
    trained with L1 loss, which also learns to predict the 'typical' month."""
    return np.median(w.history[:, :, :N_CAT], axis=1)


BASELINES = {
    "Last month": last_month,
    "3-month average": moving_average,
    "Full-history average": history_average,
    "Full-history median": history_median,
}
