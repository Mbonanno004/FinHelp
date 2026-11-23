"""
TEMPLATE for adding your own model to the comparison.

1. Copy this file to a new name WITHOUT a leading underscore, e.g.
       src/custom_models/my_small_nn.py
2. Fill in NAME, DESCRIPTION, HISTORY and the three functions below.
3. Train it:   python -m src.custom_models.my_small_nn
   (or press "Train" on the app's Compare Models page)
4. It now appears on the Compare Models page, in `python -m src.evaluate`,
   and can be picked as the prediction model in the Budget Planner.

The only rule: given a Windows object, return predicted dollars per category as an
array of shape (number of examples, 11), in the column order of src.data.CATEGORIES.

What's inside a Windows object `w` (N = number of examples):
    w.history  (N, HISTORY, 12)  past months in dollars: 11 categories, then Income
    w.month    (N,)              calendar month (1-12) of the month being predicted
    w.target   (N, 11)           the true answer (only use this while TRAINING)
    w.meta     DataFrame         sample_id and target_month for each example

Always train on splits["train"] (optionally only part of it). Never train on "test" or
"original", because those are the people everyone's models are scored on.
"""

from pathlib import Path

import numpy as np

from src.data import CATEGORIES, Windows, load_splits

NAME = "My model"                      # shown in the app
DESCRIPTION = "One sentence on the method and how much data it was trained on."
HISTORY = 6                            # months of history your model needs (1-11)
SAVE_PATH = Path(__file__).resolve().parents[2] / "models" / "my_model.npz"   # change the file name!


def train() -> dict:
    splits = load_splits(history=HISTORY)
    train_w = splits["train"]

    # Optional: use less data, e.g. only the first 50 people
    # people = sorted(train_w.meta["sample_id"].unique())[:50]
    # train_w = train_w.subset(train_w.meta["sample_id"].isin(people))

    # TODO: fit your model on train_w.history / train_w.month -> train_w.target
    # (splits["val"] is available for tuning or early stopping)
    average_share = train_w.target.mean(axis=0)            # placeholder "model"

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SAVE_PATH, average_share=average_share)       # for PyTorch use torch.save(...)
    return {"examples": len(train_w)}


def is_trained() -> bool:
    return SAVE_PATH.exists()


def load():
    """Load what train() saved and return a function: Windows -> (N, 11) dollars."""
    saved = np.load(SAVE_PATH)

    def predict(w: Windows) -> np.ndarray:
        # TODO: replace with your model's prediction
        return np.tile(saved["average_share"], (len(w.month), 1))

    return predict


if __name__ == "__main__":
    print(train())
