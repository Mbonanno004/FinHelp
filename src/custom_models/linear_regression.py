"""
Example comparison model: linear regression, trained on only 100 people.

A deliberately different and simpler method than the neural network:
  * No layers, no ReLU, no gradient descent.
  * Each category's prediction is just a weighted sum of the inputs:
        prediction = w1*input1 + w2*input2 + ... + bias
  * The best weights are found in ONE step with a formula (least squares with a small
    "ridge" penalty) instead of thousands of small training steps.
  * It only uses 100 of the 350 training people, to show the effect of less data.

It uses the same inputs as the neural network (spending as a share of the person's
normal monthly total, plus the calendar month), so the comparison is about the METHOD.

Train it (takes about a second):
    python -m src.custom_models.linear_regression
"""

from pathlib import Path

import numpy as np

from src.data import CATEGORIES, Windows, load_splits

NAME = "Linear regression (100 people)"
DESCRIPTION = ("Weighted sum of the inputs; weights found in one step with ridge least squares. "
               "Trained on only 100 people.")
HISTORY = 6                  # months of history it looks at
MAX_PEOPLE = 100             # how many training people to use
RIDGE = 1.0                  # penalty that keeps weights small (prevents overfitting)
WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "models" / "linear_regression.npz"


def features(hist: np.ndarray, month: np.ndarray):
    """Same idea as the neural network: divide by the person's normal monthly spending."""
    n_cat = len(CATEGORIES)
    scale = hist[:, :, :n_cat].sum(axis=2).mean(axis=1) + 1.0
    x = np.minimum(hist / scale[:, None, None], 10.0).reshape(len(hist), -1)
    angle = 2 * np.pi * (month - 1) / 12
    X = np.column_stack([x, np.log(scale) / 10, np.sin(angle), np.cos(angle), np.ones(len(hist))])
    return X, scale


def train(max_people: int = MAX_PEOPLE) -> dict:
    splits = load_splits(history=HISTORY)
    train_w = splits["train"]
    people = sorted(train_w.meta["sample_id"].unique())[:max_people]
    train_w = train_w.subset(train_w.meta["sample_id"].isin(people))

    X, scale = features(train_w.history, train_w.month)
    Y = train_w.target / scale[:, None]
    # Ridge least squares:  W = (XᵀX + λI)⁻¹ XᵀY
    W = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ Y)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS_PATH, W=W)
    return {"people": len(people), "examples": len(train_w), "saved to": str(WEIGHTS_PATH)}


def is_trained() -> bool:
    return WEIGHTS_PATH.exists()


def load():
    """Return a function: Windows -> predicted dollars, shape (N, 11)."""
    W = np.load(WEIGHTS_PATH)["W"]

    def predict(w: Windows) -> np.ndarray:
        X, scale = features(w.history, w.month)
        return (X @ W) * scale[:, None]

    return predict


if __name__ == "__main__":
    print(train())
