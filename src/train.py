"""
Train the spending model.

Run from the project root:
    python -m src.train                          # default settings
    python -m src.train --epochs 100 --hidden 256 --lr 0.0005

The best model (lowest validation error) is saved to models/spending_model.pt.
The Streamlit app calls train() directly so it can draw the loss curve live.
"""

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import CATEGORIES, HISTORY_FEATURES, Windows, load_splits
from .model import SpendingModel

DEFAULT_MODEL_PATH = Path("models/spending_model.pt")


@dataclass
class TrainConfig:
    history: int = 6          # months of history the model sees
    hidden: int = 128         # neurons per hidden layer
    layers: int = 2           # number of hidden layers
    dropout: float = 0.1      # fraction of neurons randomly switched off while training
    lr: float = 3e-4          # learning rate
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 200
    patience: int = 10        # stop early if validation hasn't improved for this many epochs
    loss: str = "l1"          # "l1" (mean absolute error), "mse" or "huber"
    seed: int = 0


def to_tensors(w: Windows):
    return (torch.from_numpy(w.history), torch.from_numpy(w.month),
            torch.from_numpy(w.target))


def loss_fn(name: str):
    return {"l1": torch.nn.L1Loss(), "mse": torch.nn.MSELoss(),
            "huber": torch.nn.HuberLoss(delta=0.05)}[name]


def predict_windows(model: SpendingModel, w: Windows) -> np.ndarray:
    hist, month, _ = to_tensors(w)
    return model.predict(hist, month).numpy()


def train(config: TrainConfig, splits=None, on_epoch=None, verbose=True):
    """
    Train a model. `on_epoch(epoch, train_loss, val_mae)` is called after every epoch
    (used by the app for a live chart). Returns (best_model, history_dict).
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if splits is None:
        splits = load_splits(history=config.history, seed=config.seed)

    model = SpendingModel(config.history, len(HISTORY_FEATURES), len(CATEGORIES),
                          config.hidden, config.layers, config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    criterion = loss_fn(config.loss)
    loader = DataLoader(TensorDataset(*to_tensors(splits["train"])),
                        batch_size=config.batch_size, shuffle=True)
    val = splits["val"]

    history = {"epoch": [], "train_loss": [], "val_mae": []}
    best_mae, best_state, bad_epochs = float("inf"), None, 0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for hist, month, target in loader:
            s = model.scale(hist)[:, None]
            # Compare predictions and targets as proportions of the person's spending,
            # so big spenders don't dominate the loss.
            loss = criterion(model(hist, month) / s, target / s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target)
            count += len(target)

        val_mae = float(np.abs(predict_windows(model, val) - val.target).mean())
        history["epoch"].append(epoch)
        history["train_loss"].append(total / count)
        history["val_mae"].append(val_mae)
        if on_epoch:
            on_epoch(epoch, total / count, val_mae)
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"epoch {epoch:3d}  train loss {total / count:.4f}  val MAE ${val_mae:7.2f}")

        if val_mae < best_mae - 1e-3:
            best_mae, bad_epochs = val_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                if verbose:
                    print(f"Early stop at epoch {epoch} (no improvement for {config.patience} epochs)")
                break

    model.load_state_dict(best_state)
    model.eval()
    history["best_val_mae"] = best_mae
    history["seconds"] = time.time() - start
    return model, history


def save_model(model: SpendingModel, config: TrainConfig, history: dict,
               path=DEFAULT_MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                "categories": CATEGORIES, "features": HISTORY_FEATURES,
                "history": history}, path)


def load_model(path=DEFAULT_MODEL_PATH):
    """Returns (model, config, training_history)."""
    ckpt = torch.load(path, weights_only=True)
    if ckpt["categories"] != CATEGORIES:
        raise ValueError("Saved model was trained with different categories; retrain it.")
    config = TrainConfig(**ckpt["config"])
    model = SpendingModel(config.history, len(ckpt["features"]), len(CATEGORIES),
                          config.hidden, config.layers, config.dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config, ckpt["history"]


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the spending prediction model.")
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = vars(parser.parse_args())
    out = args.pop("out")
    config = TrainConfig(**args)

    splits = load_splits(history=config.history, seed=config.seed)
    model, history = train(config, splits)
    save_model(model, config, history, out)
    print(f"\nBest validation MAE ${history['best_val_mae']:.2f} "
          f"({history['seconds']:.1f}s). Saved to {out}\n")

    from .evaluate import report
    report(splits, model)


if __name__ == "__main__":
    main()
