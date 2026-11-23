"""
The model "zoo": every predictor the app can use or compare, behind one common interface.

A Predictor has
  name, kind ("Neural network", "Custom" or "Baseline"), description
  history   how many past months it needs
  predict(windows) -> array (N, 11) of predicted dollars per category

Three sources:
  1. The main PyTorch neural network (models/spending_model.pt)
  2. Custom models: every file in src/custom_models/ whose name doesn't start with "_"
     (see src/custom_models/_template.py for how to write one)
  3. The simple baselines from baselines.py
"""

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .baselines import BASELINES
from .data import Windows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "spending_model.pt"
NN_NAME = "Neural network (main)"


@dataclass
class Predictor:
    name: str
    kind: str
    history: int
    fn: Callable[[Windows], np.ndarray]
    description: str = ""

    def predict(self, w: Windows) -> np.ndarray:
        """Predict for every example in `w`. Extra history is trimmed to what this model uses,
        so models with different history lengths are scored on exactly the same months."""
        if w.history.shape[1] < self.history:
            raise ValueError(f"{self.name} needs {self.history} months of history, "
                             f"got {w.history.shape[1]}.")
        if w.history.shape[1] > self.history:
            w = Windows(w.history[:, -self.history:], w.month, w.target, w.meta)
        pred = np.asarray(self.fn(w), dtype=np.float64)
        if pred.shape != w.target.shape:
            raise ValueError(f"{self.name} returned shape {pred.shape}, expected {w.target.shape}.")
        return np.clip(pred, 0.0, None)


# --------------------------------------------------------------------------------------

BASELINE_INFO = {
    "Last month": (1, "Next month = exactly what was spent last month."),
    "3-month average": (3, "Next month = average of the last 3 months."),
    "Full-history average": (6, "Next month = average of the last 6 months."),
    "Full-history median": (6, "Next month = the middle value of the last 6 months (ignores one-off spikes)."),
}


def baseline_predictors():
    return [Predictor(name, "Baseline", BASELINE_INFO[name][0], fn, BASELINE_INFO[name][1])
            for name, fn in BASELINES.items()]


def nn_predictor_from(model, config, name=NN_NAME):
    from .train import predict_windows
    return Predictor(name, "Neural network", config.history,
                     lambda w: predict_windows(model, w),
                     f"PyTorch MLP ({config.layers}×{config.hidden} neurons, "
                     f"{config.history} months of history), trained on 350 people.")


def nn_predictor(path=DEFAULT_MODEL_PATH):
    from .train import load_model
    model, config, _ = load_model(path)
    return nn_predictor_from(model, config)


def custom_modules():
    """Import every custom model file. Returns ({module_name: module}, {module_name: error})."""
    from . import custom_models
    modules, errors = {}, {}
    for info in pkgutil.iter_modules(custom_models.__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"src.custom_models.{info.name}")
            mod = importlib.reload(mod)          # pick up edits without restarting the app
            for attr in ("NAME", "HISTORY", "is_trained", "load"):
                if not hasattr(mod, attr):
                    raise AttributeError(f"missing `{attr}`")
            modules[info.name] = mod
        except Exception as e:  # a broken teammate file shouldn't crash everything
            errors[info.name] = f"{type(e).__name__}: {e}"
    return modules, errors


def all_predictors(model_path=DEFAULT_MODEL_PATH, include_baselines=True):
    """
    Returns (predictors, untrained, errors):
      predictors  list of ready-to-use Predictor objects
      untrained   {display name: module} for custom models that still need training
      errors      {module name: message} for custom files that failed to load
    """
    predictors, untrained = [], {}
    if Path(model_path).exists():
        predictors.append(nn_predictor(model_path))
    modules, errors = custom_modules()
    for mod_name, mod in modules.items():
        try:
            if mod.is_trained():
                predictors.append(Predictor(mod.NAME, "Custom", int(mod.HISTORY), mod.load(),
                                            getattr(mod, "DESCRIPTION", "")))
            else:
                untrained[mod.NAME] = mod
        except Exception as e:
            errors[mod_name] = f"{type(e).__name__}: {e}"
    if include_baselines:
        predictors += baseline_predictors()
    return predictors, untrained, errors
