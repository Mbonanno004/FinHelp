"""
FinHelp API: a small FastAPI server that exposes the Python code in src/ to the
SvelteKit frontend in web/.

Run from the project root:
    python -m uvicorn api.main:app --port 8000 --reload

The frontend talks to it at /api/... (the SvelteKit dev server forwards those requests
here). Interactive docs for every endpoint: http://localhost:8000/docs
"""

import io
import json
import queue
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.budget import FLEXIBILITY, GROUPS, GUIDELINE_SHARE, make_budget
from src.data import (CATEGORIES, FIXED_CATEGORIES, load_splits, load_transactions,
                      split_people, to_monthly)
from src.evaluate import backtest, category_errors, compare
from src.predict import forecast
from src.train import TrainConfig, save_model, train
from src.zoo import all_predictors, baseline_predictors, nn_predictor_from

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "data" / "generated_transactions.csv"
ORIGINAL = ROOT / "data" / "synthetic_transactions.csv"
MODEL_PATH = ROOT / "models" / "spending_model.pt"
DATASETS = {
    "original": ("Original dataset (10 people)", ORIGINAL),
    "generated": ("Generated dataset (500 people)", GENERATED),
}

app = FastAPI(title="FinHelp API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------------------
# Caching helpers
# --------------------------------------------------------------------------------------

def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def models_stamp() -> tuple:
    files = list((ROOT / "models").glob("*")) + list((ROOT / "src" / "custom_models").glob("*.py"))
    return tuple(sorted((f.name, f.stat().st_mtime) for f in files if f.is_file()))


@lru_cache(maxsize=4)
def _registry(stamp: tuple):
    return all_predictors(MODEL_PATH)


def registry():
    return _registry(models_stamp())


@lru_cache(maxsize=8)
def _splits(history: int, version: float):
    return load_splits(GENERATED, ORIGINAL, history=history)


def splits_for(history: int):
    if not GENERATED.exists():
        raise HTTPException(409, "The generated dataset is missing. Create it on the Train Model page.")
    return _splits(history, mtime(GENERATED))


@lru_cache(maxsize=4)
def _monthly(path: str, version: float) -> pd.DataFrame:
    return to_monthly(load_transactions(path))


def monthly_for(key: str) -> pd.DataFrame:
    if key not in DATASETS:
        raise HTTPException(404, f"Unknown dataset '{key}'.")
    path = DATASETS[key][1]
    if not path.exists():
        raise HTTPException(404, f"{path.name} does not exist yet.")
    return _monthly(str(path), mtime(path))


@lru_cache(maxsize=2)
def _training_people(version: float) -> frozenset:
    return frozenset(split_people(_monthly(str(GENERATED), version)["sample_id"].unique())[0])


def find_predictor(name: str):
    for p in registry()[0]:
        if p.name == name:
            return p
    raise HTTPException(404, f"Unknown model '{name}'.")


# --------------------------------------------------------------------------------------
# Converting between the monthly table and JSON rows
# --------------------------------------------------------------------------------------

class MonthRow(BaseModel):
    """One month of one person's history, e.g. {"month": "2026-05", "Income": 3000, "Housing": 1000, ...}"""
    model_config = {"extra": "allow"}
    month: str


def rows_from_monthly(df: pd.DataFrame) -> list[dict]:
    df = df.sort_values("month")
    return [{"month": str(r["month"]), "Income": round(float(r["Income"]), 2),
             **{c: round(float(r[c]), 2) for c in CATEGORIES}}
            for _, r in df.iterrows()]


def monthly_from_rows(rows: list[MonthRow]) -> pd.DataFrame:
    if not rows:
        raise HTTPException(400, "No months of history were provided.")
    records = []
    for r in rows:
        d = r.model_dump()
        rec = {"month": d["month"]}
        for c in CATEGORIES + ["Income"]:
            try:
                rec[c] = float(d.get(c) or 0.0)
            except (TypeError, ValueError):
                raise HTTPException(400, f"'{c}' in {d['month']} must be a number.")
        records.append(rec)
    df = pd.DataFrame(records)
    try:
        df["month"] = pd.PeriodIndex(df["month"].astype(str), freq="M")
    except Exception:
        raise HTTPException(400, "Months must look like 2026-05.")
    if df["month"].duplicated().any():
        raise HTTPException(400, "Each month can only appear once.")
    df["sample_id"] = 0
    df["Savings"] = 0.0
    df["Total"] = df[CATEGORIES].sum(axis=1)
    return df.sort_values("month").reset_index(drop=True)


def example_rows(months: int = 12) -> list[dict]:
    """The example person from the project proposal: $3,000 income, about $2,300 spending,
    with a little natural variation (winter/summer utilities, holiday shopping)."""
    base = {"Housing": 1000, "Groceries": 350, "Dining": 200, "Transportation": 150,
            "Utilities": 150, "Entertainment": 100, "Shopping": 120, "Other": 230}
    season = {"Utilities": [1.25, 1.2, 1, .95, .95, 1, 1.2, 1.2, 1, .95, .97, 1.2],
              "Shopping": [.95, .85, .95, 1, 1, 1, 1, 1.05, 1, .95, 1.25, 1.45]}
    end = pd.Timestamp.today().to_period("M") - 1
    rng = np.random.default_rng(0)
    rows = []
    for i in range(months):
        month = end - (months - 1 - i)
        row = {"month": str(month), "Income": 3000.0}
        for c in CATEGORIES:
            v = base.get(c, 0.0) * season.get(c, [1] * 12)[month.month - 1]
            if c not in FIXED_CATEGORIES and v:
                v *= rng.uniform(0.85, 1.15)
            row[c] = round(float(v), 2)
        rows.append(row)
    return rows


def predictor_info(p) -> dict:
    return {"name": p.name, "kind": p.kind, "history": p.history, "description": p.description}


def metric_row(r) -> dict:
    return {"accuracy": float(r["Accuracy %"]), "mae": float(r["MAE"]),
            "total_mae": float(r["Total MAE"]), "wape": float(r["WAPE %"])}


# --------------------------------------------------------------------------------------
# General information
# --------------------------------------------------------------------------------------

@app.get("/api/meta")
def meta():
    return {
        "categories": CATEGORIES,
        "fixed_categories": FIXED_CATEGORIES,
        "flexibility": FLEXIBILITY,
        "guideline_share": GUIDELINE_SHARE,
        "groups": [{"name": n, "categories": c, "share_max": s} for n, c, s in GROUPS],
        "train_defaults": asdict(TrainConfig()),
        "generated_available": GENERATED.exists(),
    }


@app.get("/api/models")
def models():
    predictors, untrained, errors = registry()
    return {
        "models": [predictor_info(p) for p in predictors],
        "untrained": [{"name": name, "module": mod.__name__.rsplit(".", 1)[-1]}
                      for name, mod in untrained.items()],
        "errors": [{"module": m, "error": e} for m, e in errors.items()],
    }


# --------------------------------------------------------------------------------------
# People and their history
# --------------------------------------------------------------------------------------

@app.get("/api/example")
def example():
    return {"rows": example_rows()}


@app.get("/api/datasets")
def datasets():
    out = []
    for key, (label, path) in DATASETS.items():
        if path.exists():
            ids = sorted(int(i) for i in monthly_for(key)["sample_id"].unique())
            out.append({"key": key, "label": label, "people": ids})
    return {"datasets": out}


@app.get("/api/datasets/{key}/{sample_id}")
def dataset_person(key: str, sample_id: int):
    monthly = monthly_for(key)
    person = monthly[monthly["sample_id"] == sample_id]
    if person.empty:
        raise HTTPException(404, f"No person {sample_id} in the {key} dataset.")
    in_training = key == "generated" and sample_id in _training_people(mtime(GENERATED))
    return {"rows": rows_from_monthly(person), "in_training": in_training}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        monthly = to_monthly(load_transactions(io.BytesIO(await file.read())))
    except Exception as e:
        raise HTTPException(400, f"Could not read that file: {e}")
    return {"people": [{"sample_id": int(sid), "rows": rows_from_monthly(group)}
                       for sid, group in monthly.groupby("sample_id")]}


# --------------------------------------------------------------------------------------
# Prediction, budget, per-person accuracy
# --------------------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    model: str
    rows: list[MonthRow]


@app.post("/api/forecast")
def forecast_endpoint(req: ForecastRequest):
    p = find_predictor(req.model)
    monthly = monthly_from_rows(req.rows)
    need = max(p.history, 3)
    if len(monthly) < need:
        raise HTTPException(400, f"This model needs at least {need} months of history; "
                                 f"there are {len(monthly)}.")
    fc = forecast(p, monthly)
    recent = monthly.tail(3)
    return {
        "month": fc.attrs["month"],
        "categories": [{"category": r["Category"], "predicted": float(r["Predicted"]),
                        "recent_average": float(r["Recent average"])} for _, r in fc.iterrows()],
        "predicted_total": float(fc["Predicted"].sum()),
        "recent_average_total": float(fc["Recent average"].sum()),
        "recent_income": float(recent["Income"].mean()),
        # Typical month per category (median of the last 6 months): used for max logical spend
        "typical": {c: float(v) for c, v in monthly.tail(6)[CATEGORIES].median().items()},
    }


class BudgetRequest(BaseModel):
    predicted: dict[str, float]
    income: float = Field(ge=0)
    goal: float = Field(0.0, ge=0)
    max_cut: float = Field(0.30, ge=0, le=1)
    protected: list[str] = []
    target: Optional[float] = None
    typical: Optional[dict[str, float]] = None


@app.post("/api/budget")
def budget_endpoint(req: BudgetRequest):
    b = make_budget(req.predicted, req.income, req.goal, req.max_cut, req.protected,
                    req.target or None, req.typical)
    return {
        "table": [{"category": r["Category"], "type": r["Type"], "predicted": float(r["Predicted"]),
                   "max": float(r["Max"]), "recommended": float(r["Recommended"]),
                   "change": float(r["Change"]), "status": r["Status"]}
                  for _, r in b.table.iterrows()],
        "groups": b.groups,
        "income": b.income, "goal": b.goal, "savings": b.savings, "leftover": b.leftover,
        "goal_met": bool(b.goal_met), "messages": b.messages,
        "months_to_target": b.months_to_target,
        "spending_total": float(b.table["Recommended"].sum()),
        "change_total": float(b.table["Change"].sum()),
    }


class BacktestRequest(BaseModel):
    models: list[str]
    rows: list[MonthRow]


def run_backtest(predictors, monthly) -> dict:
    try:
        per_month, summary = backtest(predictors, monthly)
    except ValueError:
        need = max(p.history for p in predictors) + 1
        raise HTTPException(400, f"Add at least {need} months of history to check accuracy.")
    return {
        "per_month": [{"month": r["Month"], "actual": float(r["Actual"]),
                       "predicted": {p.name: float(r[p.name]) for p in predictors}}
                      for _, r in per_month.iterrows()],
        "summary": [{"model": r["model"], "kind": r["kind"], "months": int(r["months checked"]),
                     **metric_row(r)} for _, r in summary.iterrows()],
    }


@app.post("/api/backtest")
def backtest_endpoint(req: BacktestRequest):
    predictors = [find_predictor(n) for n in req.models]
    if not predictors:
        raise HTTPException(400, "Choose at least one model.")
    return run_backtest(predictors, monthly_from_rows(req.rows))


# --------------------------------------------------------------------------------------
# Comparing models
# --------------------------------------------------------------------------------------

@app.get("/api/compare")
def compare_endpoint(models: str):
    predictors = [find_predictor(n) for n in models.split("|") if n]
    if not predictors:
        raise HTTPException(400, "Choose at least one model.")
    splits = splits_for(max(p.history for p in predictors))
    res = compare(predictors, splits)
    overall = [{"model": r["model"], "kind": r["kind"], "split": r["split"], **metric_row(r)}
               for _, r in res.iterrows()]

    def cat_rows(w):
        if w is None:
            return None
        t = category_errors(predictors, w)
        return [{"category": c, "avg_actual": float(t.loc[c, "Avg actual"]),
                 "errors": {p.name: float(t.loc[c, p.name]) for p in predictors}}
                for c in t.index]

    return {
        "test_people": int(splits["test"].meta["sample_id"].nunique()),
        "test_predictions": len(splits["test"]),
        "overall": overall,
        "categories": {"test": cat_rows(splits["test"]), "original": cat_rows(splits["original"])},
    }


@app.post("/api/custom/{module}/train")
def train_custom(module: str):
    from src.zoo import custom_modules
    mods, errors = custom_modules()
    if module not in mods:
        raise HTTPException(404, errors.get(module, f"No custom model file '{module}'."))
    try:
        result = mods[module].train()
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")
    _registry.cache_clear()
    return {"result": {k: str(v) for k, v in (result or {}).items()}}


# --------------------------------------------------------------------------------------
# Training the main network
# --------------------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    people: int = Field(500, ge=50, le=5000)
    months: int = Field(24, ge=12, le=60)
    seed: int = 42


@app.post("/api/generate")
def generate(req: GenerateRequest):
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "generate_data.py"), "--people", str(req.people),
         "--months", str(req.months), "--seed", str(req.seed), "--out", str(GENERATED),
         "--people-out", str(ROOT / "data" / "generated_people.csv")],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(500, result.stderr)
    _splits.cache_clear()
    _monthly.cache_clear()
    return {"message": result.stdout.strip()}


class TrainRequest(BaseModel):
    history: int = Field(6, ge=3, le=9)
    hidden: int = Field(128, ge=8, le=1024)
    layers: int = Field(2, ge=1, le=6)
    dropout: float = Field(0.1, ge=0, le=0.9)
    lr: float = Field(3e-4, gt=0, le=0.1)
    epochs: int = Field(200, ge=1, le=1000)
    loss: str = Field("l1", pattern="^(l1|mse|huber)$")
    seed: int = 0


TRAINED_RUNS: dict = {}   # run_id -> (model, config, history)


@app.post("/api/train")
def train_endpoint(req: TrainRequest):
    """Streams newline-delimited JSON: one {"type": "epoch"} line per pass, then
    {"type": "done"} with the comparison results (or {"type": "error"})."""
    config = TrainConfig(**req.model_dump())
    splits = splits_for(config.history)
    events: queue.Queue = queue.Queue()

    def work():
        try:
            model, hist = train(config, splits, verbose=False,
                                on_epoch=lambda e, l, v: events.put(
                                    {"type": "epoch", "epoch": e, "train_loss": l, "val_mae": v}))
            run_id = uuid.uuid4().hex[:12]
            TRAINED_RUNS.clear()           # keep only the latest run in memory
            TRAINED_RUNS[run_id] = (model, config, hist)

            new = nn_predictor_from(model, config, name="New model (not saved)")
            current = next((p for p in registry()[0] if p.kind == "Neural network"), None)
            contenders = [new] + ([current] if current else []) + [
                p for p in baseline_predictors() if p.name in ("3-month average", "Full-history median")]
            res = compare(contenders, splits_for(max(p.history for p in contenders)))
            results = []
            for p in contenders:
                rows = res[res["model"] == p.name].set_index("split")
                results.append({"model": p.name, "kind": p.kind,
                                "test": float(rows.loc["test", "Accuracy %"]),
                                "original": float(rows.loc["original", "Accuracy %"])
                                if "original" in rows.index else None})
            events.put({"type": "done", "run_id": run_id, "seconds": hist["seconds"],
                        "epochs_run": hist["epoch"][-1], "best_val_mae": hist["best_val_mae"],
                        "results": results})
        except Exception as e:
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/train/{run_id}/save")
def save_run(run_id: str):
    if run_id not in TRAINED_RUNS:
        raise HTTPException(404, "That training run is no longer in memory. Train again.")
    model, config, hist = TRAINED_RUNS[run_id]
    save_model(model, config, hist, MODEL_PATH)
    _registry.cache_clear()
    return {"message": "Saved. The Budget Planner and Compare Models pages now use this model."}


# --------------------------------------------------------------------------------------
# Serving the built website (production mode)
# --------------------------------------------------------------------------------------
# After `npm --prefix web run build`, this server also serves the website itself, so the
# finished app runs from a single command at http://localhost:8000.

WEB_BUILD = ROOT / "web" / "build"

if (WEB_BUILD / "index.html").exists():
    from fastapi.responses import FileResponse

    @app.get("/{path:path}", include_in_schema=False)
    def website(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, f"No API endpoint /{path}.")
        file = (WEB_BUILD / path).resolve()
        if path and file.is_file() and WEB_BUILD.resolve() in file.parents:
            return FileResponse(file)
        return FileResponse(WEB_BUILD / "index.html")   # single-page app: the router takes over
