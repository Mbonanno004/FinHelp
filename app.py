"""
FinHelp - interactive app.

Run from the project root:
    streamlit run app.py

Pages
  Budget Planner    enter or load someone's history, predict next month, build a budget
  Train & Evaluate  change model settings, train with a live loss curve, compare to baselines
"""

import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.budget import FLEXIBILITY, make_budget
from src.data import (CATEGORIES, FIXED_CATEGORIES, load_splits, load_transactions,
                      to_monthly)
from src.evaluate import category_table, predictors_for, score_all
from src.predict import forecast
from src.train import TrainConfig, load_model, save_model, train

ROOT = Path(__file__).parent
GENERATED = ROOT / "data" / "generated_transactions.csv"
ORIGINAL = ROOT / "data" / "synthetic_transactions.csv"
MODEL_PATH = ROOT / "models" / "spending_model.pt"

st.set_page_config(page_title="FinHelp", page_icon="💰", layout="wide")


# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Building training examples...")
def get_splits(history: int, seed: int, data_version: float):
    return load_splits(GENERATED, ORIGINAL, history=history, seed=seed)


@st.cache_resource(show_spinner="Loading model...")
def get_model(model_version: float):
    return load_model(MODEL_PATH)


@st.cache_data
def get_monthly(path: str, version: float) -> pd.DataFrame:
    return to_monthly(load_transactions(path))


def version(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


# --------------------------------------------------------------------------------------
# Budget Planner page
# --------------------------------------------------------------------------------------

def example_history(history: int) -> pd.DataFrame:
    """The example person from the project proposal: $3,000 income, $2,300 spending."""
    base = {"Housing": 1000, "Groceries": 350, "Dining": 200, "Transportation": 150,
            "Utilities": 150, "Entertainment": 100, "Other": 350}
    end = pd.Timestamp.today().to_period("M") - 1
    rows = []
    rng = np.random.default_rng(0)
    for i in range(history):
        month = end - (history - 1 - i)
        row = {"Month": str(month), "Income": 3000.0}
        for c in CATEGORIES:
            v = base.get(c, 0.0)
            if c not in FIXED_CATEGORIES and v:
                v *= rng.uniform(0.9, 1.1)
            row[c] = round(float(v), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def manual_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = pd.PeriodIndex(out.pop("Month").astype(str), freq="M")
    out["sample_id"] = 0
    out[CATEGORIES + ["Income"]] = out[CATEGORIES + ["Income"]].fillna(0.0).astype(float)
    out["Savings"] = 0.0
    out["Total"] = out[CATEGORIES].sum(axis=1)
    return out


def budget_planner():
    st.title("Budget Planner")
    if not MODEL_PATH.exists():
        st.warning("No trained model found. Train one on the **Train & Evaluate** page first.")
        return
    model, config, _ = get_model(version(MODEL_PATH))
    H = config.history

    # ---- 1. Spending history ----
    st.subheader("1. Spending history")
    source = st.radio("Where should the history come from?",
                      ["Enter it by hand", "Upload a transactions CSV", "Pick someone from the dataset"],
                      horizontal=True)
    monthly_person = None

    if source == "Enter it by hand":
        st.caption(f"One row per month, oldest first. The model uses the last {H} months. "
                   "Edit any cell; you can also add rows.")
        if "manual" not in st.session_state or len(st.session_state["manual"]) < H:
            st.session_state["manual"] = example_history(H)
        edited = st.data_editor(st.session_state["manual"], num_rows="dynamic",
                                width="stretch", key="manual_editor")
        try:
            monthly_person = manual_to_monthly(edited.dropna(subset=["Month"]))
        except Exception as e:  # bad month text etc.
            st.error(f"Could not read the table: {e}. Months must look like 2026-05.")

    elif source == "Upload a transactions CSV":
        st.caption("Same columns as data/synthetic_transactions.csv: sample_id, transaction_type, "
                   "amount, merchant_category, balance_after, transaction_date (MM-DD-YYYY).")
        file = st.file_uploader("Transactions CSV", type="csv")
        if file is not None:
            try:
                monthly = to_monthly(load_transactions(file))
                sid = st.selectbox("Person (sample_id)", sorted(monthly["sample_id"].unique()))
                monthly_person = monthly[monthly["sample_id"] == sid]
            except Exception as e:
                st.error(f"Could not read that file: {e}")

    else:
        options = {"Original dataset (10 people)": ORIGINAL}
        if GENERATED.exists():
            options["Generated dataset"] = GENERATED
        name = st.selectbox("Dataset", list(options))
        monthly = get_monthly(str(options[name]), version(options[name]))
        sid = st.selectbox("Person (sample_id)", sorted(monthly["sample_id"].unique()))
        monthly_person = monthly[monthly["sample_id"] == sid]

    if monthly_person is None or monthly_person.empty:
        return
    if len(monthly_person) < H:
        st.error(f"The model needs at least {H} months of history; this person has "
                 f"{len(monthly_person)}.")
        return

    recent = monthly_person.sort_values("month").tail(H)
    with st.expander(f"Monthly summary used by the model (last {H} months)"):
        show = recent.assign(month=recent["month"].astype(str)).set_index("month")
        st.dataframe(show[["Income"] + CATEGORIES + ["Total"]].round(0), width="stretch")

    # ---- 2. Prediction ----
    fc = forecast(model, monthly_person, H)
    st.subheader(f"2. Predicted spending for {fc.attrs['month']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted total (neural network)", money(fc["Neural network"].sum()))
    c2.metric("3-month average total", money(fc["3-month average"].sum()))
    c3.metric(f"Average income (last {H} months)", money(recent["Income"].mean()))

    # ---- 3. Goal ----
    st.subheader("3. Your goal")
    g1, g2, g3 = st.columns(3)
    goal_name = g1.text_input("What are you saving for?", "Vehicle")
    income = g2.number_input("Expected income next month ($)", min_value=0.0, step=50.0,
                             value=float(round(recent["Income"].mean(), -1)))
    goal = g3.number_input("Savings goal per month ($)", min_value=0.0, step=25.0, value=500.0)
    g4, g5, g6 = st.columns(3)
    target = g4.number_input(f"Total needed for '{goal_name}' ($, optional)", min_value=0.0,
                             step=500.0, value=0.0)
    max_cut = g5.slider("Largest cut allowed per category", 0, 60, 30, format="%d%%") / 100
    flexible = [c for c in CATEGORIES if FLEXIBILITY.get(c, 0) > 0]
    protected = g6.multiselect("Don't cut these", flexible)

    # ---- 4. Budget ----
    predicted = dict(zip(fc["Category"], fc["Neural network"]))
    b = make_budget(predicted, income, goal, max_cut, protected, target or None)

    st.subheader(f"4. Recommended budget: {goal_name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Income", money(b.income))
    change = b.table["Change"].sum()
    m2.metric("Budgeted spending", money(b.table["Recommended"].sum()),
              delta=f"{money(change)} vs. predicted" if abs(change) >= 0.5 else None,
              delta_color="inverse")
    m3.metric("Savings", money(b.savings),
              delta="goal met" if b.goal_met else f"{money(b.savings - b.goal)} short of goal")
    m4.metric("Unallocated", money(b.leftover))
    for msg in b.messages:
        (st.success if b.goal_met else st.warning)(msg)

    left, right = st.columns([3, 2])
    with left:
        long = b.table.melt(id_vars="Category", value_vars=["Predicted", "Recommended"],
                            var_name="Series", value_name="Dollars")
        chart = (alt.Chart(long).mark_bar()
                 .encode(y=alt.Y("Category:N", sort=CATEGORIES, title=None),
                         yOffset="Series:N",
                         x=alt.X("Dollars:Q", title="Dollars per month"),
                         color=alt.Color("Series:N", legend=alt.Legend(orient="bottom", title=None)),
                         tooltip=["Category", "Series", alt.Tooltip("Dollars:Q", format="$,.0f")])
                 .properties(height=420))
        st.altair_chart(chart, width="stretch")
    with right:
        st.dataframe(
            b.table.style.format({"Predicted": "${:,.0f}", "Recommended": "${:,.0f}",
                                  "Change": "{:+,.0f}"}),
            hide_index=True, width="stretch", height=420)

    with st.expander("How is this budget calculated?"):
        st.markdown(
            "1. The **neural network** predicts what you'll spend in each category next month.\n"
            "2. **Fixed** costs (housing, utilities, bills, family) stay as predicted.\n"
            "3. If income minus predicted spending covers your goal, nothing changes.\n"
            "4. Otherwise **flexible** categories are cut, each by at most the cut limit × "
            "how flexible it is:\n\n"
            + ", ".join(f"{c} {int(v * 100)}%" for c, v in FLEXIBILITY.items())
            + "\n\n5. Cuts are shared in proportion to how much each category can give."
        )
        st.dataframe(fc.round(0), hide_index=True)


# --------------------------------------------------------------------------------------
# Train & Evaluate page
# --------------------------------------------------------------------------------------

def generate_data_form():
    st.info("Training needs the generated dataset (data/generated_transactions.csv). "
            "Create it here or run `python src/generate_data.py`.")
    c1, c2, c3 = st.columns(3)
    people = c1.number_input("People", 50, 5000, 500, step=50)
    months = c2.number_input("Months per person", 12, 60, 24)
    seed = c3.number_input("Seed", 0, 10_000, 42)
    if st.button("Generate training data", type="primary"):
        with st.spinner("Generating..."):
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "generate_data.py"), "--people", str(people),
                 "--months", str(months), "--seed", str(seed),
                 "--out", str(GENERATED), "--people-out", str(ROOT / "data" / "generated_people.csv")],
                capture_output=True, text=True)
        if result.returncode == 0:
            st.success(result.stdout)
            st.rerun()
        else:
            st.error(result.stderr)


def loss_chart(epochs_done, val_mae, max_epochs):
    """Validation error per epoch. The y-axis doesn't start at 0, so small gains are visible."""
    df = pd.DataFrame({"Epoch": epochs_done, "Validation MAE ($)": val_mae})
    return (alt.Chart(df).mark_line()
            .encode(x=alt.X("Epoch:Q", scale=alt.Scale(domain=[1, max_epochs])),
                    y=alt.Y("Validation MAE ($):Q", scale=alt.Scale(zero=False)),
                    tooltip=["Epoch", alt.Tooltip("Validation MAE ($):Q", format="$.2f")])
            .properties(height=260))


def results_section(model, splits, title):
    st.markdown(f"#### {title}")
    scores = score_all(predictors_for(model), splits)
    wide = scores.pivot(index="method", columns="split", values="MAE")
    wide = wide.reindex(list(predictors_for(model)))[["test", "original"]]
    wide.columns = ["Test: unseen simulated people", "Original dataset"]
    st.dataframe(wide.style.format("${:.2f}").highlight_min(axis=0, color="#2e7d3240"),
                 width="stretch")
    st.caption("Average dollars off per category per month (MAE). Lower is better.")
    split = st.radio("Per-category detail for", ["test", "original"], horizontal=True,
                     key=f"split_{title}")
    st.dataframe(category_table(splits, model, split).style.format("${:.1f}"), width="stretch")


def train_page():
    st.title("Train & Evaluate")
    if not GENERATED.exists():
        generate_data_form()
        return

    with st.expander("Saved model (used by the Budget Planner)", expanded=False):
        if MODEL_PATH.exists():
            model, config, hist = get_model(version(MODEL_PATH))
            st.json(vars(config), expanded=False)
            splits = get_splits(config.history, config.seed, version(GENERATED))
            results_section(model, splits, "Saved model vs. baselines")
            if hist.get("epoch"):
                st.altair_chart(loss_chart(hist["epoch"], hist["val_mae"], hist["epoch"][-1]),
                                width="stretch")
        else:
            st.write("No saved model yet.")

    st.subheader("Train a new model")
    d = TrainConfig()
    c1, c2, c3, c4 = st.columns(4)
    history = c1.slider("Months of history", 3, 9, d.history,
                        help="How many past months the model sees")
    hidden = c2.select_slider("Neurons per layer", [16, 32, 64, 128, 256, 512], d.hidden)
    layers = c3.slider("Hidden layers", 1, 4, d.layers)
    dropout = c4.slider("Dropout", 0.0, 0.5, d.dropout, 0.05)
    c5, c6, c7, c8 = st.columns(4)
    lr = c5.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], d.lr,
                          format_func=lambda v: f"{v:g}")
    epochs = c6.slider("Max epochs", 10, 400, d.epochs, 10)
    loss = c7.selectbox("Loss function", ["l1", "huber", "mse"],
                        help="l1 = mean absolute error (matches how we score the model)")
    seed = c8.number_input("Random seed", 0, 10_000, d.seed)
    config = TrainConfig(history=history, hidden=hidden, layers=layers, dropout=dropout,
                         lr=lr, epochs=epochs, loss=loss, seed=int(seed))

    if st.button("Train", type="primary"):
        splits = get_splits(history, int(seed), version(GENERATED))
        st.caption(f"{len(splits['train']):,} training examples · {len(splits['val']):,} validation "
                   f"· {len(splits['test']):,} test")
        progress = st.progress(0.0, text="Training...")
        chart = st.empty()
        log = {"epoch": [], "train loss": [], "validation MAE ($)": []}

        def on_epoch(epoch, train_loss, val_mae):
            log["epoch"].append(epoch)
            log["train loss"].append(train_loss)
            log["validation MAE ($)"].append(val_mae)
            progress.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}: validation MAE ${val_mae:.2f}")
            chart.altair_chart(loss_chart(log["epoch"], log["validation MAE ($)"], epochs),
                               width="stretch")

        model, hist = train(config, splits, on_epoch=on_epoch, verbose=False)
        progress.progress(1.0, text=f"Done in {hist['seconds']:.0f}s after {hist['epoch'][-1]} epochs. "
                                    f"Best validation MAE ${hist['best_val_mae']:.2f}")
        st.session_state["trained"] = (model, config, hist)

    if "trained" in st.session_state:
        model, cfg, hist = st.session_state["trained"]
        splits = get_splits(cfg.history, cfg.seed, version(GENERATED))
        results_section(model, splits, "New model vs. baselines")
        if st.button("Save as the app's model", help=f"Overwrites {MODEL_PATH.name}"):
            save_model(model, cfg, hist, MODEL_PATH)
            st.cache_resource.clear()
            st.success("Saved. The Budget Planner now uses this model.")


# --------------------------------------------------------------------------------------

PAGES = {"Budget Planner": budget_planner, "Train & Evaluate": train_page}
with st.sidebar:
    st.markdown("## 💰 FinHelp")
    st.caption("Predict next month's spending with a neural network, then build a budget "
               "around your savings goal.")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
PAGES[page]()
