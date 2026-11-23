"""
FinHelp - interactive app.

Run from the project root:
    streamlit run app.py

Pages
  Start Here       what the app does and how to use it
  Budget Planner   load someone's history -> prediction -> accuracy check -> budget
  Compare Models   score the main model, your own models and baselines side by side
  Train Model      change the neural network's settings and retrain it
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
                      split_people, to_monthly)
from src.evaluate import ACCURACY_EXPLANATION, backtest, category_errors, compare
from src.predict import forecast
from src.train import TrainConfig, save_model, train
from src.zoo import all_predictors, baseline_predictors, nn_predictor_from

ROOT = Path(__file__).parent
GENERATED = ROOT / "data" / "generated_transactions.csv"
ORIGINAL = ROOT / "data" / "synthetic_transactions.csv"
MODEL_PATH = ROOT / "models" / "spending_model.pt"

st.set_page_config(page_title="FinHelp", page_icon="💰", layout="wide")


# --------------------------------------------------------------------------------------
# Cached loaders and small helpers
# --------------------------------------------------------------------------------------

def version(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def models_stamp() -> tuple:
    """Changes whenever a saved model or a custom model file changes, so caches refresh."""
    files = list((ROOT / "models").glob("*")) + list((ROOT / "src" / "custom_models").glob("*.py"))
    return tuple(sorted((f.name, f.stat().st_mtime) for f in files if f.is_file()))


@st.cache_resource(show_spinner="Loading models...")
def get_registry(stamp: tuple):
    return all_predictors(MODEL_PATH)


def registry():
    return get_registry(models_stamp())


@st.cache_data(show_spinner="Preparing test data...")
def get_splits(history: int, data_version: float):
    return load_splits(GENERATED, ORIGINAL, history=history)


@st.cache_data
def get_monthly(path: str, data_version: float) -> pd.DataFrame:
    return to_monthly(load_transactions(path))


@st.cache_data
def get_training_people(data_version: float) -> set:
    monthly = get_monthly(str(GENERATED), data_version)
    return split_people(monthly["sample_id"].unique())[0]


def money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def intro(text: str):
    st.info(text, icon="ℹ️")


def section(title: str, purpose: str):
    st.subheader(title)
    st.caption(purpose)


def totals_chart(per_month: pd.DataFrame, names: list):
    """Line chart of actual vs. predicted total spending per month."""
    series = ["Actual"] + names
    long = per_month.melt(id_vars="Month", value_vars=series, var_name="Series", value_name="Dollars")
    palette = ["#222222", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    return (alt.Chart(long).mark_line(point=True)
            .encode(x=alt.X("Month:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Dollars:Q", title="Total spending ($)", scale=alt.Scale(zero=False)),
                    color=alt.Color("Series:N", sort=series,
                                    scale=alt.Scale(domain=series, range=palette[:len(series)]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Month", "Series", alt.Tooltip("Dollars:Q", format="$,.0f")])
            .properties(height=300))


def pick_person(key: str):
    """Dataset + person selectors. Returns (monthly rows for that person, dataset label, id)."""
    options = {"Original dataset (10 people)": ORIGINAL}
    if GENERATED.exists():
        options["Generated dataset (500 people)"] = GENERATED
    c1, c2 = st.columns(2)
    label = c1.selectbox("Dataset", list(options), key=f"{key}_ds")
    monthly = get_monthly(str(options[label]), version(options[label]))
    sid = c2.selectbox("Person (sample_id)", sorted(monthly["sample_id"].unique()), key=f"{key}_id")
    return monthly[monthly["sample_id"] == sid], label, sid


def training_warning(label: str, sid):
    if label.startswith("Generated") and sid in get_training_people(version(GENERATED)):
        st.warning("This person was in the **training data**, so the model has seen them before and "
                   "the accuracy here is optimistic. Pick someone from the original dataset (or a "
                   "generated person the model didn't train on) for a fair check.", icon="⚠️")


# --------------------------------------------------------------------------------------
# Start Here
# --------------------------------------------------------------------------------------

def guide_page():
    st.title("💰 FinHelp")
    st.markdown("#### Predict next month's spending with machine learning, then turn it into a "
                "budget that reaches your savings goal.")

    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**1. Learn**\n\nA neural network studied 24 months of spending from 350 "
                    "simulated people and learned patterns like winter utility bills, holiday "
                    "shopping and cutting back after a job loss.")
    with c2.container(border=True):
        st.markdown("**2. Predict**\n\nGiven your last 6 months, it predicts what you'll spend "
                    "next month in 11 categories, and shows how accurate it has been on *your* "
                    "past months.")
    with c3.container(border=True):
        st.markdown("**3. Budget**\n\nA simple set of rules (not AI) compares that prediction "
                    "with your income and goal, and trims flexible categories like dining and "
                    "shopping until the goal fits.")

    st.subheader("The pages")
    st.markdown(
        "| Page | What you can do |\n|---|---|\n"
        "| **Budget Planner** | Type in or upload someone's spending, or pick a person from the dataset. "
        "See next month's prediction, get a recommended budget (with an optional savings goal), and "
        "check how accurate the model is for that person. |\n"
        "| **Compare Models** | Put the main neural network, your own models and simple baselines side "
        "by side, scored on the same people none of them trained on. Compare overall, by category, "
        "and for one person. |\n"
        "| **Train Model** | Change the neural network's settings (size, learning rate, history "
        "length…), retrain it with a live progress chart, and decide whether to keep the new "
        "version. |")

    st.subheader("Quick start")
    st.markdown(
        "1. Open **Budget Planner**. An example person is already loaded (open the table to see or edit it).\n"
        "2. Read **Next month's prediction**: the total and the prediction for each category.\n"
        "3. Under **Income and savings goal**, check the income and, if you're saving for something, "
        "turn on **I have a savings goal** and enter it (e.g. $500/month for a car).\n"
        "4. Read **Recommended budget**: what to spend in each category and whether the goal "
        "is reachable.\n"
        "5. Scroll to **How accurate is the model for this person** to see how far the "
        "predictions can be trusted. Then try someone from the dataset.\n"
        "6. Visit **Compare Models** to see how well each model does overall.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Why it's useful")
        st.markdown(
            "- **Budgets from what you actually do.** It starts from a realistic forecast instead "
            "of a generic 50/30/20 rule.\n"
            "- **Shows the trade-off.** It shows how much each category must shrink to hit a goal, "
            "or that the goal isn't reachable yet.\n"
            "- **Honest about accuracy.** Every prediction comes with a check of how well the "
            "model did on that person's past months.\n"
            "- **A test bench.** You can compare different machine-learning methods on equal "
            "footing.")
    with c2:
        st.subheader("Limits to keep in mind")
        st.markdown(
            "- Trained on **simulated** people, so real bank data may behave differently.\n"
            "- Rare one-off costs (trips, taxes) can't really be predicted. The model predicts a "
            "*typical* month, so keep a buffer for surprises.\n"
            "- It needs at least 6 months of history, and 7+ months to check its own accuracy.\n"
            "- This is a class project, not financial advice.")

    with st.expander("Key terms"):
        st.markdown(
            "- **Model:** a program that learned patterns from examples. Here it predicts spending.\n"
            "- **Baseline:** a simple method with no learning, e.g. *next month = average of the "
            "last 3 months*. A model is only worth using if it beats the baselines.\n"
            "- **Training people / test people:** the model learns from training people and is "
            "graded on test people it never saw, the same way an exam uses new questions.\n"
            "- **Average miss (MAE):** on average, how many dollars a prediction is off for one "
            "category in one month.\n"
            "- **Accuracy:** 100% minus the total miss as a share of total spending.\n"
            "- **Backtest:** checking a model on past months by hiding each month and predicting "
            "it from the months before.")


# --------------------------------------------------------------------------------------
# Budget Planner
# --------------------------------------------------------------------------------------

def example_history(months: int = 12) -> pd.DataFrame:
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
        row = {"Month": str(month), "Income": 3000.0}
        for c in CATEGORIES:
            v = base.get(c, 0.0) * season.get(c, [1] * 12)[month.month - 1]
            if c not in FIXED_CATEGORIES and v:
                v *= rng.uniform(0.85, 1.15)
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
    return out.sort_values("month")


def planner_page():
    st.title("Budget Planner")
    intro("Give the app someone's past monthly spending. It **predicts next month**, builds a "
          "**budget** (optionally around a savings goal), and shows **how accurate** its predictions "
          "have been for this person. Work top to bottom.")

    predictors, _, _ = registry()
    choices = {p.name: p for p in predictors if p.kind != "Baseline"}
    if not choices:
        st.warning("No trained model found. Train one on the **Train Model** page first.")
        return
    simple_avg = next(p for p in predictors if p.name == "3-month average")

    # ---- History ----
    section("Spending history",
            "The model learns about this person from their past months. It needs at least 6 months; "
            "with 7 or more it can also check its own accuracy.")
    c1, c2 = st.columns(2)
    source = c1.selectbox("Where should the history come from?",
                          ["Example person (editable)", "Upload a transactions CSV",
                           "Pick someone from the dataset"])
    model = choices[c2.selectbox("Prediction model", list(choices),
                                 help="Your own models appear here once they are added and trained "
                                      "(see Compare Models).")]
    H = max(model.history, 3)
    monthly_person, label, sid = None, "", None

    if source == "Example person (editable)":
        if "manual" not in st.session_state:
            st.session_state["manual"] = example_history()
        with st.expander("Example person's monthly spending (open to view or edit)"):
            st.caption("One row per month, oldest first. Click any cell to change it, or add rows at the bottom.")
            edited = st.data_editor(st.session_state["manual"], num_rows="dynamic", width="stretch",
                                    key="manual_editor", hide_index=True,
                                    height=35 * (len(st.session_state["manual"]) + 2) + 3)
        try:
            monthly_person = manual_to_monthly(edited.dropna(subset=["Month"]))
        except Exception as e:
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
        monthly_person, label, sid = pick_person("planner")

    if monthly_person is None or monthly_person.empty:
        return
    if len(monthly_person) < H:
        st.error(f"This model needs at least {H} months of history; there are {len(monthly_person)}.")
        return
    if source != "Example person (editable)":
        with st.expander("See the monthly totals the model reads"):
            show = monthly_person.assign(month=monthly_person["month"].astype(str)).set_index("month")
            st.dataframe(show[["Income"] + CATEGORIES + ["Total"]].round(0), width="stretch")

    # ---- Prediction ----
    fc = forecast(model, monthly_person)
    recent = monthly_person.sort_values("month").tail(3)
    section(f"Next month's prediction ({fc.attrs['month']})",
            "What the model expects to be spent next month, based on the months above.")
    m1, m2, m3 = st.columns(3)
    pred_total, avg_total = fc["Predicted"].sum(), fc["Recent average"].sum()
    m1.metric("Predicted spending", money(pred_total))
    m2.metric("Recent average (last 3 months)", money(avg_total),
              delta=f"{money(pred_total - avg_total)} predicted vs. average", delta_color="off")
    m3.metric("Recent income (last 3 months)", money(recent["Income"].mean()))
    table = fc.assign(Difference=fc["Predicted"] - fc["Recent average"])
    st.dataframe(table.style.format({"Predicted": "${:,.0f}", "Recent average": "${:,.0f}",
                                     "Difference": "{:+,.0f}"}),
                 hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)

    # ---- Income and optional savings goal ----
    section("Income and savings goal",
            "Set next month's expected income. Turn on a savings goal if you're saving for something.")
    i1, i2 = st.columns([1, 2])
    income = i1.number_input("Expected income next month ($)", min_value=0.0, step=50.0,
                             value=float(round(recent["Income"].mean(), -1)),
                             help="Starts at this person's average income over the last 3 months.")
    i2.markdown("&nbsp;")
    has_goal = i2.toggle("I have a savings goal", key="has_goal")

    goal, goal_name, target, max_cut, protected = 0.0, "", None, 0.30, []
    if has_goal:
        with st.container(border=True):
            g1, g2, g3 = st.columns(3)
            goal_name = g1.text_input("Saving for", placeholder="e.g. Vehicle")
            goal_input = g2.number_input("Save per month ($)", min_value=0.0, step=25.0,
                                         value=None, placeholder="e.g. 500")
            target = g3.number_input("Total needed ($, optional)", min_value=0.0, step=500.0,
                                     value=None, placeholder="e.g. 6000",
                                     help="Used to estimate how many months reaching the goal takes.")
            o1, o2 = st.columns(2)
            max_cut = o1.slider("Largest cut allowed per category", 0, 60, 30, format="%d%%",
                                help="No flexible category is reduced by more than this.") / 100
            flexible = [c for c in CATEGORIES if FLEXIBILITY.get(c, 0) > 0]
            protected = o2.multiselect("Don't cut these", flexible, placeholder="Choose categories")
        if not goal_input:
            st.info("Enter how much you'd like to save per month, and the budget below will be "
                    "built around it.")
        goal = goal_input or 0.0

    # ---- Budget ----
    predicted = dict(zip(fc["Category"], fc["Predicted"]))
    b = make_budget(predicted, income, goal, max_cut, protected, target or None)
    with_goal = goal > 0
    section("Recommended budget" + (f": {goal_name}" if with_goal and goal_name else ""),
            "Fixed costs stay as predicted. If the goal doesn't fit, flexible categories are trimmed."
            if with_goal else
            "Without a savings goal the budget follows the prediction (trimmed only if spending "
            "would exceed income).")
    change = b.table["Change"].sum()
    cols = st.columns(4 if with_goal else 3)
    cols[0].metric("Income", money(b.income))
    cols[1].metric("Spending budget", money(b.table["Recommended"].sum()),
                   delta=f"{money(change)} vs. predicted" if abs(change) >= 0.5 else None,
                   delta_color="inverse")
    if with_goal:
        cols[2].metric("Savings", money(b.savings),
                       delta="goal met" if b.goal_met else f"{money(b.savings - b.goal)} short of goal")
    cols[-1].metric("Left over", money(b.leftover))
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
        st.dataframe(b.table.style.format({"Predicted": "${:,.0f}", "Recommended": "${:,.0f}",
                                           "Change": "{:+,.0f}"}),
                     hide_index=True, width="stretch", height=420)

    with st.expander("How is the budget calculated?"):
        st.markdown(
            "1. The model's prediction is the starting point.\n"
            "2. **Fixed** costs (housing, utilities, bills, family) stay as predicted.\n"
            "3. If income − predicted spending already covers the goal (or there is no goal and "
            "income covers spending), nothing changes.\n"
            "4. Otherwise **flexible** categories are cut. Each can shrink by at most the cut limit × "
            "how flexible it is: "
            + ", ".join(f"{c} {int(v * 100)}%" for c, v in FLEXIBILITY.items())
            + ".\n5. The cuts are shared in proportion to how much each category can give.\n"
            "6. If that still isn't enough, the app reports what *can* be saved.")

    # ---- Accuracy ----
    section("How accurate is the model for this person?",
            "We replay this person's past: each month is hidden, the model predicts it from the "
            "months before, and we compare against what really happened.")
    try:
        per_month, summary = backtest([model, simple_avg], monthly_person)
    except ValueError:
        st.info(f"Add at least {H + 1} months of history to see an accuracy check. The model "
                f"needs {H} earlier months for each month it predicts.")
        return
    s = summary.set_index("model")
    acc, base_acc = s.loc[model.name, "Accuracy %"], s.loc[simple_avg.name, "Accuracy %"]
    a1, a2, a3 = st.columns(3)
    a1.metric("Model accuracy", f"{acc:.0f}%",
              help="100% − (all misses ÷ all actual spending). See the explanation below.")
    a2.metric("Average miss per category", money(s.loc[model.name, "MAE"]),
              help="On average, how many dollars one category's prediction was off in one month.")
    a3.metric("Simple 3-month average", f"{base_acc:.0f}%",
              delta=f"{acc - base_acc:+.1f} points for the model")
    n = int(s.loc[model.name, "months checked"])
    if acc > base_acc:
        st.success(f"Over the last {n} months, the model was more accurate for this person than "
                   "simply averaging their recent months.")
    else:
        st.warning(f"Over the last {n} months, the simple average did as well or better for this "
                   "person, so treat the prediction with care.")
    training_warning(label, sid)
    st.altair_chart(totals_chart(per_month, [model.name, simple_avg.name]), width="stretch")
    with st.expander("How is accuracy calculated?"):
        st.markdown(ACCURACY_EXPLANATION)
        st.markdown("**Month by month (total spending)**")
        st.dataframe(per_month.style.format({c: "${:,.0f}" for c in per_month.columns if c != "Month"}),
                     hide_index=True, width="stretch")


# --------------------------------------------------------------------------------------
# Compare Models
# --------------------------------------------------------------------------------------

ADD_MODEL_HELP = """
1. Copy `src/custom_models/_template.py` to a new file **without** the leading underscore,
   e.g. `src/custom_models/small_nn.py`.
2. Fill in `NAME`, `DESCRIPTION`, `HISTORY`, and the `train()` and `load()` functions.
   Any method works (PyTorch, NumPy, scikit-learn…) as long as `load()` returns a function
   that maps the inputs to predicted dollars for the 11 categories.
3. Train only on `splits["train"]` (or part of it), never on the test people.
4. Train it with `python -m src.custom_models.small_nn` or the **Train** button that appears here.

See `src/custom_models/linear_regression.py` for a complete working example.
"""


def compare_page():
    st.title("Compare Models")
    intro("See how the main neural network stacks up against **other models** (like ones you build "
          "yourselves) and **simple baselines**. Every model is graded on the same people that none "
          "of them trained on, so the comparison is fair.")
    if not GENERATED.exists():
        st.warning("The comparison needs the generated test people. Create them on the **Train Model** page.")
        return

    predictors, untrained, errors = registry()

    # ---- Models ----
    section("Models available", "Everything that can be compared. Your own models show up here automatically.")
    st.dataframe(pd.DataFrame([{"Model": p.name, "Type": p.kind, "Months of history": p.history,
                                "How it works": p.description} for p in predictors]),
                 hide_index=True, width="stretch")
    for name, mod in untrained.items():
        c1, c2 = st.columns([4, 1])
        c1.warning(f"**{name}** is not trained yet.")
        if c2.button("Train", key=f"train_{name}"):
            with st.spinner(f"Training {name}..."):
                try:
                    st.toast(f"Trained {name}: {mod.train()}")
                except Exception as e:
                    st.error(f"Training failed: {e}")
            st.cache_resource.clear()
            st.rerun()
    for mod_name, err in errors.items():
        st.error(f"Could not load `src/custom_models/{mod_name}.py`: {err}")
    with st.expander("How do I add my own model?"):
        st.markdown(ADD_MODEL_HELP)

    names = [p.name for p in predictors]
    default = [p.name for p in predictors if p.kind != "Baseline"] + ["3-month average", "Full-history median"]
    selected = st.multiselect("Models to compare", names, default=[n for n in default if n in names])
    sel = [p for p in predictors if p.name in selected]
    if not sel:
        return
    splits = get_splits(max(p.history for p in sel), version(GENERATED))

    # ---- Overall ----
    section("Overall accuracy",
            f"Scored on {len(splits['test']):,} predictions for {splits['test'].meta['sample_id'].nunique()} "
            "unseen simulated people, and on the original 10-person dataset.")
    res = compare(sel, splits)
    res["dataset"] = res["split"].map({"test": "Unseen simulated people",
                                       "original": "Original dataset"})
    order = [p.name for p in sel]
    low = max(0, int(res["Accuracy %"].min() // 5) * 5 - 5)
    base = alt.Chart(res).encode(
        y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("dataset:N", sort=["Unseen simulated people", "Original dataset"]),
        x=alt.X("Accuracy %:Q", scale=alt.Scale(domain=[low, 100]), title="Accuracy (%)"),
        color=alt.Color("dataset:N", sort=["Unseen simulated people", "Original dataset"],
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["model", "dataset", alt.Tooltip("Accuracy %:Q", format=".1f"),
                 alt.Tooltip("MAE:Q", format="$.2f", title="Avg miss per category")])
    bars = base.mark_bar(clip=True)
    labels = base.mark_text(align="left", dx=3, fontSize=11).encode(
        text=alt.Text("Accuracy %:Q", format=".1f"), color=alt.value("#555"))
    st.altair_chart((bars + labels).properties(height=70 * len(sel)), width="stretch")

    wide = res.pivot(index="model", columns="dataset", values=["Accuracy %", "MAE"])
    wide = wide.reindex(order)[[("Accuracy %", "Unseen simulated people"), ("Accuracy %", "Original dataset"),
                                ("MAE", "Unseen simulated people"), ("MAE", "Original dataset")]
                               if splits["original"] is not None else
                               [("Accuracy %", "Unseen simulated people"), ("MAE", "Unseen simulated people")]]
    wide.columns = [f"{'Accuracy' if m == 'Accuracy %' else 'Avg miss / category'} · {d}"
                    for m, d in wide.columns]
    fmt = {c: ("{:.1f}%" if c.startswith("Accuracy") else "${:,.2f}") for c in wide.columns}
    acc_cols = [c for c in wide.columns if c.startswith("Accuracy")]
    err_cols = [c for c in wide.columns if not c.startswith("Accuracy")]
    st.dataframe(wide.style.format(fmt)
                 .highlight_max(subset=acc_cols, color="#2e7d3240")
                 .highlight_min(subset=err_cols, color="#2e7d3240"),
                 width="stretch")
    st.caption("Green = best in each column. Accuracy: higher is better. Misses: lower is better.")
    with st.expander("How is accuracy calculated?"):
        st.markdown(ACCURACY_EXPLANATION)

    # ---- By category ----
    section("Accuracy by category", "Average dollars off per month in each category. This shows *where* each model wins or loses.")
    ds = st.radio("Dataset", ["Unseen simulated people", "Original dataset"], horizontal=True, key="cat_ds")
    w = splits["test"] if ds.startswith("Unseen") else splits["original"]
    if w is None:
        st.info("A selected model needs more history than the original dataset's 12 months allow.")
    else:
        cat = category_errors(sel, w)
        st.dataframe(cat.style.format("${:,.1f}").highlight_min(subset=order, axis=1, color="#2e7d3240"),
                     width="stretch", height=35 * (len(cat) + 1) + 3)
        st.caption("Green = the model closest to the truth in that category.")

    # ---- One person ----
    section("Accuracy for one person", "Replay one person's past months with every selected model.")
    person, label, sid = pick_person("compare")
    try:
        per_month, summary = backtest(sel, person)
    except ValueError:
        st.info("This person doesn't have enough months for the selected models.")
        return
    training_warning(label, sid)
    st.dataframe(summary.set_index("model")[["Accuracy %", "MAE", "Total MAE", "months checked"]]
                 .rename(columns={"MAE": "Avg miss / category", "Total MAE": "Avg miss / month total"})
                 .style.format({"Accuracy %": "{:.1f}%", "Avg miss / category": "${:,.2f}",
                                "Avg miss / month total": "${:,.0f}"}),
                 width="stretch")
    st.altair_chart(totals_chart(per_month, order), width="stretch")


# --------------------------------------------------------------------------------------
# Train Model
# --------------------------------------------------------------------------------------

def generate_data_form():
    st.warning("Training needs the generated dataset (data/generated_transactions.csv). Create it here "
               "(a few seconds) or run `python src/generate_data.py`.")
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
    df = pd.DataFrame({"Epoch": epochs_done, "Validation miss ($)": val_mae})
    return (alt.Chart(df).mark_line()
            .encode(x=alt.X("Epoch:Q", scale=alt.Scale(domain=[1, max_epochs])),
                    y=alt.Y("Validation miss ($):Q", scale=alt.Scale(zero=False)),
                    tooltip=["Epoch", alt.Tooltip("Validation miss ($):Q", format="$.2f")])
            .properties(height=260))


def train_page():
    st.title("Train Model")
    intro("Experiment with the main **neural network**. Change its settings, train it (about 30–60 "
          "seconds), and see whether it beats the current version and the simple baselines. Nothing "
          "else in the app changes unless you click **Save**.")
    if not GENERATED.exists():
        generate_data_form()
        return

    predictors, _, _ = registry()
    current = next((p for p in predictors if p.kind == "Neural network"), None)

    # ---- Settings ----
    section("Settings", "Each setting changes how the network learns. Hover over ⓘ for a plain-English explanation.")
    d = TrainConfig()
    c1, c2, c3, c4 = st.columns(4)
    history = c1.slider("Months of history", 3, 9, d.history,
                        help="How many past months the model looks at to predict the next one.")
    hidden = c2.select_slider("Neurons per layer", [16, 32, 64, 128, 256, 512], d.hidden,
                              help="How much 'brain power' each layer has. More can learn more patterns "
                                   "but is slower and can memorize instead of learn.")
    layers = c3.slider("Hidden layers", 1, 4, d.layers,
                       help="How many layers of pattern-finding are stacked. More layers can combine "
                            "simple patterns into complex ones.")
    dropout = c4.slider("Dropout", 0.0, 0.5, d.dropout, 0.05,
                        help="Fraction of neurons randomly switched off while practicing, so the model "
                             "can't rely on memorizing. Higher = stronger protection, slower learning.")
    c5, c6, c7, c8 = st.columns(4)
    lr = c5.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], d.lr,
                          format_func=lambda v: f"{v:g}",
                          help="How big a step the model takes each time it corrects itself. Too big "
                               "and it overshoots; too small and it learns slowly.")
    epochs = c6.slider("Max epochs", 10, 400, d.epochs, 10,
                       help="Maximum number of passes through all the training examples. Training "
                            "stops early if the validation score stops improving.")
    loss = c7.selectbox("Loss function", ["l1", "huber", "mse"],
                        help="How mistakes are scored while learning. l1 = average miss (predicts the "
                             "typical month); mse = squared miss (reacts more to big spikes).")
    seed = c8.number_input("Random seed", 0, 10_000, d.seed,
                           help="Changes the random starting point. The test people stay the same.")
    config = TrainConfig(history=history, hidden=hidden, layers=layers, dropout=dropout,
                         lr=lr, epochs=epochs, loss=loss, seed=int(seed))

    # ---- Train ----
    section("Train", "The line shows the average miss on 75 validation people after each pass. "
                       "Down is good; a flat line means it has stopped improving.")
    if st.button("Train", type="primary"):
        splits = get_splits(history, version(GENERATED))
        progress = st.progress(0.0, text="Training...")
        chart = st.empty()
        log = {"epoch": [], "val": []}

        def on_epoch(epoch, train_loss, val_mae):
            log["epoch"].append(epoch)
            log["val"].append(val_mae)
            progress.progress(epoch / epochs, text=f"Pass {epoch}/{epochs}: validation miss ${val_mae:.2f}")
            chart.altair_chart(loss_chart(log["epoch"], log["val"], epochs), width="stretch")

        model, hist = train(config, splits, on_epoch=on_epoch, verbose=False)
        progress.progress(1.0, text=f"Done in {hist['seconds']:.0f}s after {hist['epoch'][-1]} passes. "
                                    f"Best validation miss ${hist['best_val_mae']:.2f}")
        st.session_state["trained"] = (model, config, hist)

    # ---- Results ----
    if "trained" not in st.session_state:
        return
    model, cfg, hist = st.session_state["trained"]
    section("Results", "The new model next to the version the app uses now and the baselines, on "
                         "people none of them trained on.")
    new = nn_predictor_from(model, cfg, name="New model (not saved)")
    contenders = [new] + ([current] if current else []) + [p for p in baseline_predictors()
                                                           if p.name in ("3-month average", "Full-history median")]
    splits = get_splits(max(p.history for p in contenders), version(GENERATED))
    res = compare(contenders, splits)
    wide = res.pivot(index="model", columns="split", values="Accuracy %").reindex([p.name for p in contenders])
    wide = wide[[c for c in ("test", "original") if c in wide.columns]]
    wide = wide.rename(columns={"test": "Accuracy · unseen simulated people", "original": "Accuracy · original dataset"})
    st.dataframe(wide.style.format("{:.1f}%").highlight_max(color="#2e7d3240"), width="stretch")
    if st.button("Save as the app's model", help="Replaces models/spending_model.pt"):
        save_model(model, cfg, hist, MODEL_PATH)
        st.cache_resource.clear()
        st.success("Saved. The Budget Planner and Compare Models pages now use this model.")


# --------------------------------------------------------------------------------------

pages = [
    st.Page(guide_page, title="Start Here", icon="🏠", url_path="start", default=True),
    st.Page(planner_page, title="Budget Planner", icon="🧾", url_path="planner"),
    st.Page(compare_page, title="Compare Models", icon="📊", url_path="compare"),
    st.Page(train_page, title="Train Model", icon="🧠", url_path="train"),
]
st.navigation(pages, position="top").run()
