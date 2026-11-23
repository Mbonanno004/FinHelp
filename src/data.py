"""
Data pipeline: raw transactions -> monthly spending table -> training examples.

Step 1  load_transactions()   read the CSV, parse dates
Step 2  to_monthly()          one row per (person, month), one column per budget category
Step 3  make_windows()        sliding windows: "last H months" -> "next month"
Step 4  split_people()        train / validation / test split BY PERSON, so the model is
                              always tested on people it has never seen.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Budget categories: the raw merchant categories grouped into what a budget shows.
# --------------------------------------------------------------------------------------
CATEGORY_GROUPS = {
    "Housing":        ["rent", "mortgage"],
    "Utilities":      ["utilities"],
    "Bills":          ["insurance", "subscription", "debt payment"],
    "Groceries":      ["groceries"],
    "Dining":         ["dining"],
    "Transportation": ["transportation"],
    "Shopping":       ["shopping", "home & household"],
    "Entertainment":  ["entertainment"],
    "Family":         ["childcare", "kids activities"],
    "Healthcare":     ["healthcare"],
    "Other":          ["travel", "gifts / charity", "education", "taxes",
                       "cash withdrawal", "bank fee"],
}
CATEGORIES = list(CATEGORY_GROUPS)                   # fixed column order used everywhere
MERCHANT_TO_GROUP = {m: g for g, ms in CATEGORY_GROUPS.items() for m in ms}

SPEND_TYPES = {"debit", "withdrawal", "fee"}         # money spent
INCOME_TYPES = {"deposit", "credit"}                 # money earned

# Which budget categories are fixed obligations vs. flexible (used by the budget step).
FIXED_CATEGORIES = ["Housing", "Utilities", "Bills", "Family"]
FLEXIBLE_CATEGORIES = [c for c in CATEGORIES if c not in FIXED_CATEGORIES]


# --------------------------------------------------------------------------------------
# Step 1 + 2: transactions -> monthly table
# --------------------------------------------------------------------------------------

def load_transactions(path_or_buffer) -> pd.DataFrame:
    """Read a transactions CSV in the project's format."""
    df = pd.read_csv(path_or_buffer)
    required = {"sample_id", "transaction_type", "amount", "merchant_category", "transaction_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%m-%d-%Y")
    df["transaction_type"] = df["transaction_type"].str.strip().str.lower()
    df["merchant_category"] = df["merchant_category"].str.strip().str.lower()
    return df


def to_monthly(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions into one row per person per month.

    Columns: sample_id, month, Income, Savings, <one column per category>, Total
    * Spending is positive dollars.
    * Refunds are subtracted from the category they belong to.
    * Months with no activity in a category get 0.
    """
    tx = tx.copy()
    tx["month"] = tx["transaction_date"].dt.to_period("M")
    tx["group"] = tx["merchant_category"].map(MERCHANT_TO_GROUP).fillna("Other")

    spend = tx[tx["transaction_type"].isin(SPEND_TYPES) & (tx["amount"] < 0)].copy()
    spend["value"] = -spend["amount"]
    refunds = tx[tx["transaction_type"] == "refund"].copy()
    refunds["value"] = -refunds["amount"].abs()
    spending = pd.concat([spend, refunds])
    table = spending.pivot_table(index=["sample_id", "month"], columns="group",
                                 values="value", aggfunc="sum", fill_value=0.0)
    table = table.reindex(columns=CATEGORIES, fill_value=0.0).clip(lower=0.0)

    income = (tx[tx["transaction_type"].isin(INCOME_TYPES)]
              .groupby(["sample_id", "month"])["amount"].sum().rename("Income"))
    savings = (tx[tx["transaction_type"] == "transfer"]
               .groupby(["sample_id", "month"])["amount"].sum().abs().rename("Savings"))

    # Make sure every person has every month between their first and last transaction.
    spans = tx.groupby("sample_id")["month"].agg(["min", "max"])
    full_index = pd.MultiIndex.from_tuples(
        [(sid, m) for sid, row in spans.iterrows()
         for m in pd.period_range(row["min"], row["max"], freq="M")],
        names=["sample_id", "month"])

    monthly = (table.join(income, how="outer").join(savings, how="outer")
               .reindex(full_index).fillna(0.0))
    monthly["Total"] = monthly[CATEGORIES].sum(axis=1)
    monthly = monthly[["Income", "Savings"] + CATEGORIES + ["Total"]].reset_index()
    return monthly


# --------------------------------------------------------------------------------------
# Step 3: sliding windows
# --------------------------------------------------------------------------------------

@dataclass
class Windows:
    """Supervised-learning examples built from the monthly table."""
    history: np.ndarray    # (N, H, F)  last H months of [categories..., Income]
    month: np.ndarray      # (N,)       calendar month (1-12) of the month being predicted
    target: np.ndarray     # (N, C)     actual spending per category in that month
    meta: pd.DataFrame     # sample_id and target month for each example

    def __len__(self):
        return len(self.target)

    def subset(self, mask) -> "Windows":
        mask = np.asarray(mask)
        return Windows(self.history[mask], self.month[mask], self.target[mask],
                       self.meta[mask].reset_index(drop=True))


HISTORY_FEATURES = CATEGORIES + ["Income"]


def make_windows(monthly: pd.DataFrame, history: int = 6) -> Windows:
    """
    For every person and every month t that has `history` months before it, create one
    example: input = months t-history .. t-1, target = month t.
    """
    hist, months, targets, meta = [], [], [], []
    for sid, person in monthly.groupby("sample_id", sort=False):
        person = person.sort_values("month")
        values = person[HISTORY_FEATURES].to_numpy(dtype=np.float32)
        spend = person[CATEGORIES].to_numpy(dtype=np.float32)
        periods = person["month"].tolist()
        for t in range(history, len(person)):
            hist.append(values[t - history:t])
            targets.append(spend[t])
            months.append(periods[t].month)
            meta.append((sid, str(periods[t])))
    if not hist:
        raise ValueError(f"No person has more than {history} months of data.")
    return Windows(
        history=np.stack(hist),
        month=np.array(months, dtype=np.int64),
        target=np.stack(targets),
        meta=pd.DataFrame(meta, columns=["sample_id", "target_month"]),
    )


# --------------------------------------------------------------------------------------
# Step 4: split by person
# --------------------------------------------------------------------------------------

SPLIT_SEED = 0   # fixed, so every model (yours and your teammates') uses the SAME test people


def split_people(sample_ids, val_frac=0.15, test_frac=0.15, seed=SPLIT_SEED):
    """Randomly assign each person to train / val / test. Returns three sets of ids."""
    ids = np.array(sorted(set(sample_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    n_val = int(round(len(ids) * val_frac))
    test, val, train = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]
    return set(train.tolist()), set(val.tolist()), set(test.tolist())


_ROOT = Path(__file__).resolve().parents[1]


def load_splits(generated_path=_ROOT / "data" / "generated_transactions.csv",
                original_path=_ROOT / "data" / "synthetic_transactions.csv",
                history=6):
    """
    Convenience loader used by training and evaluation.
    Returns dict with Windows for 'train', 'val', 'test' (held-out generated people)
    and 'original' (the 10 people from the hand-made dataset; None if `history` is
    too long for its 12 months).
    """
    gen = make_windows(to_monthly(load_transactions(generated_path)), history)
    train_ids, val_ids, test_ids = split_people(gen.meta["sample_id"])
    ids = gen.meta["sample_id"]
    splits = {
        "train": gen.subset(ids.isin(train_ids)),
        "val": gen.subset(ids.isin(val_ids)),
        "test": gen.subset(ids.isin(test_ids)),
    }
    try:
        splits["original"] = make_windows(to_monthly(load_transactions(original_path)), history)
    except ValueError:
        splits["original"] = None
    return splits
