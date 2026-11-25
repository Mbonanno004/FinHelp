"""
Budget generation: turn predicted spending + income + a savings target into a budget.

This part is a plain rule-based algorithm, NOT machine learning. The model predicts
what someone WILL spend; this decides what they SHOULD spend.

Three numbers per category
  Predicted     what the model expects next month
  Max           the most it's logical to spend there ("max logical spend"):
                  * a guideline share of income (e.g. rent at most 30% of income), and
                  * for flexible categories, never below the least this person can realistically
                    get by on: their typical month minus the largest reasonable cut.
  Recommended   what to actually budget: the prediction, capped at the max, then trimmed
                further if needed to reach the savings target.

Algorithm
  1. Fixed bills (housing, utilities, bills, family) stay at their prediction. If one is
     above its max it is flagged, since it can't be cut within a month.
  2. Flexible categories above their max are trimmed down to it.
  3. If income - spending still doesn't cover the savings target, flexible categories are
     cut further. Each can shrink by at most `max_cut` x its flexibility (dining is easy to
     cut, healthcare is not), shared in proportion to how much each can give.
  4. If that still isn't enough, report the savings that ARE achievable.
"""

from dataclasses import dataclass, field

import pandas as pd

from .data import CATEGORIES, FIXED_CATEGORIES

# How easy each flexible category is to cut back (0 = not at all, 1 = fully).
FLEXIBILITY = {
    "Dining": 1.0,
    "Entertainment": 1.0,
    "Shopping": 1.0,
    "Other": 0.6,
    "Transportation": 0.4,
    "Groceries": 0.3,
    "Healthcare": 0.0,
}

# Max logical spend per category, as a share of monthly income. Rules of thumb based on
# the common "30% rule" for housing and the 50/30/20 budget (50% needs, 30% wants, 20% savings).
GUIDELINE_SHARE = {
    "Housing": 0.30,
    "Utilities": 0.08,
    "Bills": 0.10,
    "Family": 0.12,
    "Groceries": 0.12,
    "Transportation": 0.12,
    "Healthcare": 0.06,
    "Dining": 0.08,
    "Shopping": 0.08,
    "Entertainment": 0.05,
    "Other": 0.08,
}

# Groups shown as a summary. Each has its own guideline ceiling (share of income).
GROUPS = [
    ("Rent / housing", ["Housing"], 0.30),
    ("Other consistent bills", ["Utilities", "Bills", "Family"], 0.20),
    ("All bills", ["Housing", "Utilities", "Bills", "Family"], 0.50),
    ("Everyday needs", ["Groceries", "Transportation", "Healthcare"], 0.25),
    ("Wants", ["Dining", "Shopping", "Entertainment", "Other"], 0.30),
]


@dataclass
class Budget:
    table: pd.DataFrame          # per category: type, predicted, max, recommended, change, status
    groups: list                 # summary rows for GROUPS
    income: float
    goal: float                  # desired savings per month (0 = no target)
    savings: float               # savings this budget actually achieves
    leftover: float              # unallocated money after savings (>= 0)
    goal_met: bool
    messages: list = field(default_factory=list)
    months_to_target: float | None = None


def floor_spend(amount: float, category: str, max_cut: float) -> float:
    """The least that's realistic for a flexible category: `amount` minus the largest cut."""
    return amount * (1 - FLEXIBILITY.get(category, 0.0) * max_cut)


def spending_limits(income: float, typical: dict | None = None, max_cut: float = 0.30) -> dict:
    """Max logical spend per category: guideline share of income, but for flexible categories
    never below what the person can realistically get by on."""
    typical = typical or {}
    limits = {}
    for c in CATEGORIES:
        guide = GUIDELINE_SHARE[c] * max(income, 0.0)
        if c in FIXED_CATEGORIES:
            limits[c] = guide
        else:
            limits[c] = max(guide, floor_spend(max(0.0, typical.get(c, 0.0)), c, max_cut))
    return limits


def make_budget(predicted: dict, income: float, goal: float, max_cut: float = 0.30,
                protected=(), target_amount: float | None = None,
                typical: dict | None = None) -> Budget:
    """
    predicted      category -> predicted dollars for next month
    income         expected income next month
    goal           savings target per month in dollars (0 = no savings target)
    max_cut        largest fraction any category may be cut (scaled by its flexibility)
    protected      categories the user does not want cut (or capped)
    target_amount  optional total savings target (e.g. 6000 for a car), for a timeline
    typical        category -> the person's typical monthly spend (used for the max)
    """
    pred = {c: max(0.0, float(predicted.get(c, 0.0))) for c in CATEGORIES}
    limits = spending_limits(income, typical if typical is not None else pred, max_cut)
    rec = dict(pred)
    messages = []

    # 1. Fixed bills above their max: flag them (they can't be cut this month).
    over = [c for c in FIXED_CATEGORIES if pred[c] > limits[c] + 0.5]
    for c in over:
        pct = GUIDELINE_SHARE[c] * 100
        messages.append(f"{c} (${pred[c]:,.0f}) is above its max logical spend of ${limits[c]:,.0f} "
                        f"({pct:.0f}% of income). It can't be cut within a month, but it leaves less "
                        "room for everything else.")

    # 2. Flexible categories above their max are trimmed down to it.
    capped = [c for c in CATEGORIES if c not in FIXED_CATEGORIES and c not in protected
              and pred[c] > limits[c] + 0.5]
    for c in capped:
        rec[c] = limits[c]
    if capped:
        trimmed = sum(pred[c] - rec[c] for c in capped)
        messages.append(f"Trimmed {', '.join(capped)} down to the max logical spend "
                        f"(${trimmed:,.0f} less than predicted).")

    # 3. Cut flexible categories further if the savings target doesn't fit yet.
    slack = income - sum(rec.values())
    if slack >= goal:
        savings, leftover = goal, slack - goal
        if goal > 0:
            messages.append(f"This budget reaches the ${goal:,.0f} savings target.")
            if leftover > 0.5:
                messages.append(f"About ${leftover:,.0f} is left over. Consider saving it or "
                                "keeping it as a buffer.")
        else:
            messages.append(f"Income covers the recommended spending with ${leftover:,.0f} left over.")
    else:
        need = goal - slack        # money to free up (includes any predicted deficit)
        capacity = {c: max(0.0, rec[c] - floor_spend(pred[c], c, max_cut))
                    for c in CATEGORIES if c not in FIXED_CATEGORIES and c not in protected}
        total_capacity = sum(capacity.values())
        share = min(1.0, need / total_capacity) if total_capacity > 0 else 0.0
        for c, cap in capacity.items():
            rec[c] -= cap * share
        cut = total_capacity * share
        total_rec = sum(rec.values())
        savings = max(0.0, income - total_rec)
        if goal <= 0:      # no target, but spending is predicted to exceed income
            messages.append(f"Predicted spending is ${-slack:,.0f} more than income, so flexible "
                            f"categories were trimmed by ${cut:,.0f} to try to break even.")
            if income < total_rec - 0.5:
                messages.append(f"Even after those cuts, spending is still ${total_rec - income:,.0f} "
                                "more than income. Fixed costs may need attention.")
        elif share < 1.0:
            messages.append(f"Cutting another ${cut:,.0f} from flexible categories reaches the "
                            f"${goal:,.0f} savings target.")
        else:
            messages.append(f"Even with the maximum cuts (${cut:,.0f}), only ${savings:,.0f} of the "
                            f"${goal:,.0f} savings target is reachable.")
            if income < total_rec:
                messages.append(f"Spending is predicted to exceed income by "
                                f"${total_rec - income:,.0f}. Fixed costs may need attention.")
            messages.append("Options: a smaller target, a longer timeline, a higher cut limit, "
                            "or fewer protected categories.")
        savings = min(savings, goal)
        leftover = max(0.0, income - total_rec - savings)

    table = pd.DataFrame({
        "Category": CATEGORIES,
        "Type": ["Fixed" if c in FIXED_CATEGORIES else "Flexible" for c in CATEGORIES],
        "Predicted": [pred[c] for c in CATEGORIES],
        "Max": [limits[c] for c in CATEGORIES],
        "Recommended": [rec[c] for c in CATEGORIES],
    })
    table["Change"] = table["Recommended"] - table["Predicted"]
    table["Status"] = ["Over max" if c in over
                       else "Capped at max" if c in capped and rec[c] >= limits[c] - 0.5
                       else "Cut to save" if rec[c] < pred[c] - 0.5
                       else "OK"
                       for c in CATEGORIES]

    groups = []
    for name, cats, share in GROUPS:
        p, r = sum(pred[c] for c in cats), sum(rec[c] for c in cats)
        cap = share * max(income, 0.0)
        groups.append({"name": name, "categories": cats, "predicted": p, "recommended": r,
                       "max": cap, "share_max": share,
                       "share_recommended": r / income if income > 0 else None,
                       "over": r > cap + 0.5})

    months = None
    if target_amount and savings > 0:
        months = target_amount / savings
        messages.append(f"At ${savings:,.0f}/month you would reach ${target_amount:,.0f} "
                        f"in about {months:.1f} months.")

    balanced = income >= table["Recommended"].sum() - 0.5
    return Budget(table=table, groups=groups, income=income, goal=goal, savings=savings,
                  leftover=leftover, goal_met=savings >= goal - 0.5 and balanced,
                  messages=messages, months_to_target=months)
