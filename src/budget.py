"""
Budget generation: turn predicted spending + income + a savings goal into a budget.

This part is a plain rule-based algorithm, NOT machine learning. The model predicts
what someone WILL spend; this decides what they SHOULD spend.

Algorithm
  1. Fixed costs (housing, utilities, bills, family) are kept at their predicted amount.
  2. Slack = income - all predicted spending.
  3. If slack covers the savings goal, the budget equals the prediction and the extra is
     reported as leftover.
  4. Otherwise, cut flexible categories to close the gap. Each category can be cut by at
     most `max_cut` (e.g. 30%) times its flexibility weight (dining is easy to cut,
     healthcare is not). Cuts are shared in proportion to how much each category can give.
  5. If the maximum cuts still aren't enough, report the savings that ARE achievable.
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


@dataclass
class Budget:
    table: pd.DataFrame          # per category: predicted, recommended, change, type
    income: float
    goal: float                  # desired savings per month
    savings: float               # savings this budget actually achieves
    leftover: float              # unallocated money after savings (>= 0)
    goal_met: bool
    messages: list = field(default_factory=list)
    months_to_target: float | None = None


def make_budget(predicted: dict, income: float, goal: float, max_cut: float = 0.30,
                protected=(), target_amount: float | None = None) -> Budget:
    """
    predicted      category -> predicted dollars for next month
    income         expected income next month
    goal           desired savings per month
    max_cut        largest fraction any category may be cut (scaled by its flexibility)
    protected      categories the user does not want cut
    target_amount  optional total savings target (e.g. 6000 for a car), for a timeline
    """
    pred = {c: max(0.0, float(predicted.get(c, 0.0))) for c in CATEGORIES}
    recommended = dict(pred)
    total_pred = sum(pred.values())
    slack = income - total_pred
    messages = []

    if slack >= goal:
        savings, leftover = goal, slack - goal
        messages.append(f"Your predicted spending already leaves room for the ${goal:,.0f} goal.")
        if leftover > 0:
            messages.append(f"About ${leftover:,.0f} is left over. Consider saving it or "
                            "keeping it as a buffer.")
    else:
        need = goal - slack        # money to free up (includes any predicted deficit)
        capacity = {c: pred[c] * max_cut * FLEXIBILITY.get(c, 0.0)
                    for c in CATEGORIES
                    if c not in FIXED_CATEGORIES and c not in protected}
        total_capacity = sum(capacity.values())
        share = min(1.0, need / total_capacity) if total_capacity > 0 else 0.0
        for c, cap in capacity.items():
            recommended[c] = pred[c] - cap * share
        cut = total_capacity * share
        total_rec = sum(recommended.values())
        savings = max(0.0, income - total_rec)
        leftover = 0.0
        if share < 1.0:
            messages.append(f"Cutting ${cut:,.0f} from flexible categories reaches the "
                            f"${goal:,.0f} goal.")
        else:
            messages.append(f"Even with the maximum cuts (${cut:,.0f}), only ${savings:,.0f} of the "
                            f"${goal:,.0f} goal is reachable.")
            if income < total_rec:
                messages.append(f"Spending is predicted to exceed income by "
                                f"${total_rec - income:,.0f}. Fixed costs may need attention.")
            messages.append("Options: a smaller goal, a longer timeline, a higher cut limit, "
                            "or fewer protected categories.")
        savings = min(savings, goal)
        leftover = max(0.0, income - total_rec - savings)

    table = pd.DataFrame({
        "Category": CATEGORIES,
        "Type": ["Fixed" if c in FIXED_CATEGORIES else "Flexible" for c in CATEGORIES],
        "Predicted": [pred[c] for c in CATEGORIES],
        "Recommended": [recommended[c] for c in CATEGORIES],
    })
    table["Change"] = table["Recommended"] - table["Predicted"]

    months = None
    if target_amount and savings > 0:
        months = target_amount / savings
        messages.append(f"At ${savings:,.0f}/month you would reach ${target_amount:,.0f} "
                        f"in about {months:.1f} months.")

    return Budget(table=table, income=income, goal=goal, savings=savings,
                  leftover=leftover, goal_met=savings >= goal - 0.5,
                  messages=messages, months_to_target=months)
