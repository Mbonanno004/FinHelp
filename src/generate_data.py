"""
Synthetic transaction generator for FinHelp.

Produces bank-transaction data in the SAME format as data/synthetic_transactions.csv:

    sample_id, transaction_type, amount, merchant_category, balance_after, transaction_date

so that the model can be trained on many simulated people and then tested on the
original hand-made dataset.

The generator deliberately builds in patterns that a model can learn:
  * Seasonality      - utilities peak in winter/summer, shopping & gifts in Nov/Dec,
                       travel/entertainment in summer.
  * Income effects   - wealthier people spend more; job loss cuts discretionary spending;
                       a raise slowly increases it.
  * Fixed vs. variable costs - rent, insurance, subscriptions, debt payments are constant;
                       groceries, dining, shopping, etc. vary month to month.
  * Trends           - each person's spending drifts slowly (inflation / lifestyle creep).
  * Balance feedback - people cut back when their bank balance gets low.
  * Irregular spikes - travel, taxes, education (hard to predict, as in real life).

Only the Python standard library is used, so this runs without installing anything.

Usage (from the project root):
    python src/generate_data.py                      # 500 people x 24 months
    python src/generate_data.py --people 1000 --months 36 --seed 7
"""

import argparse
import calendar
import csv
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# --------------------------------------------------------------------------------------
# Configuration tables
# --------------------------------------------------------------------------------------

# Day-to-day spending categories: average number of transactions per month.
FREQUENCY = {
    "groceries": 7.0,
    "dining": 7.0,
    "transportation": 4.0,
    "shopping": 3.0,
    "entertainment": 2.5,
    "home & household": 0.6,
    "healthcare": 0.6,
    "kids activities": 2.0,
}

# Categories people keep paying for even when money is tight.
ESSENTIAL = {"groceries", "healthcare", "kids activities"}

# Seasonal multipliers, index 0 = January ... 11 = December. Missing categories = 1.0.
SEASON = {
    "utilities":      [1.25, 1.20, 1.00, 0.95, 0.95, 1.00, 1.20, 1.22, 1.00, 0.95, 0.97, 1.20],
    "shopping":       [0.95, 0.85, 0.95, 1.00, 1.00, 1.00, 1.00, 1.05, 1.00, 0.95, 1.25, 1.45],
    "dining":         [0.90, 0.92, 1.00, 1.00, 1.05, 1.08, 1.10, 1.05, 1.00, 1.00, 1.00, 1.12],
    "entertainment":  [0.90, 0.90, 1.00, 1.00, 1.05, 1.15, 1.20, 1.10, 0.95, 1.00, 1.00, 1.10],
    "groceries":      [1.00, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.08, 1.12],
    "transportation": [0.95, 0.95, 1.00, 1.00, 1.05, 1.10, 1.12, 1.10, 1.00, 1.00, 0.97, 1.00],
}

# Probability of a gifts / charity purchase in each month (holidays, Mother's Day).
GIFT_PROB = [0.03, 0.05, 0.03, 0.03, 0.15, 0.05, 0.03, 0.03, 0.03, 0.05, 0.85, 0.95]

# Probability of a trip in each month (before scaling by a person's love of travel).
TRAVEL_PROB = [0.03, 0.04, 0.05, 0.04, 0.06, 0.12, 0.14, 0.12, 0.05, 0.04, 0.06, 0.09]

SUBSCRIPTION_PRICES = [6.99, 9.99, 10.99, 11.99, 12.99, 13.99, 15.49, 15.99, 17.99, 22.99]

# Person "archetypes". Money values are monthly dollars for a person in the MIDDLE of
# the income range; actual people are scaled up/down by their income.
ARCHETYPES = {
    "student": dict(
        weight=0.12, income=(1000, 1700), pay=["biweekly"],
        housing=("rent", 1.0, (400, 750)),
        spend={"groceries": 180, "dining": 100, "transportation": 60, "shopping": 60,
               "entertainment": 40, "home & household": 15, "healthcare": 15},
        utilities=(50, 90), insurance=(0, 1), subs=(1, 3), debt=0.3, savings=0.25,
        start_balance=(100, 1500),
    ),
    "single": dict(
        weight=0.26, income=(2800, 4800), pay=["biweekly", "semimonthly", "monthly"],
        housing=("rent", 1.0, (900, 1600)),
        spend={"groceries": 380, "dining": 260, "transportation": 140, "shopping": 200,
               "entertainment": 80, "home & household": 40, "healthcare": 40},
        utilities=(150, 250), insurance=(1, 1), subs=(2, 5), debt=0.5, savings=0.6,
        start_balance=(500, 6000),
    ),
    "couple": dict(
        weight=0.18, income=(4500, 7500), pay=["biweekly", "semimonthly"],
        housing=("rent", 0.7, (1300, 2000)),
        spend={"groceries": 600, "dining": 400, "transportation": 180, "shopping": 350,
               "entertainment": 180, "home & household": 60, "healthcare": 60},
        utilities=(260, 400), insurance=(1, 2), subs=(3, 6), debt=0.5, savings=0.7,
        start_balance=(1500, 10000),
    ),
    "family": dict(
        weight=0.20, income=(5500, 9500), pay=["biweekly", "semimonthly"],
        housing=("mortgage", 0.75, (1700, 2700)),
        spend={"groceries": 1100, "dining": 380, "transportation": 260, "shopping": 420,
               "entertainment": 200, "home & household": 60, "healthcare": 110,
               "kids activities": 130},
        utilities=(420, 620), insurance=(2, 2), subs=(3, 6), debt=0.5, savings=0.6,
        start_balance=(1500, 12000),
    ),
    "retiree": dict(
        weight=0.12, income=(2200, 3800), pay=["pension"],
        housing=("rent", 0.35, (800, 1300)),
        spend={"groceries": 420, "dining": 200, "transportation": 80, "shopping": 120,
               "entertainment": 75, "home & household": 40, "healthcare": 150},
        utilities=(300, 420), insurance=(2, 3), subs=(1, 3), debt=0.15, savings=0.2,
        start_balance=(10000, 40000),
    ),
    "freelancer": dict(
        weight=0.12, income=(3000, 6000), pay=["freelance"],
        housing=("rent", 1.0, (1000, 1700)),
        spend={"groceries": 400, "dining": 280, "transportation": 120, "shopping": 220,
               "entertainment": 100, "home & household": 40, "healthcare": 50},
        utilities=(150, 260), insurance=(1, 1), subs=(3, 6), debt=0.4, savings=0.4,
        start_balance=(2000, 12000),
    ),
}

# Order in which same-day transactions are applied to the balance.
TYPE_ORDER = {"deposit": 0, "credit": 0, "refund": 1, "debit": 2, "transfer": 3,
              "withdrawal": 3, "fee": 4, "interest": 5}


# --------------------------------------------------------------------------------------
# Small random helpers
# --------------------------------------------------------------------------------------

def poisson(rng: random.Random, lam: float) -> int:
    """Number of events in a month when `lam` happen on average (Knuth's method)."""
    if lam <= 0:
        return 0
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def noise(rng: random.Random, sigma: float) -> float:
    """Multiplicative noise with an average of 1.0 (log-normal)."""
    return rng.lognormvariate(-0.5 * sigma * sigma, sigma)


def month_iter(start_year: int, start_month: int, n_months: int):
    """Yield (index, year, month) for n consecutive months."""
    y, m = start_year, start_month
    for k in range(n_months):
        yield k, y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


# --------------------------------------------------------------------------------------
# A simulated person
# --------------------------------------------------------------------------------------

@dataclass
class Person:
    sample_id: int
    archetype: str
    monthly_income: float
    pay_schedule: str
    first_payday: date | None
    housing_category: str | None
    housing_cost: float
    utility_bills: list            # [(day, base_amount)]
    insurance: list                # [(day, amount)]
    subscriptions: list            # [(day, amount)]
    debt_payment: tuple | None     # (day, amount)
    childcare: float
    savings_amount: float
    savings_day: int
    spend_base: dict               # category -> monthly base spend
    frequency: dict                # category -> transactions per month
    trend: float                   # monthly growth rate of variable spending
    travel_propensity: float
    cash_rate: float
    apr: float
    maintenance_fee: bool
    side_income: float
    child_support: float
    family_support: float
    start_balance: float
    job_loss: tuple | None = None          # (start_month_index, duration)
    unemployment_amount: float = 0.0
    raise_event: tuple | None = None       # (month_index, factor)
    move_event: tuple | None = None        # (month_index, factor)
    new_subscription: tuple | None = None  # (month_index, day, amount)
    events: list = field(default_factory=list)


def make_person(rng: random.Random, sample_id: int, start: date, n_months: int) -> Person:
    names = list(ARCHETYPES)
    archetype = rng.choices(names, weights=[ARCHETYPES[n]["weight"] for n in names])[0]
    a = ARCHETYPES[archetype]

    lo, hi = a["income"]
    income = rng.uniform(lo, hi)
    wealth = (income / ((lo + hi) / 2)) ** 0.7     # >1 for richer people
    spender = noise(rng, 0.20)                      # frugal (<1) or big spender (>1)

    # Pay schedule
    schedule = rng.choice(a["pay"])
    first_payday = None
    if schedule == "biweekly":
        d = start + timedelta(days=rng.randint(0, 9))
        while d.weekday() != 4:                     # paydays are Fridays
            d += timedelta(days=1)
        first_payday = d

    # Housing
    housing_cat, housing_cost = None, 0.0
    cat, prob, (hlo, hhi) = a["housing"]
    if rng.random() < prob:
        housing_cat = cat
        housing_cost = round(rng.uniform(hlo, hhi) * wealth ** 0.5 / 5) * 5
    elif archetype in ("couple", "family"):
        housing_cat = "mortgage" if cat == "rent" else "rent"
        housing_cost = round(rng.uniform(hlo, hhi) * wealth ** 0.5 / 5) * 5

    # Utilities: split the monthly total into 2-4 bills (electric, water, internet, phone)
    util_total = rng.uniform(*a["utilities"]) * wealth ** 0.4
    n_bills = rng.randint(2, 4)
    weights = [rng.random() + 0.3 for _ in range(n_bills)]
    utility_bills = [(rng.randint(3, 26), util_total * w / sum(weights)) for w in weights]

    insurance = [(rng.randint(1, 20), round(rng.uniform(80, 180) * wealth ** 0.5))
                 for _ in range(rng.randint(*a["insurance"]))]
    subscriptions = [(rng.randint(1, 28), rng.choice(SUBSCRIPTION_PRICES))
                     for _ in range(rng.randint(*a["subs"]))]

    debt = None
    if rng.random() < a["debt"]:
        debt = (rng.randint(5, 25), round(rng.uniform(40, 0.12 * income) / 5) * 5)

    childcare = 0.0
    if archetype == "family" and rng.random() < 0.7:
        childcare = round(rng.uniform(450, 950) / 10) * 10

    savings = 0.0
    if rng.random() < a["savings"]:
        savings = round(rng.uniform(0.03, 0.12) * income / 10) * 10

    spend_base = {c: v * wealth * spender * noise(rng, 0.25) for c, v in a["spend"].items()}
    frequency = {c: FREQUENCY[c] * rng.uniform(0.7, 1.3) for c in spend_base}

    person = Person(
        sample_id=sample_id,
        archetype=archetype,
        monthly_income=income,
        pay_schedule=schedule,
        first_payday=first_payday,
        housing_category=housing_cat,
        housing_cost=housing_cost,
        utility_bills=utility_bills,
        insurance=insurance,
        subscriptions=subscriptions,
        debt_payment=debt,
        childcare=childcare,
        savings_amount=savings,
        savings_day=rng.randint(2, 20),
        spend_base=spend_base,
        frequency=frequency,
        trend=rng.gauss(0.002, 0.003),
        travel_propensity=rng.uniform(0.2, 1.6) * wealth,
        cash_rate=rng.uniform(0.0, 3.0),
        apr=rng.choice([0.0, 0.001, 0.005, 0.01, 0.02, 0.035, 0.04]),
        maintenance_fee=rng.random() < 0.2,
        side_income=rng.uniform(300, 1200) if (archetype != "freelancer" and rng.random() < 0.1) else 0.0,
        child_support=rng.uniform(300, 600) if (archetype == "family" and rng.random() < 0.12) else 0.0,
        family_support=rng.uniform(200, 400) if (archetype == "student" and rng.random() < 0.3) else 0.0,
        start_balance=rng.uniform(*a["start_balance"]),
    )

    fit_to_income(person, rng)

    # ---- Life events ----
    if archetype in ("single", "couple", "family", "freelancer") and n_months >= 8 and rng.random() < 0.12:
        k = rng.randint(3, n_months - 4)
        person.job_loss = (k, rng.randint(2, 5))
        person.unemployment_amount = round(0.4 * income / 5) * 5
        person.events.append(f"job_loss@{k}x{person.job_loss[1]}")
    if schedule in ("biweekly", "semimonthly", "monthly") and n_months >= 8 and rng.random() < 0.3:
        k = rng.randint(6, n_months - 1)
        person.raise_event = (k, rng.uniform(1.04, 1.15))
        person.events.append(f"raise@{k}")
    if housing_cat == "rent" and rng.random() < 0.08:
        k = rng.randint(3, n_months - 1)
        person.move_event = (k, rng.uniform(0.85, 1.25))
        person.events.append(f"move@{k}")
    if rng.random() < 0.3:
        k = rng.randint(1, n_months - 1)
        person.new_subscription = (k, rng.randint(1, 28), rng.choice(SUBSCRIPTION_PRICES))
        person.events.append(f"new_sub@{k}")
    return person


def fit_to_income(p: Person, rng: random.Random) -> None:
    """Real people adapt their lifestyle to their income. Scale this person's spending so
    their expected monthly outflow is 80-100% of their expected income (a few will still
    overspend, which is realistic)."""
    income = p.monthly_income + 0.35 * p.side_income + p.child_support + p.family_support
    fixed = (p.housing_cost + sum(b for _, b in p.utility_bills) + sum(a for _, a in p.insurance)
             + sum(a for _, a in p.subscriptions) + (p.debt_payment[1] if p.debt_payment else 0)
             + p.childcare)
    extras = p.cash_rate * 57 + 25 + 0.07 * p.travel_propensity * 1800 * (p.monthly_income / 4500) ** 0.6
    if p.archetype == "freelancer":
        extras += 0.15 * p.monthly_income
    elif p.archetype == "retiree":
        extras += 217
    elif p.archetype == "student":
        extras += 107
    variable = sum(p.spend_base.values())

    target = income * rng.uniform(0.80, 1.00)
    if fixed + extras + variable + p.savings_amount <= target:
        return
    p.savings_amount = 0.0                       # can't afford a savings habit
    room = target - fixed - extras
    if room < 0.25 * variable:                   # even fixed costs are too high: cheaper housing
        p.housing_cost = round(p.housing_cost * 0.7 / 5) * 5
        p.travel_propensity *= 0.3
        room = max(target - (fixed - 0.3 * p.housing_cost) - extras, 0.25 * variable)
    scale = min(1.0, room / variable)
    p.spend_base = {c: v * scale for c, v in p.spend_base.items()}


# --------------------------------------------------------------------------------------
# Month-by-month simulation
# --------------------------------------------------------------------------------------

def simulate(person: Person, rng: random.Random, start_year: int, start_month: int, n_months: int):
    """Return a list of transaction rows (dicts) for one person."""
    rows = []
    balance = person.start_balance
    p = person

    for k, y, m in month_iter(start_year, start_month, n_months):
        dim = calendar.monthrange(y, m)[1]
        mi = m - 1
        txs = []  # (day, type, amount, category)

        def add(day, ttype, amount, category):
            txs.append((max(1, min(dim, day)), ttype, round(amount, 2), category))

        unemployed = p.job_loss is not None and p.job_loss[0] <= k < p.job_loss[0] + p.job_loss[1]
        income_mult = p.raise_event[1] if (p.raise_event and k >= p.raise_event[0]) else 1.0

        # How freely this person spends this month on "wants"
        discretionary = income_mult ** 0.6
        if unemployed:
            discretionary *= 0.55
        if balance < 0:
            discretionary *= 0.5
        elif balance < 300:
            discretionary *= 0.8
        trend = (1 + p.trend) ** k

        # ---------------- Income ----------------
        if p.pay_schedule in ("biweekly", "semimonthly", "monthly") and not unemployed:
            if p.pay_schedule == "biweekly":
                per_check = p.monthly_income * 12 / 26
                days = [d for d in range(1, dim + 1)
                        if (date(y, m, d) - p.first_payday).days % 14 == 0
                        and date(y, m, d) >= p.first_payday]
            elif p.pay_schedule == "semimonthly":
                per_check, days = p.monthly_income / 2, [15, dim]
            else:
                per_check, days = p.monthly_income, [1]
            for d in days:
                amt = per_check * income_mult
                if rng.random() < 0.2:                       # occasional overtime
                    amt *= 1 + rng.uniform(0.003, 0.015)
                add(d, "deposit", amt, "payroll")
        if p.pay_schedule == "pension":
            add(3, "credit", p.monthly_income, "pension / social security")
        if p.pay_schedule == "freelance":
            n = poisson(rng, 1.8)
            scale = 0.25 if unemployed else 1.0             # "dry spell" instead of job loss
            for _ in range(n):
                add(rng.randint(1, dim), rng.choice(["deposit", "credit"]),
                    p.monthly_income / 1.8 * scale * noise(rng, 0.5), "freelance income")
        if unemployed and p.pay_schedule != "freelance":
            add(5, "credit", p.unemployment_amount, "unemployment benefits")
        if p.side_income and rng.random() < 0.35:
            add(rng.randint(1, dim), "deposit", p.side_income * noise(rng, 0.4), "freelance income")
        if p.child_support:
            add(10, "credit", p.child_support * rng.uniform(0.97, 1.03), "child support")
        if p.family_support:
            add(15, "credit", p.family_support * rng.uniform(0.97, 1.03), "family support")

        # ---------------- Fixed bills ----------------
        if p.housing_category:
            cost = p.housing_cost
            if p.move_event and k >= p.move_event[0]:
                cost = round(cost * p.move_event[1] / 5) * 5
            add(1, "debit", -cost, p.housing_category)
        for day, base in p.utility_bills:
            add(day, "debit", -base * SEASON["utilities"][mi] * noise(rng, 0.05) * (1 + p.trend / 2) ** k,
                "utilities")
        for day, amt in p.insurance:
            add(day, "debit", -amt, "insurance")
        subs = list(p.subscriptions)
        if p.new_subscription and k >= p.new_subscription[0]:
            subs.append(p.new_subscription[1:])
        for day, amt in subs:
            add(day, "debit", -amt, "subscription")
        if p.debt_payment:
            add(p.debt_payment[0], "debit", -p.debt_payment[1], "debt payment")
        if p.childcare:
            add(rng.randint(1, 5), "debit", -p.childcare, "childcare")
        if p.savings_amount and not unemployed and balance > 2 * p.savings_amount:
            add(p.savings_day, "transfer", -p.savings_amount, "savings transfer")
        if p.maintenance_fee:
            add(1, "fee", -5.0, "bank fee")

        # ---------------- Variable spending ----------------
        for cat, base in p.spend_base.items():
            mult = SEASON.get(cat, [1.0] * 12)[mi] * trend * noise(rng, 0.10)
            if cat in ESSENTIAL:
                mult *= 1 - 0.4 * (1 - discretionary)       # essentials are cut less
            else:
                mult *= discretionary
            freq = p.frequency[cat]
            # Habits (groceries, dining...) happen a steady number of times a month;
            # rare purchases (healthcare, home goods) are random, so they use Poisson.
            if freq >= 1.5:
                n_tx = max(0, round(freq * noise(rng, 0.15)))
            else:
                n_tx = poisson(rng, freq)
            for _ in range(n_tx):
                amt = max(3.0, base / freq * mult * noise(rng, 0.35))
                add(rng.randint(1, dim), "debit", -amt, cat)

        # ---------------- Irregular / seasonal one-offs ----------------
        if rng.random() < GIFT_PROB[mi]:
            add(rng.randint(1, dim), "debit", -rng.uniform(80, 180) * (p.monthly_income / 4000) ** 0.4,
                "gifts / charity")
        if not unemployed and rng.random() < TRAVEL_PROB[mi] * p.travel_propensity:
            add(rng.randint(1, dim), "debit", -rng.uniform(600, 3000) * (p.monthly_income / 4500) ** 0.6,
                "travel")
        if p.archetype == "student" and m in (1, 8) and rng.random() < 0.8:
            add(rng.randint(5, 20), "debit", -rng.uniform(400, 1200), "education")
        if p.archetype == "freelancer" and m in (1, 4, 6, 9):
            add(15, "debit", -0.15 * 3 * p.monthly_income * rng.uniform(0.85, 1.15), "taxes")
        if p.archetype == "retiree" and m in (4, 10):
            add(15, "debit", -rng.uniform(1000, 1600), "taxes")

        # ---------------- Cash, refunds ----------------
        for _ in range(poisson(rng, p.cash_rate)):
            add(rng.randint(1, dim), "withdrawal", -float(rng.choice([20, 40, 40, 60, 80, 100])),
                "cash withdrawal")
        if rng.random() < 0.25:
            cat = rng.choice(["shopping", "shopping", "dining", "transportation", "subscription"])
            add(rng.randint(1, dim), "refund", rng.uniform(5, 60), cat)

        # ---------------- Apply to running balance ----------------
        txs.sort(key=lambda t: (t[0], TYPE_ORDER[t[1]]))
        overdraft_charged = False                   # at most one overdraft fee per month
        for day, ttype, amt, cat in txs:
            balance += amt
            rows.append(_row(p.sample_id, ttype, amt, cat, balance, y, m, day))
            if balance < 0 and amt < 0 and not overdraft_charged:
                overdraft_charged = True
                balance -= 35.0
                rows.append(_row(p.sample_id, "fee", -35.0, "bank fee", balance, y, m, day))
        # People move extra cash out of checking once it piles up.
        surplus = balance - 2.5 * p.monthly_income
        if surplus > 0 and rng.random() < 0.5:
            amt = -float(round(surplus * rng.uniform(0.5, 1.0) / 50) * 50)
            if amt < 0:
                balance += amt
                rows.append(_row(p.sample_id, "transfer", amt, "savings transfer", balance, y, m, dim))
        if p.apr > 0 and balance > 0:
            interest = round(balance * p.apr / 12, 2)
            if interest >= 0.01:
                balance += interest
                rows.append(_row(p.sample_id, "interest", interest, "interest", balance, y, m, dim))
    return rows


def _row(sid, ttype, amount, category, balance, y, m, d):
    return {
        "sample_id": sid,
        "transaction_type": ttype,
        "amount": f"{amount:.2f}",
        "merchant_category": category,
        "balance_after": f"{balance:.2f}",
        "transaction_date": f"{m:02d}-{d:02d}-{y}",
    }


# --------------------------------------------------------------------------------------
# Command line entry point
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bank transactions.")
    parser.add_argument("--people", type=int, default=500, help="number of simulated people")
    parser.add_argument("--months", type=int, default=24, help="months of history per person")
    parser.add_argument("--start", default="2024-01", help="first month, YYYY-MM")
    parser.add_argument("--seed", type=int, default=42, help="random seed (same seed = same data)")
    parser.add_argument("--first-id", type=int, default=1001, help="first sample_id to use")
    parser.add_argument("--out", default="data/generated_transactions.csv")
    parser.add_argument("--people-out", default="data/generated_people.csv",
                        help="summary of each simulated person (archetype, income, life events)")
    args = parser.parse_args()

    start_year, start_month = map(int, args.start.split("-"))
    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fields = ["sample_id", "transaction_type", "amount", "merchant_category",
              "balance_after", "transaction_date"]
    n_rows = 0
    with open(args.out, "w", newline="") as f_tx, open(args.people_out, "w", newline="") as f_people:
        tx_writer = csv.DictWriter(f_tx, fieldnames=fields)
        tx_writer.writeheader()
        people_writer = csv.writer(f_people)
        people_writer.writerow(["sample_id", "archetype", "monthly_income", "pay_schedule",
                                "housing", "housing_cost", "savings_transfer", "events"])
        for i in range(args.people):
            sid = args.first_id + i
            person = make_person(rng, sid, date(start_year, start_month, 1), args.months)
            rows = simulate(person, rng, start_year, start_month, args.months)
            tx_writer.writerows(rows)
            n_rows += len(rows)
            people_writer.writerow([sid, person.archetype, round(person.monthly_income, 2),
                                    person.pay_schedule, person.housing_category or "none",
                                    person.housing_cost, person.savings_amount,
                                    ";".join(person.events)])

    print(f"Wrote {n_rows:,} transactions for {args.people} people x {args.months} months -> {args.out}")
    print(f"Wrote person summaries -> {args.people_out}")


if __name__ == "__main__":
    main()
