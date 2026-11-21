# FinHelp
A simple artificial intelligence program to help predict and budget future expenses.

## Setup (Windows, PowerShell)

Requires Python 3.12 or newer (tested on 3.14).

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/generate_data.py        # creates data/generated_transactions.csv
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Running the app

```
streamlit run app.py
```

This opens http://localhost:8501 in your browser and has two pages:

- **Budget Planner:** enter 6 months of spending by hand, upload a transactions CSV, or pick a person from the dataset. The model predicts next month's spending per category. You then set your income, a savings goal (and optionally a total target, like $6,000 for a car), a cut limit, and any categories you don't want cut, and the app builds a recommended budget.
- **Train & Evaluate:** change the model settings (history length, layer size, layers, dropout, learning rate, epochs, loss), train with a live validation-error chart, compare against the baselines, and optionally save the result as the model the Budget Planner uses.

## Project layout

| Path | Purpose |
|------|---------|
| `src/generate_data.py` | Creates the large synthetic training set |
| `src/data.py` | Transactions → monthly table per category → training windows; split by person |
| `src/baselines.py` | Non-learning predictors (last month, 3-month average, full-history average) |
| `src/model.py` | The PyTorch model (MLP that predicts a correction to the history average) |
| `src/train.py` | Training loop with early stopping; saves `models/spending_model.pt` |
| `src/evaluate.py` | Metrics (MAE, Total MAE, WAPE); scores baselines and the saved model |
| `models/spending_model.pt` | Trained model, committed so the app works without retraining |
| `src/predict.py` | Forecast next month for one person (used by the app) |
| `src/budget.py` | Rule-based budget algorithm: predictions + income + goal → recommended budget |
| `app.py` | Streamlit app |

## How the budget is built

The model predicts what you **will** spend; `src/budget.py` decides what you **should** spend:

1. Fixed costs (Housing, Utilities, Bills, Family) stay at their predicted amount.
2. If income minus predicted spending covers the savings goal, the budget is the prediction, and any extra is reported as leftover.
3. Otherwise, flexible categories are cut. Each can be cut by at most *cut limit × flexibility*: Dining, Entertainment and Shopping 100%, Other 60%, Transportation 40%, Groceries 30%, Healthcare 0%. The cuts are shared in proportion to how much each category can give.
4. If the maximum cuts still aren't enough, the app reports the savings that are achievable and suggests options.

## Training and evaluating

```
python -m src.train                                   # ~30 s on a laptop CPU
python -m src.train --hidden 256 --dropout 0.2 --lr 0.0005   # any TrainConfig field can be changed
python -m src.evaluate                                # compare saved model vs. baselines
```

**How the model works:** it looks at the last 6 months of spending per category, plus income and the calendar month of the month being predicted. Every input is divided by the person's average monthly spending, so it learns proportions instead of raw dollars. The output is *history average + learned correction*, and the correction layer starts at zero, so an untrained model behaves exactly like the average baseline. It is trained with L1 loss (mean absolute error) using the Adam optimizer, and training stops early once validation error stops improving.

**Current results** (MAE = average dollars off per category per month; lower is better):

| Method | Test (75 unseen simulated people) | Original dataset (10 people) |
|---|---|---|
| Last month | $89.60 | $72.96 |
| 3-month average | $80.08 | $69.31 |
| Full-history average | $74.79 | $62.23 |
| Full-history median | $68.44 | $56.74 |
| **Neural network** | **$57.93** | **$51.46** |

## Budget categories

Raw merchant categories are grouped into 11 budget categories (see `CATEGORY_GROUPS` in `src/data.py`):
Housing, Utilities, Bills, Groceries, Dining, Transportation, Shopping, Entertainment, Family, Healthcare, Other.
Refunds are subtracted from their category, and savings transfers are tracked separately (they are not spending).

## Data

| File | What it is |
|------|------------|
| `data/synthetic_transactions.csv` | Original dataset: 10 people, 12 months (2025). Used as the **final test set**. |
| `data/generated_transactions.csv` | Large training set made by `src/generate_data.py` (not committed; regenerate it). |
| `data/generated_people.csv` | One row per generated person: archetype, income, life events. Useful for analysis. |

Both transaction files share the same columns:
`sample_id, transaction_type, amount, merchant_category, balance_after, transaction_date` (dates are `MM-DD-YYYY`, money out is negative).

### Generating the training data

The generator uses only the Python standard library, and the same seed always produces the same file:

```
python src/generate_data.py                                  # 500 people x 24 months (~485k rows, ~3 s)
python src/generate_data.py --people 1000 --months 36 --seed 7
```

The simulated people are built with patterns a model can learn:

- **Archetypes:** student, single, couple, family, retiree and freelancer, each with its own income range and spending habits.
- **Seasonality:** utilities rise in winter and summer, shopping and gifts in November and December, travel in summer.
- **Income effects:** higher income means more spending. Job loss cuts discretionary spending, and a raise slowly increases it.
- **Fixed vs. variable costs:** rent, insurance, subscriptions and debt payments are constant; food, shopping and similar categories vary.
- **Trends and feedback:** spending drifts slowly upward over time, and people cut back when their balance runs low.
- **Irregular spikes:** travel, taxes and education are hard to predict, just as in real life.
