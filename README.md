# FinHelp
A simple artificial intelligence program to help predict and budget future expenses.

FinHelp has two parts that work together:

- **`api/` + `src/` (Python):** the machine learning. This covers data processing, the PyTorch model, training, evaluation and the budget algorithm, exposed as a small **FastAPI** web server.
- **`web/` (SvelteKit):** the website you use. It runs in the browser and gets every prediction, budget and accuracy number from the Python API.

```
Browser ──► SvelteKit website (web/) ──/api/...──► FastAPI server (api/main.py) ──► src/ (PyTorch model, budget, evaluation)
```

## Setup (one time)

Requires **Python 3.12+** (tested on 3.14) and **Node.js 20+** (tested on 24). Run these from the project folder in PowerShell:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/generate_data.py            # creates data/generated_transactions.csv (~3 s)
npm --prefix web install               # installs the website's packages
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.
If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Running the app

### Option A: one command (recommended)

```
.\start.ps1
```

This opens the Python API in its own window (http://localhost:8000) and starts the website at **http://localhost:5173**, opening it in your browser. Close both windows (or press Ctrl+C) to stop.
If PowerShell blocks the script, use `powershell -ExecutionPolicy Bypass -File start.ps1`.

### Option B: two terminals (any OS)

```
# Terminal 1: the Python API (with the .venv activated)
python -m uvicorn api.main:app --port 8000

# Terminal 2: the website
npm --prefix web run dev
```

Then open **http://localhost:5173**. The website forwards every `/api/...` request to the Python server.

### Option C: final build for a demo (a single server)

```
npm --prefix web run build                 # builds the website into web/build
python -m uvicorn api.main:app --port 8000
```

Then open **http://localhost:8000**. The Python server now serves the website as well, so nothing else needs to run. Rebuild after changing anything in `web/`.

**Tips**
- Interactive documentation for every API endpoint is at http://localhost:8000/docs.
- Changed Python code in `api/` or `src/`? Restart the API. New or edited custom models in `src/custom_models/` are picked up automatically.
- Changed code in `web/`? The website updates instantly in Options A and B.

## Using the app

The app has four pages, listed in the bar across the top:

| Page | What you can do |
|---|---|
| **Start Here** | A short guide to what the app does, a quick start, why it's useful, its limits, and key terms. |
| **Budget Planner** | Load a person's spending history (an editable example, a CSV upload, or someone from the datasets), see next month's prediction, optionally set a savings goal, get a recommended budget, and check **how accurate the model has been for that person**. You can choose which model makes the prediction. |
| **Compare Models** | Score the main neural network, any custom models and the baselines side by side: overall accuracy, accuracy per category, and month-by-month accuracy for one person. |
| **Train Model** | Change the neural network's settings, retrain it with a live progress chart, compare it with the current model, and optionally save it. |

### Quick guide

1. Open **Budget Planner**. An example person ($3,000 income, about $2,300 spending) is already loaded. Open the collapsible table to see or edit their months.
2. **Next month's prediction** shows what the model expects them to spend in total and in each category, compared with their recent average.
3. **Income and savings:** check the expected income, then choose **No savings goal**, **Save a % of income** (a slider, 20% by default), or **Save for something** (what it's for, how much per month, and optionally the total needed). *Budget options* sets the cut limit and categories not to cut.
4. **Recommended budget** shows summary cards for **Rent / housing**, **Other consistent bills**, **All bills**, **Everyday needs** and **Wants**, each against its maximum. Below them, every category lists its **predicted**, **max logical** and **recommended** spend, how much gets saved, and whether the target is met.
5. **How accurate is the model for this person?** (at the bottom) replays their past months and reports the model's accuracy next to a simple 3-month average. Use this to decide how much to trust the prediction.
6. Choose *Pick someone from the dataset* to try other people, or open **Compare Models** to see overall results.

**Why it's useful:** budgets start from a realistic forecast of each person's own habits instead of a generic rule. The app shows exactly which categories must shrink to reach a goal, and it is upfront about how reliable each prediction is.
**Limits:** it is trained on simulated people, it can't predict rare one-off costs well (keep a buffer), and it needs at least 6 months of history (7+ months to check its own accuracy).

## How accuracy is measured

The same method is used everywhere: in the app, in `python -m src.evaluate`, and in the results below.

1. **Hide a month, then predict it** using only the months before it (backtesting / walk-forward validation).
2. For each category in each month, record the miss `|predicted − actual|`.
3. Summarize:
   - **Average miss per category (MAE):** the mean of all misses, in dollars.
   - **Error rate (WAPE):** sum of misses ÷ sum of actual spending.
   - **Accuracy = 100% − WAPE**.
4. For dataset-wide scores, this is done on **test people no model trained on**: 75 held-out simulated people, plus the original 10-person dataset. The split is fixed (`SPLIT_SEED` in `src/data.py`), so every model is scored on the same people.

## Comparing your own models

Any model can be added to the comparison, whatever the method (PyTorch, NumPy, scikit-learn…):

1. Copy `src/custom_models/_template.py` to a new file **without** the leading underscore, e.g. `src/custom_models/small_nn.py`.
2. Fill in `NAME`, `DESCRIPTION`, `HISTORY` (months it needs) and the `train()` and `load()` functions. `load()` must return a function that takes a `Windows` object and returns predicted dollars with shape `(N, 11)`.
3. Train only on `splits["train"]` (or a subset of it, to test "less data"), never on test or original people.
4. Train it with `python -m src.custom_models.small_nn`, or with the **Train** button on the Compare Models page.

It then appears automatically on **Compare Models**, in the Budget Planner's model picker, and in `python -m src.evaluate`.
`src/custom_models/linear_regression.py` is a complete working example: ridge linear regression trained on only 100 people.

## Project layout

| Path | Purpose |
|------|---------|
| `src/generate_data.py` | Creates the large synthetic training set |
| `src/data.py` | Transactions → monthly table per category → training windows; split by person |
| `src/baselines.py` | Non-learning predictors (last month, 3-month average, full-history average) |
| `src/model.py` | The PyTorch model (MLP that predicts a correction to the history average) |
| `src/train.py` | Training loop with early stopping; saves `models/spending_model.pt` |
| `src/evaluate.py` | Accuracy metrics, model comparison, per-person backtest: `python -m src.evaluate` |
| `src/zoo.py` | Registry of every model (main network, custom models, baselines) behind one interface |
| `src/custom_models/` | Extra models to compare: a `_template.py` and a working `linear_regression.py` example |
| `models/spending_model.pt` | Trained neural network, committed so the app works without retraining |
| `models/linear_regression.npz` | Trained example comparison model |
| `src/predict.py` | Forecast next month for one person (used by the app) |
| `src/budget.py` | Rule-based budget algorithm: predictions + income + goal → recommended budget |
| `api/main.py` | FastAPI server: exposes forecasting, budgets, accuracy checks, comparison and training to the website; serves the built website in Option C |
| `web/src/routes/` | The four pages: `+page.svelte` (Start Here), `planner/`, `compare/`, `train/`, and `+layout.svelte` (top bar) |
| `web/src/lib/` | Shared frontend code: `api.ts` (calls the Python API), `state.svelte.ts` (inputs kept between pages), `components/` (charts, metric cards, toggle…) |
| `web/vite.config.ts` | SvelteKit config, including the `/api` forwarding to the Python server |
| `start.ps1` | Starts the API and the website together (Option A) |

## How the budget is built

The model predicts what you **will** spend; `src/budget.py` decides what you **should** spend. Each category gets three numbers:

| Number | Meaning |
|---|---|
| **Predicted** | What the neural network expects next month |
| **Max logical spend** | A guideline share of income, based on the "30% rule" for housing and the 50/30/20 budget: Housing 30%, Utilities 8%, Bills 10%, Family 12%, Groceries 12%, Transportation 12%, Healthcare 6%, Dining 8%, Shopping 8%, Entertainment 5%, Other 8%. For flexible categories the max never drops below the person's typical month (median of the last 6 months) minus the largest allowed cut, so it fits their real life. |
| **Recommended** | What to budget (see below) |

Groups are summarized with their own maximums: **Rent / housing** ≤ 30% of income, **Other consistent bills** (utilities, insurance/subscriptions/debt, childcare) ≤ 20%, **All bills** ≤ 50%, **Everyday needs** (groceries, transportation, healthcare) ≤ 25%, **Wants** (dining, shopping, entertainment, other) ≤ 30%.

The savings target is $0 with no goal, *rate × income* when saving a percent of income, or the typed amount when saving for something specific. Then:

1. Fixed bills (Housing, Utilities, Bills, Family) stay at their predicted amount. Any above its max is flagged, since it can't be cut within a month.
2. Flexible categories above their max are trimmed down to it.
3. If income minus spending still doesn't cover the savings target, flexible categories are cut further. Each can be cut by at most *cut limit × flexibility*: Dining, Entertainment and Shopping 100%, Other 60%, Transportation 40%, Groceries 30%, Healthcare 0%. The cuts are shared in proportion to how much each category can give.
4. If the maximum cuts still aren't enough, the app reports the savings that are achievable and suggests options.

## Training and evaluating

```
python -m src.train                                   # ~30 s on a laptop CPU
python -m src.train --hidden 256 --dropout 0.2 --lr 0.0005   # any TrainConfig field can be changed
python -m src.evaluate                                # compare saved model vs. baselines
```

**How the model works:** it looks at the last 6 months of spending per category, plus income and the calendar month of the month being predicted. Every input is divided by the person's average monthly spending, so it learns proportions instead of raw dollars. The output is *history average + learned correction*, and the correction layer starts at zero, so an untrained model behaves exactly like the average baseline. It is trained with L1 loss (mean absolute error) using the Adam optimizer, and training stops early once validation error stops improving.

**Current results** (accuracy, with the average miss per category per month in parentheses):

| Method | Test (75 unseen simulated people) | Original dataset (10 people) |
|---|---|---|
| Last month | 73.9% ($89.60) | 78.2% ($72.96) |
| 3-month average | 76.7% ($80.08) | 79.3% ($69.31) |
| Full-history average | 78.2% ($74.79) | 81.4% ($62.23) |
| Full-history median | 80.1% ($68.44) | 83.0% ($56.74) |
| Linear regression (100 people) | 78.1% ($75.31) | 80.5% ($65.07) |
| **Neural network** | **83.1% ($57.93)** | **84.6% ($51.46)** |

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
