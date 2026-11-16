# ALDER — Multi-Agent Quantitative Research & Paper-Trading Terminal

ALDER is a Streamlit application that runs a four-stage quantitative research pipeline over live
market data and lets you act on its output with a **simulated** portfolio. Three small ML models
are trained locally — no model APIs, no prompt wrappers — and each stage hands a typed result to
the next one: price prediction → volatility estimate → BUY/HOLD/SELL signal → capital weight.

It is a research and engineering showcase, not a live trading system: orders fill at the latest
quoted price with no slippage, commission or borrow modelling, and the directional model is
documented below as having **no measurable edge**.

---

## Measured results

Produced by `python model_eval.py` — NVDA, 2 years of daily bars (501 rows), **chronological**
80/20 split (~100 held-out trading days), `random_state=42`.

| Model | Task | Metric | Result |
|---|---|---|---|
| `RandomForestRegressor` (100 trees) | Next-day close | MAE | **$5.62** (~2.6% of price level) |
| | | RMSE | **$7.10** |
| `LinearRegression` | Expected \|return\| | R² | **0.19** |
| `DecisionTreeClassifier` (depth 3) | BUY/HOLD/SELL | Accuracy | **51.6%** (F1 44.8%) |

**Read those numbers honestly.** The price model tracks the level well, but the *directional*
classifier is a coin flip at a daily horizon — so the app presents signals as a research artefact
and keeps execution simulated. Earlier revisions of this README claimed "~96% directional
accuracy"; that figure was `100 − MAE/mean price`, a price-level error proxy, and has been removed.

Re-run the harness any time — the numbers move with the data:

```bash
python model_eval.py
```

---

## How the pipeline works

```
        ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
ticker →│ Researcher │──▶│    Risk    │──▶│ Strategist │──▶│  Manager   │→ allocation
        │  RF price  │   │  LR vol    │   │  DT signal │   │ risk-scaled│
        └────────────┘   └────────────┘   └────────────┘   └────────────┘
```

| Stage | Implementation | Output |
|---|---|---|
| **Researcher** | `RandomForestRegressor` on Close/Vol/MA5/MA20 | Predicted next-day close |
| **Risk Manager** | `LinearRegression` on Vol/Returns → abs(return) | Expected volatility score |
| **Strategist** | `DecisionTreeClassifier` on Returns/Vol | BUY / HOLD / SELL |
| **Portfolio Manager** | `min(0.5, 0.20 − risk × 10)`, 0% on SELL | Capital weight |

Each agent (`agents.py`) returns a uniform `{metric, value, status, detail}` dict, so stages can be
swapped or tested in isolation.

---

## Features

- **6 analytical views** — Dashboard (pipeline run + candlestick chart), Research report, Portfolio
  risk audit, Cross-asset growth ranking, Portfolio narrative, Trade desk.
- **20-ticker preset universe** (mega-cap names) plus free-form ticker entry for any Yahoo Finance symbol.
- **Per-user portfolios** — accounts, cash balance, holdings and AUM valuation persisted server-side.
- **Simulated order execution** with balance and position checks; orders fill at the latest quote.
- **Dual database backend** — PostgreSQL when `DATABASE_URL` is set, local SQLite otherwise, behind
  a single dialect-aware API. Zero configuration to run locally.
- **Design prototype included** — `alder-ai.html`, a self-contained 1,257-line, 9-screen dashboard
  mockup (6 Chart.js visualisations) built to define the product surface before implementation.

---

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/amoghb14/alder.git
cd alder
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the app and create an account — you start with **$100,000** of simulated cash. The first run
also seeds a demo account (`amogh` / `abc`); change or remove it before exposing the app anywhere.

### Optional: PostgreSQL

The app runs on SQLite with no configuration. To use PostgreSQL (e.g. Supabase), supply a
connection string via environment variable or Streamlit secrets — no code changes needed:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/dbname"
```

```toml
# .streamlit/secrets.toml  (git-ignored)
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

---

## Tech stack

Python 3.11 · Streamlit · scikit-learn · Pandas / NumPy · Plotly · yfinance (2y daily bars) ·
PostgreSQL (psycopg2) or SQLite · Chart.js for the static prototype.

---

## Repository map

| File | LOC | Role |
|---|---:|---|
| `app.py` | 458 | All UI: theming, auth screens, 6 views, pipeline orchestration |
| `database.py` | 287 | Dual backend (PostgreSQL / SQLite), users + portfolios, password hashing |
| `models.py` | 162 | `AlderMLModels`: feature engineering, 3 models, report + risk-audit generators |
| `model_eval.py` | 75 | Standalone evaluation harness (MAE/RMSE/R²/accuracy/F1) |
| `agents.py` | 67 | The four pipeline agents |
| `crew.py`, `tools.py` | 111 | **Experimental, not wired in** — an earlier CrewAI/LLM agent sketch kept for reference; `crewai` is not a dependency |
| `alder-ai.html` | 1,257 | Static 9-screen UI prototype |

~1,160 lines of Python in the running application.

---

## Design decisions

- **Chronological train/test split** (`shuffle=False`) for every model — random shuffling would leak
  future prices into training and inflate scores.
- **Deterministic pipeline** — fixed `random_state=42`, sequential handoffs, no hidden agent chatter,
  so a result can be reproduced and audited.
- **Absolute database path** (overridable via `ALDER_DB_FILE`) so `streamlit run` from a different
  working directory can't silently create a second, empty database.
- **Signals are not collapsed** — the real BUY/HOLD/SELL value reaches the allocation stage, so a
  SELL allocates 0% instead of defaulting to a HOLD weight.
- **Failures are loud** — database problems raise `DatabaseError` with the cause and are surfaced in
  the UI, rather than leaving a `None` connection that breaks every page afterwards.

---

## Limitations

- Orders are simulated at the last quoted price: **no slippage, commission, borrow cost or volume
  impact**, and no P&L backtest of the strategy itself.
- Directional classifier sits at ~52% accuracy on held-out data — treat signals as research output,
  not tradeable alpha.
- Models are refit on every pipeline run (no caching), so a run takes seconds and is subject to
  Yahoo Finance rate limits; the data is not tick-level or guaranteed real-time.
- Authentication is demo-grade: SHA-256 without per-user salt, no session expiry, no rate limiting.
  Put it behind a real identity provider before any multi-user deployment.
- No automated test suite yet; the database layer and UI flow were verified manually with Streamlit's
  `AppTest` harness.

---

## Roadmap

- Backtest the signal with slippage and commission — publish the equity curve whether or not it works.
- Cache trained models per (ticker, day) to cut per-request fitting.
- Replace the hashing scheme with a salted KDF and add automated tests around the persistence layer.
