# ALDER — Multi-Agent Quantitative Research & Paper-Trading Terminal

ALDER is a Streamlit terminal for systematic equity research. A four-stage pipeline — researcher,
risk manager, strategist, portfolio manager — runs three locally trained ML models over live market
data, an independent **walk-forward backtest** asks whether the resulting signal survives transaction
costs, and orders execute into a **simulated** portfolio that keeps its own trade log and daily
valuation history.

It is a research and engineering showcase, not a trading system: fills happen at the latest quoted
price with no slippage beyond a flat bps charge, and the models are documented below with the naive
baselines they are measured against.

![ALDER dashboard](docs/dashboard.png)

---

## Features

- **7 views** — Dashboard, Equity Research, Risk, Strategy Signals, Portfolio, Trade Desk,
  Strategy Lab — in a black, white and gold palette.
- **Universal ticker search** — every listed common stock on NASDAQ, NYSE and AMEX (**7,457 symbols**,
  refreshed weekly from the NASDAQ Trader symbol directory) searchable instantly by ticker *or*
  company name, offline, with no rate limit.
- **Company identities that always render** — real logos from a CDN with a deterministic monogram
  fallback painted underneath, so a missing or blocked image never shows a broken-image icon.
- **Four agents** with typed `{metric, value, status, detail}` contracts, deterministic and auditable.
- **Three locally trained models** on a five-session horizon — a RandomForest + HistGradientBoosting
  blend for expected return, HistGradientBoosting for volatility, and a class-weighted
  HistGradientBoosting classifier for direction. No LLM in the decision path, no model APIs, no
  shuffling anywhere: every score is produced by expanding-window walk-forward refits.
- **Strategy Lab** — expanding-window walk-forward backtest with per-change cost modelling; equity
  curve, Sharpe, Sortino, max drawdown, hit rate, exposure and trade count against buy & hold.
- **Persistent portfolios with a history** — accounts, cash and holdings live in PostgreSQL or SQLite
  behind one API, alongside a trade log (timestamp, side, shares, price, notional) and a daily
  valuation snapshot that drives the portfolio-value chart.
- **Design prototype included** — `alder-ai.html`, a self-contained 9-screen dashboard mockup.

---

## Measured results

Produced by `python model_eval.py`: two years of daily bars, **expanding-window walk-forward**
(each refit sees past bars only, no shuffling), scored on 38 out-of-sample five-session windows.

| Ticker | Return MAE | Return R² | Price MAE (5 sessions) | Volatility R² | Direction (up/down) | Always-up baseline |
|---|---:|---:|---:|---:|---:|---:|
| NVDA | 455 bps | +0.029 | $9.25 | −0.39 | **63.2%** | 39.5% |
| AAPL | 286 bps | −0.009 | $8.42 | +0.24 | **63.2%** | 52.6% |
| MSFT | 400 bps | −0.004 | $16.74 | −0.12 | 39.5% | 47.4% |

Mean directional accuracy **55.3%** against an always-up baseline of **46.5%** (+8.8 pp). The honest
reading: the five-session signal carries a real but modest edge on two of the three names and loses
to the baseline on MSFT. Return R² is essentially zero, which is normal — you cannot forecast a
week of equity return from price features with meaningful R² — so the expected return is shrunk
toward zero (`SHRINK = 0.20`) and clipped to ±1.25σ before it becomes a price target, which is what
keeps the target inside a plausible weekly band instead of extrapolating a rally.

Every figure is measured against a naive rule of thumb, because on a rising market a coin flip
already clears 50%: a 52% "accuracy" headline from an earlier version of this project came from a
single static 80/20 split on a one-day horizon and is not comparable to these numbers.

The Strategy Lab is the economic test. Long when the strategist calls BUY, held five sessions,
5 bps charged on every position change (2 years):

| Ticker | Strategy | Buy & hold | Excess | Sharpe | Max drawdown | Hit rate | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVDA | **+51.3%** | +43.5% | **+7.8 pp** | 1.32 | −20.5% | 61% | 19 |
| AAPL | +13.8% | +36.0% | −22.3 pp | 0.65 | −13.4% | 57% | 17 |
| MSFT | −5.8% | +18.4% | −24.2 pp | −0.17 | −22.7% | 49% | 23 |

The signal beats buy & hold on NVDA with a similar drawdown, and lags badly on AAPL and MSFT where
the book sat in cash through most of the uptrend (16–21% time in market). The app says this on the
page rather than hiding it.

---

## How the pipeline works

```
        ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
ticker →│ Researcher │──▶│    Risk    │──▶│ Strategist │──▶│  Manager   │→ allocation
        │ RF+HGB ret │   │  HGB vol   │   │  HGB dir   │   │ risk-scaled│
        └────────────┘   └────────────┘   └────────────┘   └────────────┘
```

| Stage | Implementation | Output |
|---|---|---|
| **Researcher** | RF(200, depth 6) blended 50/50 with HGB on 16 engineered features | Expected 5-session return → price target |
| **Risk Manager** | HGB regressor on the same features | Expected 5-session realised volatility |
| **Strategist** | Class-weighted HGB classifier, ±1% deadband | BUY / HOLD / SELL, gated by the researcher's return and the risk score |
| **Portfolio Manager** | `min(0.5, 0.20 − risk × 10)`, 0% on SELL | Capital weight |

---

## Data sources and why these

| Purpose | Source | Notes |
|---|---|---|
| Quotes & daily bars | Yahoo Finance via `yfinance` | Works without keys; the raw Yahoo REST endpoints rate-limit (HTTP 429), so requests are cached and retried rather than called per render |
| Ticker universe | NASDAQ Trader symbol directory (`nasdaqtraded.txt`) | 7,457 tradeable common stocks, cached to disk for a week |
| Company logos | `assets.parqet.com` symbol logos | 100×100 PNG per ticker, with a monogram fallback |

Every network call in `market_data.py` has a timeout, bounded retries with exponential
backoff, an in-process TTL cache and, for reference data, a disk cache — so a UI rerun never re-hits
an API, and a dead source degrades to the other one instead of breaking the page.

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

Create an account — you start with **$100,000** of simulated cash. A demo account (`amogh` / `abc`) is
seeded on first run; change or remove it before exposing the app anywhere.

### Optional: PostgreSQL

SQLite needs no configuration. For PostgreSQL (e.g. Supabase) supply a connection string — no code
changes:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/dbname"
```

```toml
# .streamlit/secrets.toml  (git-ignored)
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

---

## Repository map

| File | LOC | Role |
|---|---:|---|
| `app.py` | 770 | Views, auth, pipeline orchestration, trade execution |
| `ui.py` | 395 | Design system: theme CSS, navigation, KPI tiles, avatars, charts |
| `market_data.py` | 356 | Ticker universe, quotes, history, retries, caching, logos |
| `analytics.py` | 296 | Walk-forward backtest, risk metrics, cached model scoring |
| `database.py` | 392 | Dual backend (PostgreSQL / SQLite), users, portfolios, trades, valuation history |
| `models.py` | 460 | `AlderMLModels`: feature engineering, 3 models, walk-forward evaluator |
| `agents.py` | 67 | The four pipeline agents |
| `model_eval.py` | 66 | CLI evaluation harness with naive baselines |
| `crew.py`, `tools.py` | 111 | **Experimental, not wired in** — earlier CrewAI sketch; `crewai` is not a dependency |
| `alder-ai.html` | 1,257 | Static 9-screen UI prototype |

~2,700 lines of Python in the running application.

---

## Design decisions

- **Full-universe local search** rather than a "type a ticker" box: the symbol directory is downloaded
  once and cached, so name search over 7,400 symbols costs no API call and cannot rate-limit.
- **Chronological train/test split** (`shuffle=False`) everywhere — random shuffling leaks future
  prices into training and inflates every score.
- **Walk-forward backtest** with expanding-window refits and per-change costs, because a model that
  scores well on a static hold-out can still lose money once it trades.
- **Monogram fallback under every logo** — CDN logos fail silently; an `<img onerror>` swap means the
  UI never shows a broken image.
- **Honest failure states**: a dead symbol, a blocked data source or a database error surfaces as a
  readable message, not a traceback or a silent `None`.
- **Absolute database path** (overridable via `ALDER_DB_FILE`) so launching from another directory
  cannot silently create a second, empty database.

---

## Limitations

- Orders simulate at the last quoted price with a flat bps assumption: **no order-book impact, borrow
  cost or partial fills**, and no live broker integration.
- The directional edge is modest and inconsistent: 55.3% mean walk-forward accuracy against a 46.5%
  always-up baseline, but negative on MSFT, and the backtest only beats buy & hold on NVDA of the
  three sampled names. Treat signals as research output, not tradeable alpha.
- The volatility model is only better than a naive estimate on some names (R² +0.23 on AAPL,
  negative on NVDA and MSFT), so the risk score is a guide rather than a calibrated forecast.
- Models are refit per pipeline run (no persistence layer for models yet), so each analysis costs a
  few seconds and depends on Yahoo Finance availability.
- Authentication is demo-grade: SHA-256 without per-user salt, no session expiry, no rate limiting.
- No automated test suite yet; flows were verified manually with Streamlit's `AppTest` harness.

---

## Roadmap

- Persist trained models per (ticker, day) and add a proper walk-forward parameter sweep.
- Replace the hashing scheme with a salted KDF and add tests around persistence and the data layer.
