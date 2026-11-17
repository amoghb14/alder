# ALDER — Multi-Agent Quantitative Research & Paper-Trading Terminal

ALDER is a Streamlit terminal for systematic equity research. A four-stage pipeline — researcher,
risk manager, strategist, portfolio manager — runs three locally trained ML models over live market
data, an independent **walk-forward backtest** asks whether the resulting signal survives transaction
costs, and orders execute into a **simulated** portfolio. A second desk (**Talent Radar**) mines live
software-engineering internship postings to rank the skills the market actually demands.

It is a research and engineering showcase, not a trading system: fills happen at the latest quoted
price with no slippage beyond a flat bps charge, and the directional model is documented below as
having **no measurable edge**.

![ALDER dashboard](docs/alder-dashboard.png)

---

## Features

- **8 views** — Dashboard, Equity Research, Risk, Strategy Signals, Portfolio, Trade Desk,
  Strategy Lab, Talent Radar.
- **Universal ticker search** — every listed common stock on NASDAQ, NYSE and AMEX (**7,457 symbols**,
  refreshed weekly from the NASDAQ Trader symbol directory) searchable instantly by ticker *or*
  company name, offline, with no rate limit.
- **Company identities that always render** — real logos from a CDN with a deterministic monogram
  fallback painted underneath, so a missing or blocked image never shows a broken-image icon.
- **Four agents** with typed `{metric, value, status, detail}` contracts, deterministic and auditable.
- **Three locally trained models** (`RandomForestRegressor`, `LinearRegression`,
  `DecisionTreeClassifier`) — no LLM in the decision path, no model APIs.
- **Strategy Lab** — expanding-window walk-forward backtest with per-change cost modelling; equity
  curve, Sharpe, Sortino, max drawdown, hit rate, exposure and trade count against buy & hold.
- **Talent Radar** — live internship postings normalised from two keyless job APIs, filtered to
  software roles, mined against a 46-term skill taxonomy, and scored against your own skill profile.
- **Persistent portfolios** — accounts, cash, holdings and AUM in PostgreSQL or SQLite behind one API.
- **Design prototype included** — `alder-ai.html`, a self-contained 9-screen dashboard mockup.

---

## Measured results

Produced by `python model_eval.py` — NVDA, 2 years of daily bars (501 rows), **chronological** 80/20
split (~100 held-out trading days), `random_state=42`:

| Model | Task | Metric | Result |
|---|---|---|---|
| `RandomForestRegressor` (100 trees) | Next-day close | MAE | **$5.62** (~2.6% of price level) |
| | | RMSE | **$7.10** |
| `LinearRegression` | Expected \|return\| | R² | **0.19** |
| `DecisionTreeClassifier` (depth 3) | BUY/HOLD/SELL | Accuracy | **51.6%** (F1 44.8%) |

The price model tracks the level well; the **directional classifier is a coin flip**, which is why the
app presents signals as research output and keeps execution simulated. An earlier README claimed
"~96% directional accuracy" — that figure was `100 − MAE/mean price`, a price-level error proxy, and
has been removed.

The Strategy Lab gives the honest test instead. Example (MSFT, 2 years, 5 bps per position change,
threshold 0.55): strategy **−5.0%** vs buy & hold **+17.4%**, Sharpe −0.15, max drawdown −18.4%,
94 trades, 48% hit rate. The signal does not beat the benchmark after costs — and the UI says so
rather than hiding it.

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

---

## Data sources and why these

| Purpose | Source | Notes |
|---|---|---|
| Quotes & daily bars | Yahoo Finance via `yfinance` | Works without keys; the raw Yahoo REST endpoints rate-limit (HTTP 429), so requests are cached and retried rather than called per render |
| Ticker universe | NASDAQ Trader symbol directory (`nasdaqtraded.txt`) | 7,457 tradeable common stocks, cached to disk for a week |
| Company logos | `assets.parqet.com` symbol logos | 100×100 PNG per ticker, with a monogram fallback |
| Internship postings | Arbeitnow + Remotive public APIs | Keyless; The Muse's public API is Cloudflare-gated (HTTP 403) and was rejected after testing |

Every network call in `market_data.py` and `talent.py` has a timeout, bounded retries with exponential
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
| `app.py` | 817 | Views, auth, pipeline orchestration, trade execution |
| `talent.py` | 394 | Job-board clients, normalisation, skill taxonomy, gap analysis |
| `ui.py` | 373 | Design system: theme CSS, navigation, KPI tiles, avatars, charts |
| `market_data.py` | 356 | Ticker universe, quotes, history, retries, caching, logos |
| `analytics.py` | 264 | Walk-forward backtest, risk metrics, model scoring |
| `database.py` | 288 | Dual backend (PostgreSQL / SQLite), users, portfolios |
| `models.py` | 162 | `AlderMLModels`: feature engineering, 3 models, report generators |
| `agents.py` | 67 | The four pipeline agents |
| `model_eval.py` | 75 | CLI evaluation harness |
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
- Directional classifier sits at ~52% accuracy on held-out data, and the backtest underperforms
  buy & hold on the sampled names — treat signals as research output, not tradeable alpha.
- Models are refit per pipeline run (no persistence layer for models yet), so each analysis costs a
  few seconds and depends on Yahoo Finance availability.
- Talent Radar covers keyless job boards, which skew toward EU and remote listings; it is not a
  crawl of every company's careers page. Skill extraction is a regex taxonomy, not an LLM.
- Authentication is demo-grade: SHA-256 without per-user salt, no session expiry, no rate limiting.
- No automated test suite yet; flows were verified manually with Streamlit's `AppTest` harness.

---

## Roadmap

- Persist trained models per (ticker, day) and add a proper walk-forward parameter sweep.
- Cache and version the talent scan with historical snapshots to show skill-demand trends over time.
- Replace the hashing scheme with a salted KDF and add tests around persistence and the data layer.
