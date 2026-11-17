# ALDER: a stock research app that shows its work

ALDER is a web app that studies a stock and guesses what it will do next. It looks at two years of
daily prices, trains three small machine-learning models on them, and turns the answers into a price
target, a risk number, and a BUY / HOLD / SELL signal. You can then "buy" and "sell" with fake money
and watch your portfolio change over time.

Everything runs on your own computer. There is no paid API and no GPT-style model making decisions.
It is a learning and portfolio project, **not** a way to get rich: the trades are simulated and the
README tells you exactly how well (and how badly) the models do.

![ALDER dashboard](docs/dashboard.png)

---

## What it can do

- **Seven pages**: Dashboard, Research, Risk, Strategy, Portfolio, Trade Desk, Strategy Lab.
- **Search 7,457 stocks** by ticker or company name. The list comes from NASDAQ and is saved to disk,
  so searching is instant and never hits a rate limit.
- **Three models** trained on your machine. Nothing is pre-baked and no AI service is called.
- **A backtest** that replays the signal on old data, charges a fee per trade, and compares the
  result to simply buying and holding the stock.
- **A real database** for your account, cash, holdings, every trade you make, and one portfolio value
  per day (which draws the chart on the Portfolio page).
- **A login page** and a sign-out button, because otherwise anyone could spend your fake money.

---

## How the four steps work

Think of it as four people passing a note to each other.

```
  pick a        Researcher        Risk           Strategist        Manager
  stock   →   "what will it  →  "how bumpy   →  "so: BUY,      →  "how much
              do in 5 days?"    will the ride   HOLD or SELL?"     money?"
                                 be?"
```

| Step | What it actually does | What it hands on |
|---|---|---|
| **Researcher** | Two models (a random forest and a gradient-boosted model) guess the next 5 days of return; their answers are averaged | A 5-day price target |
| **Risk manager** | A gradient-boosted model guesses how jumpy the next 5 days will be | A volatility number |
| **Strategist** | A classifier picks BUY / HOLD / SELL, but only says BUY if the Researcher also expects a rise and the Risk manager says it is not too bumpy | The signal |
| **Manager** | Turns the signal into a percentage of your cash, smaller when risk is high | How much to buy |

The point of the last two rows is that the steps actually depend on each other. A signal is not
produced in isolation.

---

## Does it work? (honest answer)

The models were tested the only fair way: train on the past, predict the future, then move forward
one week and do it again. They never see the answer before guessing. 38 guesses per stock.

| Stock | How far off the 5-day guess was | Right about up vs down | "Always guess up" would get |
|---|---:|---:|---:|
| NVDA | about 4.6% of the price | **63.2%** | 39.5% |
| AAPL | about 2.9% | **63.2%** | 52.6% |
| MSFT | about 4.0% | 39.5% | 47.4% |

Average: **55.3% right**, versus **46.5%** for the lazy "always guess up" rule. That is a real but
small edge. On Microsoft the model is worse than the lazy rule. We print that instead of hiding it.

Two things worth knowing:

- **Guessing the exact return is nearly impossible.** The models explain almost none of the week-to-week
  movement. That is normal for stock prices. So the prediction is deliberately shrunk toward zero and
  capped, otherwise a good week makes the app promise a silly +18% target.
- **A 52% score in an old version of this project was not real.** It came from one lucky train/test
  split on one-day guesses. The numbers above come from many forward steps, and they are compared
  against what a dumb rule would score on the same days.

### Would it make money?

The Strategy Lab buys when the signal says BUY, holds for five days, and pays a small fee on every
trade. Over two years:

| Stock | ALDER | Buy and hold | Sharpe | Worst drop |
|---|---:|---:|---:|---:|
| NVDA | **+51.3%** | +43.5% | 1.32 | −20.5% |
| AAPL | +13.8% | +36.0% | 0.65 | −13.4% |
| MSFT | −5.8% | +18.4% | −0.17 | −22.7% |

It beat buy-and-hold on NVDA and lost badly on the other two, mostly because it sat in cash during
uptrends. "Sharpe" is a score for how much return you got per unit of wobble: higher is better, and
above 1 is good. "Worst drop" is the biggest fall from a peak, which is the number that makes people
sell at the bottom.

---

## Where the data comes from

| What | Where | Why |
|---|---|---|
| Prices and history | Yahoo Finance (via `yfinance`) | Free, no key needed |
| List of stocks | NASDAQ's public symbol file | 7,457 stocks, saved for a week |
| Company logos | parqet image CDN | If an image fails, a letter badge is drawn instead |

Every internet call has a time limit, retries three times with growing waits, and saves its answer
in memory (quotes 45 seconds, history 15 minutes) and on disk (stock list one week). That way the
page does not freeze, and if Yahoo is down you see a readable message instead of an error dump.

---

## Run it yourself

You need Python 3.11 or newer.

```bash
git clone https://github.com/amoghb14/alder.git
cd alder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the link it prints. Make an account and you start with **$100,000** of fake cash. There is also
a demo login (`amogh` / `abc`) created on first run, so delete or change it before putting this on
the internet.

### Saving data across restarts (optional)

By default everything is stored in a small SQLite file next to the code. That is fine locally, but
free hosting wipes the disk when the app restarts. To keep accounts and trades, use a free
PostgreSQL database and give the app its address:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/dbname"
```

```toml
# .streamlit/secrets.toml  (never commit this file)
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

Nothing else changes. The same code talks to both databases.

---

## What is in each file

| File | Lines | Job |
|---|---:|---|
| `app.py` | 781 | The pages, the login, and the trade buttons |
| `models.py` | 463 | Builds the features, trains the three models, scores them |
| `ui.py` | 421 | Colours, cards, charts, sidebar |
| `database.py` | 401 | Users, portfolios, trades, and the value history |
| `market_data.py` | 356 | Downloads prices and the stock list, with retries and caching |
| `analytics.py` | 240 | The backtest and the portfolio maths |
| `agents.py` | 67 | Glue that lets each step talk to the next |
| `model_eval.py` | 58 | Command-line tool that prints the results table above |
| `alder-ai.html` | 1,257 | An early design mock-up, not used by the app |
| `crew.py`, `tools.py` | 111 | Leftover experiment, not used by the app |

About 2,900 lines of Python run the app.

---

## What it does not do

- **It does not trade real money.** Orders fill at the last price. No broker, no order book, no
  partial fills, no borrowing costs.
- **The edge is small and uneven.** 55.3% against 46.5% is not a money machine, and one of the three
  stocks came out worse than the lazy rule.
- **The risk number is rough.** It beats a simple estimate on Apple but not on NVDA or Microsoft.
- **Models retrain every time you press the button**, so answers take a few seconds and need an
  internet connection to Yahoo.
- **The login is a demo.** Passwords are hashed with SHA-256 and no salt, and sessions never expire.
  Do not reuse a real password.
- **There are no automated tests yet.** I checked the pages with Streamlit's `AppTest` tool and by
  clicking through the app.

---

## What I would do next

- Save trained models to disk so pages load faster.
- Add tests for the database code and the model scoring.
- Replace the password hashing with a proper salted scheme.
