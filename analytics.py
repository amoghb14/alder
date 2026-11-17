"""Quantitative analytics: walk-forward backtest, risk metrics, model scoring.

This is the module that answers the question a research desk actually asks —
*does the signal make money after costs?* — instead of only displaying a
prediction. It reuses the same feature set and estimator family as the live
pipeline, but evaluates them the only honest way: an expanding-window walk-forward
with a transaction cost on every position change, benchmarked against buy & hold.

Nothing here touches the network except through ``market_data``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier

from market_data import history

TRADING_DAYS = 252
WARMUP = 60  # rows needed before the first out-of-sample decision
PROB_THRESHOLD = 0.55


# --------------------------------------------------------------------------- #
# Feature engineering (shared with the live pipeline)
# --------------------------------------------------------------------------- #
def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Vol"] = df["Close"].rolling(window=5).std()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["Momentum"] = df["Close"].pct_change(periods=10)
    df["Target"] = df["Close"].shift(-1)
    df["Up"] = (df["Target"] > df["Close"]).astype(int)
    return df


FEATURES = ["Returns", "Vol", "MA5", "MA20", "Momentum"]


def _annualise(daily: pd.Series) -> Dict[str, float]:
    daily = daily.dropna()
    if daily.empty:
        return {"total_return": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "max_drawdown": 0.0, "calmar": 0.0}
    equity = (1 + daily).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = len(daily) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else 0.0
    volatility = float(daily.std() * np.sqrt(TRADING_DAYS))
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 and downside.std() > 0 else 0.0
    peak = equity.cummax()
    drawdown = (equity / peak - 1)
    max_dd = float(drawdown.min())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else 0.0,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
def backtest(
    symbol: str,
    period: str = "2y",
    cost_bps: float = 5.0,
    refit_every: int = 21,
    threshold: float = PROB_THRESHOLD,
) -> Dict[str, Any]:
    """Out-of-sample walk-forward backtest of the directional signal.

    On every rebalance the classifier is refitted only on data available up to
    that day, so no future information reaches the position sizing. A position is
    either fully long or flat, and ``cost_bps`` is charged on every change.

    Returns ``{symbol, curve, metrics, strategy, benchmark, trades, ...}``.
    """
    raw = history(symbol, period=period)
    df = build_features(raw).dropna()
    if len(df) < WARMUP + 30:
        raise ValueError(f"Not enough history for {symbol} to backtest")

    dates = df.index
    positions = pd.Series(0.0, index=dates)
    probabilities = pd.Series(np.nan, index=dates)

    model = None
    for i in range(WARMUP, len(df) - 1):
        if model is None or (i - WARMUP) % refit_every == 0:
            train = df.iloc[:i]
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            model.fit(train[FEATURES], train["Up"])
        row = df[FEATURES].iloc[[i]]
        try:
            probability = float(model.predict_proba(row)[0][-1])
        except Exception:
            probability = 0.5
        probabilities.iloc[i] = probability
        positions.iloc[i] = 1.0 if probability >= threshold else 0.0

    next_returns = df["Returns"].shift(-1)
    gross = positions * next_returns
    traded = positions.diff().abs().fillna(positions.abs())
    costs = traded * (cost_bps / 10_000.0)
    strategy_returns = (gross - costs).dropna()
    benchmark_returns = next_returns.reindex(strategy_returns.index).fillna(0.0)

    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()

    curve = pd.DataFrame(
        {
            "Strategy": strategy_equity,
            "Buy & Hold": benchmark_equity,
        }
    )

    active_days = strategy_returns[positions.reindex(strategy_returns.index) > 0]
    hit_rate = float((active_days > 0).mean() * 100) if len(active_days) else 0.0
    metrics = {
        "strategy": _annualise(strategy_returns),
        "benchmark": _annualise(benchmark_returns),
        "hit_rate": hit_rate,
        "exposure": float((positions.reindex(strategy_returns.index) > 0).mean() * 100),
        "trades": int(traded.sum()),
        "cost_bps": cost_bps,
        "days": int(len(strategy_returns)),
        "threshold": threshold * 100,
    }
    metrics["excess_return"] = (
        metrics["strategy"]["total_return"] - metrics["benchmark"]["total_return"]
    ) * 100

    return {
        "symbol": symbol.upper(),
        "curve": curve,
        "metrics": metrics,
        "positions": positions,
        "probabilities": probabilities,
        "frame": df,
        "last_price": float(df["Close"].iloc[-1]),
    }


# --------------------------------------------------------------------------- #
# Portfolio analytics
# --------------------------------------------------------------------------- #
def holdings_frame(holdings: Dict[str, float], quote_lookup) -> pd.DataFrame:
    """Value a holdings dict. ``quote_lookup(symbol) -> dict`` supplies prices."""
    rows: List[Dict[str, Any]] = []
    for symbol, shares in (holdings or {}).items():
        try:
            price = float(quote_lookup(symbol)["price"])
        except Exception:
            price = 0.0
        rows.append({"Symbol": symbol, "Shares": shares, "Price": price, "Value": shares * price})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["Symbol", "Shares", "Price", "Value", "Weight"])
    total = frame["Value"].sum()
    frame["Weight"] = (frame["Value"] / total * 100).round(2) if total else 0.0
    return frame.sort_values("Value", ascending=False).reset_index(drop=True)


def concentration(frame: pd.DataFrame) -> Dict[str, float]:
    """Herfindahl-based concentration plus largest-position weight."""
    if frame is None or frame.empty or frame["Value"].sum() <= 0:
        return {"hhi": 0.0, "largest": 0.0, "positions": 0}
    weights = (frame["Value"] / frame["Value"].sum()).values
    return {
        "hhi": float((weights**2).sum()),
        "largest": float(weights.max() * 100),
        "positions": int(len(frame)),
    }


def portfolio_volatility(symbols: List[str], period: str = "1y") -> Optional[pd.DataFrame]:
    """Daily returns matrix for the held symbols (for correlation / volatility)."""
    series = {}
    for symbol in symbols or []:
        try:
            series[symbol] = history(symbol, period=period)["Returns"]
        except Exception:
            continue
    if not series:
        return None
    return pd.DataFrame(series).dropna(how="all")


# --------------------------------------------------------------------------- #
# Model scoring (chronological hold-out)
# --------------------------------------------------------------------------- #
def evaluate_models(symbol: str, period: str = "2y", test_fraction: float = 0.2) -> Dict[str, Any]:
    """Score the three models on the most recent slice of data (never shuffled)."""
    df = build_features(history(symbol, period=period)).dropna()
    if len(df) < 120:
        raise ValueError(f"Not enough history for {symbol}")

    split = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:split], df.iloc[split:]

    # Researcher — next-day close
    researcher = RandomForestRegressor(n_estimators=100, random_state=42)
    researcher.fit(train[["Close", "Vol", "MA5", "MA20"]], train["Target"])
    pred_price = researcher.predict(test[["Close", "Vol", "MA5", "MA20"]])
    mae = float(mean_absolute_error(test["Target"], pred_price))
    rmse = float(mean_squared_error(test["Target"], pred_price) ** 0.5)
    mean_price = float(test["Target"].mean())
    inverse_error = max(0.0, (1 - mae / mean_price) * 100) if mean_price else 0.0

    # Risk — expected absolute return
    risk_model = LinearRegression()
    risk_model.fit(train[["Vol", "Returns"]], train["Returns"].abs())
    risk_pred = risk_model.predict(test[["Vol", "Returns"]])
    r2 = float(r2_score(test["Returns"].abs(), risk_pred))
    current_risk = float(risk_model.predict(df[["Vol", "Returns"]].tail(1))[0])

    # Strategist — direction
    signal = np.where((df["Target"] > df["Close"]) & (df["Vol"] < df["Vol"].median()), 1,
                      np.where(df["Target"] < df["Close"], -1, 0))
    train_signal = signal[:split]
    test_signal = signal[split:]
    strategist = DecisionTreeClassifier(max_depth=3, random_state=42)
    strategist.fit(train[["Returns", "Vol"]], train_signal)
    signal_pred = strategist.predict(test[["Returns", "Vol"]])
    accuracy = float(accuracy_score(test_signal, signal_pred) * 100)
    f1 = float(f1_score(test_signal, signal_pred, average="weighted", zero_division=0) * 100)
    current_signal = int(strategist.predict(df[["Returns", "Vol"]].tail(1))[0])

    predicted_next = float(researcher.predict(df[["Close", "Vol", "MA5", "MA20"]].tail(1))[0])

    return {
        "symbol": symbol.upper(),
        "rows": int(len(df)),
        "test_rows": int(len(test)),
        "researcher": {
            "mae": mae,
            "rmse": rmse,
            "inverse_error": inverse_error,
            "predicted_next": predicted_next,
            "last_close": float(df["Close"].iloc[-1]),
        },
        "risk": {"r2": r2, "current": current_risk},
        "strategist": {
            "accuracy": accuracy,
            "f1": f1,
            "current": current_signal,
            "label": {1: "BUY", 0: "HOLD", -1: "SELL"}.get(current_signal, "HOLD"),
        },
    }
