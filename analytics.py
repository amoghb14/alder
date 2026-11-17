"""Quantitative analytics: walk-forward backtest, risk metrics, model scoring.

The backtest answers the question a research desk actually asks — *does the signal
make money after costs?* — and it is deliberately strict: expanding-window refits,
a 5-session holding period matching the model's horizon, a transaction cost on
every position change, and a buy & hold benchmark.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import models
from market_data import CACHE_DIR, history

TRADING_DAYS = 252
EVAL_TTL = 12 * 3600


# --------------------------------------------------------------------------- #
# Small disk cache for expensive walk-forward evaluation
# --------------------------------------------------------------------------- #
def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"eval_{symbol.upper()}.json")


def _read_cache(symbol: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(symbol)
    try:
        if os.path.exists(path) and (pd.Timestamp.now().timestamp() - os.path.getmtime(path)) < EVAL_TTL:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception:
        pass
    return None


def _write_cache(symbol: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(symbol), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(daily: pd.Series) -> Dict[str, float]:
    daily = daily.dropna()
    if daily.empty:
        return {"total_return": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "max_drawdown": 0.0, "calmar": 0.0}
    equity = (1 + daily).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = max(len(daily) / TRADING_DAYS, 1e-9)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else 0.0
    volatility = float(daily.std() * np.sqrt(TRADING_DAYS))
    downside = daily[daily < 0]
    sortino = (
        float(daily.mean() / downside.std() * np.sqrt(TRADING_DAYS))
        if len(downside) > 1 and downside.std() > 0 else 0.0
    )
    peak = equity.cummax()
    max_dd = float((equity / peak - 1).min())
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
    threshold: Optional[float] = None,
    start_fraction: float = 0.6,
) -> Dict[str, Any]:
    """Long/flat walk-forward backtest of the 5-session directional signal.

    On each rebalance the classifier is refitted on data up to that day only, and
    the resulting position is held for ``models.HORIZON`` sessions. Costs are charged
    whenever the position changes.
    """
    engine = models.AlderMLModels(symbol, period=period)
    frame = engine.data
    if len(frame) < 120:
        raise ValueError(f"Not enough history for {symbol} to backtest")

    horizon = models.HORIZON
    start = int(len(frame) * start_fraction)
    positions = pd.Series(0.0, index=frame.index)
    probabilities_series = pd.Series(np.nan, index=frame.index)

    classifier = None
    for i in range(start, len(frame), horizon):
        if classifier is None or (i - start) % horizon == 0:
            classifier = engine._fit_strategist(frame.iloc[:i])
        probabilities = engine._probabilities(classifier, engine._matrix(frame).iloc[[i]])
        p_up = probabilities.get(1, 0.0)
        probabilities_series.iloc[i] = p_up
        call = max(probabilities, key=probabilities.get)
        go_long = call == 1 and (threshold is None or p_up >= threshold)
        positions.iloc[i : i + horizon] = 1.0 if go_long else 0.0

    forward_returns = frame["Close"].pct_change().shift(-1)
    gross = positions * forward_returns
    traded = positions.diff().abs().fillna(positions.abs())
    costs = traded * (cost_bps / 10_000.0)
    strategy_returns = (gross - costs).dropna()
    benchmark_returns = forward_returns.reindex(strategy_returns.index).fillna(0.0)

    curve = pd.DataFrame({
        "Strategy": (1 + strategy_returns).cumprod(),
        "Buy & Hold": (1 + benchmark_returns).cumprod(),
    })

    active = strategy_returns[positions.reindex(strategy_returns.index) > 0]
    metrics = {
        "strategy": _metrics(strategy_returns),
        "benchmark": _metrics(benchmark_returns),
        "hit_rate": float((active > 0).mean() * 100) if len(active) else 0.0,
        "exposure": float((positions.reindex(strategy_returns.index) > 0).mean() * 100),
        "trades": int(traded.sum()),
        "cost_bps": cost_bps,
        "days": int(len(strategy_returns)),
        "threshold": (threshold or 0.0) * 100,
        "horizon": horizon,
    }
    metrics["excess_return"] = (
        metrics["strategy"]["total_return"] - metrics["benchmark"]["total_return"]
    ) * 100

    return {
        "symbol": engine.ticker,
        "curve": curve,
        "metrics": metrics,
        "positions": positions,
        "probabilities": probabilities_series,
        "frame": frame,
    }


# --------------------------------------------------------------------------- #
# Portfolio analytics
# --------------------------------------------------------------------------- #
def holdings_frame(holdings: Dict[str, float], quote_lookup) -> pd.DataFrame:
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
    if frame is None or frame.empty or frame["Value"].sum() <= 0:
        return {"hhi": 0.0, "largest": 0.0, "positions": 0}
    weights = (frame["Value"] / frame["Value"].sum()).values
    return {"hhi": float((weights**2).sum()), "largest": float(weights.max() * 100), "positions": int(len(frame))}


def portfolio_volatility(symbols: List[str], period: str = "1y") -> Optional[pd.DataFrame]:
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
# Model scoring used by the UI (cached, lighter walk-forward)
# --------------------------------------------------------------------------- #
def evaluate_models(symbol: str, use_cache: bool = True) -> Dict[str, Any]:
    """Walk-forward scores for the three models, mapped for the UI."""
    symbol = symbol.upper()
    if use_cache:
        cached = _read_cache(symbol)
        if cached:
            return cached

    scores = models.evaluate(symbol, refit_every=10, start_fraction=0.7)
    engine = models.AlderMLModels(symbol)
    signal = engine.train_strategist()

    payload = {
        "symbol": symbol,
        "horizon": scores["horizon"],
        "rows": scores["rows"],
        "test_rows": scores["test_rows"],
        "refits": scores["refits"],
        "researcher": {
            "mae": scores["researcher"]["price_mae"],
            "rmse": scores["researcher"]["price_rmse"],
            "mae_bps": scores["researcher"]["mae_bps"],
            "r2": scores["researcher"]["r2"],
            "hit_rate": scores["researcher"]["hit_rate"],
        },
        "risk": {
            "r2": scores["risk"]["r2"],
            "mae_bps": scores["risk"]["mae_bps"],
            "current": engine.train_risk_manager(),
        },
        "strategist": {
            "accuracy": scores["strategist"]["accuracy"],
            "f1": scores["strategist"]["f1"],
            "updown_accuracy": scores["strategist"]["updown_accuracy"],
            "always_up": scores["strategist"]["always_up"],
            "current": signal,
            "label": {1: "BUY", 0: "HOLD", -1: "SELL"}.get(signal, "HOLD"),
        },
    }
    if use_cache:
        _write_cache(symbol, payload)
    return payload
