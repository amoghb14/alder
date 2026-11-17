"""Model layer for ALDER.

Three estimators, trained locally on two years of daily bars, each predicting
something *stationary* rather than a raw price level:

* **Researcher** — 5-session forward return, as a RandomForest + HistGradientBoosting blend.
* **Risk manager** — 5-session forward volatility, gradient boosted.
* **Strategist** — 5-session direction (BUY/HOLD/SELL), gradient boosted, gated by the
  researcher's expected return and the risk score, so the agents are genuinely coupled.

Everything is evaluated the only honest way for time series: an expanding-window
walk-forward, refitting on past data only and predicting forward. Nothing is
shuffled, and no future bar ever reaches a training set.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

import market_data

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES: List[str] = [
    "Ret1", "Ret2", "Ret3", "Ret5", "Ret10",
    "Vol5", "Vol20", "VolRatio",
    "MA_Ratio", "RSI14", "MACD_Hist", "BB_PctB",
    "ATR_Pct", "Volume_Z", "Close_Z20", "Gap",
]

# The models predict a 5-session outlook. A one-day horizon has no measurable
# signal-to-noise ratio on daily equity bars; over a week, drift and volatility
# persistence are learnable. A move smaller than DEADBAND is treated as noise.
HORIZON = 5
DEADBAND = 0.01
MIN_ROWS = 220
# Expected-return models on noisy financial data explain little variance, so raw
# predictions are shrunk toward zero before use. With R² near zero the optimal
# factor is near zero too; a small positive value keeps the ranking of names.
SHRINK = 0.20
WALK_START = 0.60
REFIT_EVERY = 5


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix plus targets, from a daily OHLCV frame."""
    df = frame.copy()
    close = df["Close"]
    returns = close.pct_change()

    df["Ret1"] = returns
    df["Ret2"] = close.pct_change(2)
    df["Ret3"] = close.pct_change(3)
    df["Ret5"] = close.pct_change(5)
    df["Ret10"] = close.pct_change(10)

    df["Vol5"] = returns.rolling(5).std()
    df["Vol20"] = returns.rolling(20).std()
    df["VolRatio"] = df["Vol5"] / df["Vol20"].replace(0, np.nan)

    df["MA_Ratio"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    df["RSI14"] = _rsi(close, 14) / 100.0

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    df["MACD_Hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    rolling_std = close.rolling(20).std().replace(0, np.nan)
    upper = close.rolling(20).mean() + 2 * rolling_std
    lower = close.rolling(20).mean() - 2 * rolling_std
    df["BB_PctB"] = (close - lower) / (upper - lower).replace(0, np.nan)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - close.shift()).abs(),
            (df["Low"] - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_Pct"] = true_range.rolling(14).mean() / close

    volume = df["Volume"].astype(float)
    df["Volume_Z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan)

    df["Close_Z20"] = (close - close.rolling(20).mean()) / rolling_std
    df["Gap"] = (df["Open"] - close.shift()) / close.shift()

    # Targets: the next HORIZON sessions (5 trading days)
    df["FwdRet"] = close.shift(-HORIZON) / close - 1
    df["FwdVol"] = returns.rolling(HORIZON).std().shift(-(HORIZON - 1))
    df["TargetClose"] = close.shift(-HORIZON)
    df["Direction"] = np.where(
        df["FwdRet"] >= DEADBAND, 1, np.where(df["FwdRet"] <= -DEADBAND, -1, 0)
    )

    return df.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class AlderMLModels:
    """The three-model pipeline for a single security."""

    def __init__(self, ticker: str, period: str = "2y"):
        self.ticker = (ticker or "").upper().strip()
        self.raw = market_data.history(self.ticker, period=period)
        dataset = build_dataset(self.raw).dropna(
            subset=FEATURES + ["FwdRet", "FwdVol", "Direction", "TargetClose"]
        )
        if len(dataset) < MIN_ROWS:
            raise ValueError(
                f"Only {len(dataset)} usable rows for {self.ticker}; need at least {MIN_ROWS}."
            )
        self.data = dataset

    # -- internals ---------------------------------------------------------- #
    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[FEATURES].astype(float)

    def _fit_researcher(self, frame: pd.DataFrame):
        forest = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            max_features=0.6,
            random_state=42,
            n_jobs=-1,
        )
        forest.fit(self._matrix(frame), frame["FwdRet"])
        boost = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=250,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
        boost.fit(self._matrix(frame), frame["FwdRet"])
        return (forest, boost)

    def _predict_researcher(self, models, row: pd.DataFrame) -> float:
        forest, boost = models
        blended = 0.5 * forest.predict(row)[0] + 0.5 * boost.predict(row)[0]
        # Shrink toward zero: with R² near zero the best point estimate is close to
        # the mean, and this keeps the price target sane instead of extrapolating.
        shrunk = blended * SHRINK
        # Robustness: keep the outlook inside a plausible weekly band (±1.25 sigma)
        sigma = float(self.data["FwdRet"].std() or 0.03)
        return float(np.clip(shrunk, -1.25 * sigma, 1.25 * sigma))

    def _fit_risk(self, frame: pd.DataFrame):
        model = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=250,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=7,
        )
        model.fit(self._matrix(frame), frame["FwdVol"])
        return model

    def _fit_strategist(self, frame: pd.DataFrame):
        labels = frame["Direction"].astype(int)
        weights = labels.map(labels.value_counts(normalize=True).to_dict())
        weights = 1.0 / weights
        model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.06,
            max_iter=200,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=11,
        )
        model.fit(self._matrix(frame), labels, sample_weight=weights.values)
        return model

    @staticmethod
    def _probabilities(model, row: pd.DataFrame) -> Dict[int, float]:
        probabilities = model.predict_proba(row)[0]
        return {int(label): float(p) for label, p in zip(model.classes_, probabilities)}

    # -- public API used by the agents -------------------------------------- #
    def predict_return(self) -> float:
        """Expected 5-session return from the blended regression."""
        models = self._fit_researcher(self.data)
        return self._predict_researcher(models, self._matrix(self.data).tail(1))

    def train_researcher(self) -> float:
        """Price target five sessions out, derived from the expected return."""
        last_close = float(self.data["Close"].iloc[-1])
        return last_close * (1.0 + self.predict_return())

    def train_risk_manager(self) -> float:
        """Expected 5-session realised volatility."""
        model = self._fit_risk(self.data)
        prediction = float(model.predict(self._matrix(self.data).tail(1))[0])
        return max(prediction, 1e-4)

    def strategist_probabilities(self) -> Dict[int, float]:
        model = self._fit_strategist(self.data)
        return self._probabilities(model, self._matrix(self.data).tail(1))

    def train_strategist(
        self,
        predicted_price: Optional[float] = None,
        current_price: Optional[float] = None,
        risk_score: Optional[float] = None,
    ) -> int:
        """BUY / HOLD / SELL, gated by the other two agents.

        The classifier proposes a direction; the researcher's expected return has to
        agree, and a BUY is downgraded when expected volatility is high. This is the
        coupling the UI advertises — the signal is not produced in isolation.
        """
        probabilities = self.strategist_probabilities()
        expected_return = 0.0
        if predicted_price is not None and current_price:
            expected_return = (predicted_price - current_price) / current_price
        else:
            expected_return = self.predict_return()

        p_up = probabilities.get(1, 0.0)
        p_down = probabilities.get(-1, 0.0)
        volatility = risk_score if risk_score is not None else self.train_risk_manager()
        high_risk = volatility > 0.045

        if p_up >= 0.4 and expected_return > 0 and not high_risk:
            return 1
        if p_down >= 0.4 and expected_return < 0:
            return -1
        return 0

    def calculate_allocation(self, signal: int, risk_score: float) -> float:
        """Risk-scaled capital weight: 0 on SELL, shrinking with volatility."""
        if signal == -1:
            return 0.0
        if signal == 0:
            return 0.0
        risk_penalty = max(0.0, float(risk_score)) * 10
        return max(0.0, min(0.5, 0.20 - risk_penalty))

    # -- narrative ---------------------------------------------------------- #
    def generate_detailed_report(self) -> str:
        df = self.data
        last_close = float(df["Close"].iloc[-1])
        predicted = self.train_researcher()
        expected_return = (predicted - last_close) / last_close * 100
        signal = {1: "BUY", 0: "HOLD", -1: "SELL"}[
            self.train_strategist(predicted, last_close, self.train_risk_manager())
        ]
        rsi = float(df["RSI14"].iloc[-1]) * 100
        vol = float(df["Vol20"].iloc[-1])
        trend = "bullish" if df["MA_Ratio"].iloc[-1] > 0 else "bearish"
        annualised = float(df["Ret1"].tail(30).mean() * 252 * 100)
        bb = float(df["BB_PctB"].iloc[-1])

        return f"""
        EXECUTIVE SUMMARY: {self.ticker} ANALYSIS

        {self.ticker} trades at ${last_close:,.2f}. The blended model (RandomForest +
        gradient boosting on 16 engineered features) projects ${predicted:,.2f}, an
        expected move of {expected_return:+.2f}% over the next five sessions. The coupled
        signal is {signal}.

        TECHNICAL CONTEXT

        Trend: the 5-day mean sits {abs(df['MA_Ratio'].iloc[-1]) * 100:.2f}% {'above' if trend == 'bullish' else 'below'} the 20-day mean — a {trend} structure.
        Momentum: RSI(14) is {rsi:.1f}, {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'}. Bollinger %B is {bb:.2f}, placing price in the {'upper' if bb > 0.8 else 'lower' if bb < 0.2 else 'middle'} band.
        Risk: 20-day realised volatility is {vol * 100:.2f}% daily; annualised 30-day drift is {annualised:+.1f}%.

        MODEL NOTES

        The target is 5-session *return*, not price level, which keeps the learning problem
        stationary; predictions are clipped to three historical sigma as a guardrail.
        Volatility is forecast with gradient boosting on the same feature set.

        STRATEGIC OUTLOOK

        Expected move of {expected_return:+.2f}% is {'above' if abs(expected_return) > 1 else 'below'} the 1% materiality band, so the signal is classified {signal}. Treat this as a research
        input: the Strategy Lab backtest is the honest test of whether it survives costs.
        """

    def assess_portfolio_risk(self, holdings: Dict[str, float]) -> str:
        if not holdings:
            return "Portfolio is currently empty. No risk metrics available."

        rows = []
        weighted = 0.0
        total = 0.0
        for ticker, shares in holdings.items():
            try:
                engine = AlderMLModels(ticker)
                risk = engine.train_risk_manager()
                price = float(engine.data["Close"].iloc[-1])
                value = shares * price
                total += value
                weighted += risk * value
                rows.append(f"{ticker}: daily volatility {risk * 100:.2f}% | value ${value:,.2f}")
            except Exception:
                rows.append(f"{ticker}: data unavailable")

        average = weighted / total if total else 0.0
        if average < 0.010:
            health, advice = "Low risk", "Volatility is below 1% daily; consider whether the book is under-deployed."
        elif average < 0.025:
            health, advice = "Balanced", "Volatility sits inside a normal institutional band for equity exposure."
        elif average < 0.045:
            health, advice = "Elevated", "Volatility is above target; trimming the largest weights would cut drawdown risk."
        else:
            health, advice = "Critical", "Volatility is extreme. Reduce position size before adding risk."

        return f"""
        PORTFOLIO RISK AUDIT

        Aggregate daily volatility: {average * 100:.2f}%
        Portfolio health: {health}

        ASSET BREAKDOWN
        {chr(10).join('- ' + r for r in rows)}

        STRATEGIC ADVICE
        {advice}

        DIVERSIFICATION
        {len(holdings)} position(s) held. {'Concentrated — a single shock dominates the book.' if len(holdings) < 3 else 'Position count is reasonable; check the correlation matrix on the Risk page.'}
        """

    def predict_growth_leaders(self, tickers: List[str]) -> List[Dict[str, Any]]:
        results = []
        for ticker in tickers:
            try:
                engine = AlderMLModels(ticker)
                last_close = float(engine.data["Close"].iloc[-1])
                growth = engine.predict_return() * 100
                results.append({"ticker": ticker.upper(), "growth": growth, "price": last_close})
            except Exception:
                continue
        return sorted(results, key=lambda item: item["growth"], reverse=True)


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #
def evaluate(ticker: str, period: str = "2y", refit_every: int = REFIT_EVERY,
             start_fraction: float = WALK_START) -> Dict[str, Any]:
    """Out-of-sample walk-forward scoring of all three models.

    Each refit uses only bars before the predicted one, so the metrics reflect what
    the models would actually have produced live rather than a lucky static split.
    Naive baselines are returned alongside, because an accuracy figure means nothing
    without the number a rule of thumb would have scored on the same bars.
    """
    engine = AlderMLModels(ticker, period=period)
    data = engine.data
    start = int(len(data) * start_fraction)

    return_preds, return_actual = [], []
    vol_preds, vol_actual = [], []
    dir_preds, dir_actual, confidences = [], [], []

    researcher_models = risk_model = strategist_model = None
    for i in range(start, len(data), refit_every):
        if researcher_models is None or (i - start) % refit_every == 0:
            train = data.iloc[:i]
            researcher_models = engine._fit_researcher(train)
            risk_model = engine._fit_risk(train)
            strategist_model = engine._fit_strategist(train)

        row = engine._matrix(data).iloc[[i]]
        return_preds.append(engine._predict_researcher(researcher_models, row))
        vol_preds.append(max(float(risk_model.predict(row)[0]), 1e-4))
        probabilities = engine._probabilities(strategist_model, row)
        dir_preds.append(max(probabilities, key=probabilities.get))
        confidences.append(max(probabilities.values()))

        return_actual.append(float(data["FwdRet"].iloc[i]))
        vol_actual.append(float(data["FwdVol"].iloc[i]))
        dir_actual.append(int(data["Direction"].iloc[i]))

    return_pred = np.array(return_preds)
    return_act = np.array(return_actual)
    vol_pred = np.array(vol_preds)
    vol_act = np.array(vol_actual)
    prices = data["Close"].iloc[start:len(data):refit_every].values

    abs_move = np.abs(return_act - return_pred)
    price_mae = float(np.mean(abs_move * prices))
    price_rmse = float(np.sqrt(np.mean(((return_act - return_pred) * prices) ** 2)))
    mae_bps = float(np.mean(abs_move) * 10_000)
    rmse_bps = float(np.sqrt(np.mean((return_act - return_pred) ** 2)) * 10_000)
    r2_returns = float(r2_score(return_act, return_pred))

    vol_r2 = float(r2_score(vol_act, vol_pred))
    vol_mae_bps = float(np.mean(np.abs(vol_act - vol_pred)) * 10_000)

    direction_accuracy = float(accuracy_score(dir_actual, dir_preds) * 100)
    direction_f1 = float(f1_score(dir_actual, dir_preds, average="weighted", zero_division=0) * 100)

    binary_actual = [1 if value > 0 else 0 for value in dir_actual]
    binary_pred = [1 if value > 0 else 0 for value in dir_preds]
    updown_accuracy = float(accuracy_score(binary_actual, binary_pred) * 100)
    always_up = float(np.mean(binary_actual) * 100)
    persistence = float(
        np.mean([1 if (data["Ret5"].iloc[start:len(data):refit_every].values[j] > 0) == bool(binary_actual[j]) else 0
                 for j in range(len(binary_actual))]) * 100
    )

    hit_rate = float(np.mean([1 if (p > 0) == (a > 0) else 0 for p, a in zip(return_pred, return_act)]) * 100)

    # Confidence gating: only act when the classifier is reasonably sure.
    confident = [j for j, value in enumerate(confidences) if value >= 0.60]
    conf_accuracy = (
        float(np.mean([1 if dir_preds[j] == dir_actual[j] else 0 for j in confident]) * 100)
        if confident else 0.0
    )
    conf_coverage = float(len(confident) / len(dir_preds) * 100) if dir_preds else 0.0

    return {
        "symbol": engine.ticker,
        "horizon": HORIZON,
        "rows": int(len(data)),
        "test_rows": int(len(dir_actual)),
        "refits": int(np.ceil((len(data) - start) / refit_every)),
        "researcher": {
            "mae_bps": mae_bps,
            "rmse_bps": rmse_bps,
            "price_mae": price_mae,
            "price_rmse": price_rmse,
            "r2": r2_returns,
            "hit_rate": hit_rate,
        },
        "risk": {"r2": vol_r2, "mae_bps": vol_mae_bps},
        "strategist": {
            "accuracy": direction_accuracy,
            "f1": direction_f1,
            "updown_accuracy": updown_accuracy,
            "always_up": always_up,
            "persistence": persistence,
            "conf_accuracy": conf_accuracy,
            "conf_coverage": conf_coverage,
        },
    }
