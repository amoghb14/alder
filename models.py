import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

class AlderMLModels:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self._prepare_data()

    def _prepare_data(self):
        df = yf.download(self.ticker, period="2y", interval="1d")
        if df.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['Returns'] = df['Close'].pct_change()
        df['Vol'] = df['Close'].rolling(window=5).std()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Target'] = df['Close'].shift(-1)

        return df.dropna()

    def train_researcher(self):
        df = self.data
        X = df[['Close', 'Vol', 'MA5', 'MA20']]
        y = df['Target']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        last_row = X.iloc[-1].values.reshape(1, -1)
        return model.predict(last_row)[0]

    def train_risk_manager(self):
        df = self.data
        X = df[['Vol', 'Returns']]
        y = df['Returns'].abs()
        model = LinearRegression()
        model.fit(X, y)
        last_row = X.iloc[-1].values.reshape(1, -1)
        return model.predict(last_row)[0]

    def train_strategist(self, predicted_price, current_price, risk_score):
        df = self.data
        df['Signal'] = np.where((df['Target'] > df['Close']) & (df['Vol'] < df['Vol'].median()), 1,
                                np.where(df['Target'] < df['Close'], -1, 0))
        X = df[['Returns', 'Vol']]
        y = df['Signal']
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)
        current_state = np.array([[df['Returns'].iloc[-1], df['Vol'].iloc[-1]]])
        return model.predict(current_state)[0]

    def calculate_allocation(self, signal, risk_score):
        if signal == -1: return 0.0
        base_allocation = 0.20
        risk_penalty = risk_score * 10
        return max(0, min(0.5, base_allocation - risk_penalty))

    def generate_detailed_report(self):
        df = self.data
        last_close = df['Close'].iloc[-1]
        pred_price = self.train_researcher()
        expected_return = ((pred_price - last_close) / last_close) * 100
        ma5 = df['MA5'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        vol = df['Vol'].iloc[-1]
        returns = df['Returns'].tail(30).mean() * 252 * 100

        report = f"""
        EXECUTIVE SUMMARY: {self.ticker} ANALYSIS

        The asset is currently trading at ${last_close:.2f}. Our ML pipeline suggests a target price of ${pred_price:.2f}, representing an expected movement of {expected_return:.2f}% over the immediate horizon.

        TECHNICAL ANALYSIS

        Price Momentum: The 5-day moving average ({ma5:.2f}) relative to the 20-day moving average ({ma20:.2f}) indicates a {'bullish' if ma5 > ma20 else 'bearish'} trend. This suggests that short-term momentum is {'aligning with' if ma5 > ma20 else 'diverging from'} the broader monthly trend.

        Volatility Profile: The current rolling volatility is recorded at {vol:.4f}. Compared to historical norms, this suggests a {'low' if vol < df['Vol'].median() else 'high'} risk environment. Annualized returns over the last 30 trading sessions are approximately {returns:.2f}%.

        ML MODEL INSIGHTS

        The RandomForestRegressor utilized for this prediction has analyzed a 2-year historical dataset consisting of price action, volatility, and moving average convergence. The model identified key patterns in the relation between MA5/MA20 and subsequent price shifts.

        STRATEGIC OUTLOOK

        Based on the quantitative data, {self.ticker} is exhibiting {'strong' if expected_return > 2 else 'moderate'} growth potential. Investors should monitor the {ma20:.2f} support level closely.

        The convergence of the ML prediction and the technical indicators suggests that the asset is currently in a {'consolidation' if abs(expected_return) < 1 else 'trend'} phase. We recommend a {'Long' if expected_return > 0 else 'Short/Neutral'} bias for the upcoming trading session.
        """
        return report

    def assess_portfolio_risk(self, holdings):
        """Analyzes risk across multiple assets and provides nuanced advisory."""
        if not holdings:
            return "Portfolio is currently empty. No risk metrics available."

        total_risk = 0
        details = []
        total_value = 0

        for ticker, shares in holdings.items():
            try:
                temp_engine = AlderMLModels(ticker)
                risk_score = temp_engine.train_risk_manager()
                price = yf.Ticker(ticker).info.get("currentPrice", 0)
                asset_value = shares * price
                total_value += asset_value
                total_risk += risk_score * asset_value
                details.append(f"{ticker}: Volatility {risk_score:.4f} | Value: ${asset_value:,.2f}")
            except:
                details.append(f"{ticker}: Data unavailable for risk analysis.")

        avg_risk = total_risk / total_value if total_value > 0 else 0

        if avg_risk < 0.01:
            advice = "CONSERVATIVE: Your portfolio is extremely stable. While risk is minimized, you may be missing growth opportunities. Consider diversifying into high-beta assets to capture market upside."
            health = "Low Risk / Under-exposed"
        elif avg_risk < 0.03:
            advice = "BALANCED: Your risk profile is aligned with standard institutional benchmarks. Current volatility is manageable. Maintain positions and rebalance quarterly."
            health = "Optimal / Balanced"
        elif avg_risk < 0.06:
            advice = "AGGRESSIVE: High volatility detected. Your portfolio is susceptible to significant drawdowns. We recommend introducing hedging instruments or increasing allocation to low-volatility indices."
            health = "High Risk / Volatile"
        else:
            advice = "CRITICAL: Extreme volatility detected. Your current positions are highly unstable. Immediate risk mitigation is required. Reduce leverage and shift capital to capital-preservation assets."
            health = "Critical / Unstable"

        report = f"""
        PORTFOLIO RISK AUDIT

        Aggregate Risk Score: {avg_risk:.4f}
        Portfolio Health: {health}

        ASSET BREAKDOWN:
        {chr(10).join(details)}

        STRATEGIC ADVICE:
        {advice}

        DIVERSIFICATION ANALYSIS:
        The current portfolio consists of {len(holdings)} unique assets. {'High' if len(holdings) < 3 else 'Moderate'} concentration risk detected. Recommended action: {'Increase diversification to reduce unsystematic risk' if len(holdings) < 3 else 'Maintain current allocation'}.
        """
        return report

    def predict_growth_leaders(self, tickers):
        growth_results = []
        for ticker in tickers:
            try:
                temp_engine = AlderMLModels(ticker)
                last_close = temp_engine.data['Close'].iloc[-1]
                pred_price = temp_engine.train_researcher()
                growth = ((pred_price - last_close) / last_close) * 100
                growth_results.append({"ticker": ticker, "growth": growth})
            except:
                continue
        return sorted(growth_results, key=lambda x: x['growth'], reverse=True)
