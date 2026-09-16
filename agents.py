from models import AlderMLModels

class BaseAgent:
    def __init__(self, name):
        self.name = name

class ResearcherAgent(BaseAgent):
    def __init__(self, ml_engine):
        super().__init__("Market Researcher")
        self.engine = ml_engine

    def act(self):
        prediction = self.engine.train_researcher()
        return {
            "metric": "Predicted Price",
            "value": prediction,
            "status": "Success",
            "detail": f"Based on a RandomForestRegressor trained on 2 years of historical data, the predicted closing price for tomorrow is ${prediction:.2f}."
        }

class RiskAgent(BaseAgent):
    def __init__(self, ml_engine):
        super().__init__("Risk Manager")
        self.engine = ml_engine

    def act(self):
        risk_score = self.engine.train_risk_manager()
        return {
            "metric": "Volatility Score",
            "value": risk_score,
            "status": "Success",
            "detail": f"Expected volatility is {risk_score:.4f}. This was calculated using Linear Regression on absolute returns and rolling volatility, indicating the potential for price swings."
        }

class StrategistAgent(BaseAgent):
    def __init__(self, ml_engine):
        super().__init__("Trade Strategist")
        self.engine = ml_engine

    def act(self, price_pred, current_price, risk_score):
        signal = self.engine.train_strategist(price_pred, current_price, risk_score)
        signals = {1: "BUY", 0: "HOLD", -1: "SELL"}
        signal_text = signals.get(signal, "HOLD")
        return {
            "metric": "Trade Signal",
            "value": signal_text,
            "status": "Success",
            "detail": f"The strategist has issued a {signal_text} signal. This decision integrates the Researcher's price prediction (${price_pred:.2f}) against the current price (${current_price:.2f}) and the Risk Manager's volatility score ({risk_score:.4f})."
        }

class ManagerAgent(BaseAgent):
    def __init__(self, ml_engine):
        super().__init__("Portfolio Manager")
        self.engine = ml_engine

    def act(self, signal_val, risk_score):
        # Convert signal string back to numeric for the engine
        sig_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        allocation = self.engine.calculate_allocation(1 if signal_val == "BUY" else 0, risk_score)
        return {
            "metric": "Capital Allocation",
            "value": f"{allocation*100:.2f}%",
            "status": "Success",
            "detail": f"Based on the {signal_val} signal and a risk score of {risk_score:.4f}, the portfolio manager suggests allocating {allocation*100:.2f}% of available capital to this asset."
        }
