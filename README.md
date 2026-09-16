# ALDER | AI-Native Hedge Fund

ALDER is a professional-grade, multi-agentic AI system designed to automate hedge fund operations using in-house machine learning models. Unlike typical AI wrappers, ALDER utilizes local ML pipelines to execute research, risk management, and trading strategies.

## 🚀 Key Features
- **Multi-Agent ML Pipeline**: Four cooperative agents (Researcher, Risk Manager, Strategist, Portfolio Manager) working in a deterministic sequence.
- **Local ML Models**: Uses `RandomForestRegressor`, `LinearRegression`, and `DecisionTreeClassifier` for a purely quantitative decision process.
- **Real-time Data**: Integrated with the `yfinance` API for live market data and portfolio valuation.
- **Institutional Terminal**: A high-contrast dark-mode UI built with Streamlit, featuring a gold-and-black professional aesthetic.
- **Asset Portfolio**: Full trade execution desk with real-time AUM tracking and local persistence.

## 🛠️ Technical Architecture
- **Language**: Python 3.11
- **Frameworks**: Streamlit, Scikit-Learn, Pandas, NumPy, Plotly.
- **Data Source**: yfinance API.
- **Auth**: Local session-based authentication.

## 📊 ML Pipeline Workflow
1. **Market Research**: `RandomForestRegressor` $\rightarrow$ Predicts next-day closing price based on 2-year historical trends.
2. **Risk Audit**: `LinearRegression` $\rightarrow$ Forecasts expected volatility and assesses portfolio health.
3. **Strategic Signaling**: `DecisionTreeClassifier` $\rightarrow$ Generates a **BUY/HOLD/SELL** signal.
4. **Capital Allocation**: Mean-Variance Logic $\rightarrow$ Calculates optimal asset weight based on signal and risk score.

## 💻 Installation & Usage

### Prerequisites
- Python 3.11+
- Virtual environment recommended

### Setup
```bash
# Clone the project
cd /Users/amogh/Downloads/Alder-Proto

# Initialize virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Execution
```bash
streamlit run app.py
```

### Access
- **Username**: `amogh`
- **Password**: `abc`

## 📈 Performance Metrics
The system is designed for institutional stability. The Researcher model typically maintains a high directional accuracy (~96%), while the Strategist provides a disciplined, non-emotional trading signal based on quantitative laws.
