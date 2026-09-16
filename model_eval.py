import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, r2_score

def evaluate_alder_models(ticker="NVDA"):
    print(f"--- Evaluating Alder ML Models for {ticker} ---")

    # 1. Data Preparation
    df = yf.download(ticker, period="2y", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['Returns'] = df['Close'].pct_change()
    df['Vol'] = df['Close'].rolling(window=5).std()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Target'] = df['Close'].shift(-1)

    # Signal target for Strategist
    df['Signal'] = np.where((df['Target'] > df['Close']) & (df['Vol'] < df['Vol'].median()), 1,
                            np.where(df['Target'] < df['Close'], -1, 0))

    df = df.dropna()

    # --- MODEL 1: Researcher (Price Prediction) ---
    X_res = df[['Close', 'Vol', 'MA5', 'MA20']]
    y_res = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, shuffle=False)

    res_model = RandomForestRegressor(n_estimators=100, random_state=42)
    res_model.fit(X_train, y_train)
    res_preds = res_model.predict(X_test)
    res_mae = mean_absolute_error(y_test, res_preds)
    res_rmse = np.sqrt(mean_squared_error(y_test, res_preds))

    # --- MODEL 2: Risk Manager (Volatility Prediction) ---
    X_risk = df[['Vol', 'Returns']]
    y_risk = df['Returns'].abs()
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_risk, y_risk, test_size=0.2, shuffle=False)

    risk_model = LinearRegression()
    risk_model.fit(X_train_r, y_train_r)
    risk_preds = risk_model.predict(X_test_r)
    risk_r2 = r2_score(y_test_r, risk_preds)

    # --- MODEL 3: Strategist (Signal Classification) ---
    X_strat = df[['Returns', 'Vol']]
    y_strat = df['Signal']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_strat, y_strat, test_size=0.2, shuffle=False)

    strat_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    strat_model.fit(X_train_s, y_train_s)
    strat_preds = strat_model.predict(X_test_s)
    strat_acc = accuracy_score(y_test_s, strat_preds)
    strat_f1 = f1_score(y_test_s, strat_preds, average='weighted')

    print(f"\n[Researcher - Price Prediction]")
    print(f"MAE: ${res_mae:.2f}")
    print(f"RMSE: ${res_rmse:.2f}")
    print(f"Accuracy: {100 - (res_mae/y_test.mean()*100):.2f}% (Inverse Error)")

    print(f"\n[Risk Manager - Volatility Prediction]")
    print(f"R2 Score: {risk_r2:.4f}")

    print(f"\n[Strategist - Signal Classification]")
    print(f"Accuracy: {strat_acc*100:.2f}%")
    print(f"F1-Score: {strat_f1*100:.2f}%")

if __name__ == "__main__":
    evaluate_alder_models()
