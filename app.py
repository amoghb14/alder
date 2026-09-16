import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import json
import os
from urllib.parse import urlparse
from models import AlderMLModels
from agents import ResearcherAgent, RiskAgent, StrategistAgent, ManagerAgent
import database as db

# --- DB INITIALIZATION ---
db.init_db()
db.seed_test_account()

st.set_page_config(page_title="ALDER | Asset Management", layout="wide")

# --- DARK PROFESSIONAL THEMING ---
st.markdown(f"""
    <style>
    .stApp {{
        background-color: #0E1117;
        color: #FFFFFF;
    }}
    .stApp p, .stApp span, .stApp div, .stApp label {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: #000000 !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: #D4AF37 !important;
    }}
    div[role="radiogroup"] {{
        gap: 10px !important;
    }}
    div[role="radiogroup"] label {{
        background-color: #1A1A1A !important;
        border: 1px solid #D4AF37 !important;
        color: #FFFFFF !important;
        padding: 10px 15px !important;
        border-radius: 0px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }}
    div[role="radiogroup"] label:hover {{
        background-color: #D4AF37 !important;
        color: #000000 !important;
    }}
    div[role="radiogroup"] input {{
        display: none !important;
    }}
    .stButton>button {{
        background-color: #D4AF37 !important;
        color: #000000 !important;
        border-radius: 0px !important;
        border: none !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
    }}
    [data-testid="stMetric"] {{
        background-color: #1A1A1A !important;
        border-left: 5px solid #D4AF37 !important;
        padding: 10px !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stMetric"] div {{
        color: #FFFFFF !important;
    }}
    h1, h2, h3 {{
        color: #FFFFFF !important;
        font-family: 'Times New Roman', serif !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid #D4AF37 !important;
        padding-bottom: 10px !important;
    }}
    .stTextInput > div > div > input {{
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 2px solid #D4AF37 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.authenticated:
    auth_mode = st.radio("Account Access", ["Login", "Create Account"], horizontal=True)

    if auth_mode == "Login":
        st.title("ALDER AUTHENTICATION")
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Access Terminal")
            if submit:
                user_data = db.authenticate_user(user, pwd)
                if user_data:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_data["id"]
                    st.session_state.username = user_data["username"]
                    st.rerun()
                else:
                    st.error("Unauthorized Credentials")
    else:
        st.title("CREATE ALDER ACCOUNT")
        with st.form("reg_form"):
            new_user = st.text_input("Username")
            new_email = st.text_input("Email Address")
            new_pwd = st.text_input("Password", type="password")
            submit_reg = st.form_submit_button("Register Account")
            if submit_reg:
                success, msg = db.create_user(new_user, new_email, new_pwd)
                if success:
                    st.success(msg)
                    st.info("You can now log in with your credentials.")
                else:
                    st.error(msg)
    st.stop()

# --- SESSION STATE ---
if "portfolio" not in st.session_state:
    st.session_state.portfolio = db.get_portfolio(st.session_state.user_id)

if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "NVDA"

if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

# --- NAVIGATION ---
st.sidebar.title("ALDER")
st.sidebar.markdown("---")
st.sidebar.write(f"User: **{st.session_state.username}**")
st.sidebar.write("Status: **Authorized**")
st.sidebar.markdown("---")

page = st.sidebar.radio("System Navigation", [
    "Main Dashboard",
    "Researcher",
    "Risk Manager",
    "Strategist",
    "Portfolio Manager",
    "Asset Portfolio"
])

# --- UTILS ---
POPULAR_STOCKS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Amazon": "AMZN", "Meta": "META",
    "Nvidia": "NVDA", "Tesla": "TSLA", "Berkshire Hathaway": "BRK-B", "Eli Lilly": "LLY", "Broadcom": "AVGO",
    "JPMorgan Chase": "JPM", "Visa": "V", "UnitedHealth": "UNH", "Exxon Mobil": "XOM", "Johnson & Johnson": "JNJ",
    "Walmart": "WMT", "Mastercard": "MA", "Procter & Gamble": "PG", "Home Depot": "HD", "Chevron": "CVX"
}
STOCK_OPTIONS = [f"{name} ({tick})" for name, tick in POPULAR_STOCKS.items()]

def parse_ticker(selected_option):
    if not selected_option:
        return None
    if "(" in selected_option:
        return selected_option.split("(")[-1].replace(")", "")
    return selected_option.upper()

def get_company_logo(ticker):
    try:
        info = yf.Ticker(ticker).info
        website = info.get("website")
        if website:
            # Extract domain and strip 'www.' for better Clearbit compatibility
            domain = urlparse(website).netloc
            if domain.startswith("www."):
                domain = domain[4:]
            return f"https://logo.clearbit.com/{domain}"
    except Exception:
        pass
    # Fallback to a high-quality professional placeholder if logo fails
    return f"https://ui-avatars.com/api/?name={ticker}&background=D4AF37&color=000000&size=128"

def run_pipeline(ticker):
    try:
        ml_engine = AlderMLModels(ticker)
        current_price = yf.Ticker(ticker).info.get("currentPrice", 0)
        researcher = ResearcherAgent(ml_engine)
        risk_mgr = RiskAgent(ml_engine)
        strategist = StrategistAgent(ml_engine)
        manager = ManagerAgent(ml_engine)
        res_out = researcher.act()
        risk_out = risk_mgr.act()
        strat_out = strategist.act(res_out["value"], current_price, risk_out["value"])
        mgr_out = manager.act(strat_out["value"], risk_out["value"])
        return {
            "ticker": ticker,
            "current_price": current_price,
            "researcher": res_out,
            "risk": risk_out,
            "strategist": strat_out,
            "manager": mgr_out
        }
    except Exception as e:
        st.error(f"Pipeline Error: {e}")
        return None

# --- PAGES ---

if page == "Main Dashboard":
    st.title("Dashboard")
    col_input, col_action = st.columns([2, 1])
    with col_input:
        # Use selectbox for autocomplete behavior
        selected_stock = st.selectbox("Select Company or Enter Ticker",
                                     options=["Custom Ticker..."] + STOCK_OPTIONS,
                                     index=0)

        if selected_stock == "Custom Ticker...":
            custom_ticker = st.text_input("Enter Ticker (e.g. TSLA)").upper()
            st.session_state.current_ticker = custom_ticker if custom_ticker else "NVDA"
        else:
            st.session_state.current_ticker = parse_ticker(selected_stock)

    with col_action:
        if st.button("Execute Strategy Pipeline"):
            with st.spinner("Processing ML Models..."):
                results = run_pipeline(st.session_state.current_ticker)
                st.session_state.pipeline_results = results

    if st.session_state.pipeline_results:
        res = st.session_state.pipeline_results
        ticker = res['ticker']
        col_logo, col_name = st.columns([1, 5])
        with col_logo:
            st.image(get_company_logo(ticker), width=80)
        with col_name:
            company_name = yf.Ticker(ticker).info.get("longName", ticker)
            st.subheader(company_name)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Price", f"${res['researcher']['value']:.2f}")
        c2.metric("Volatility", f"{res['risk']['value']:.4f}")
        c3.metric("Signal", res['strategist']['value'])
        c4.metric("Suggested Allocation", res['manager']['value'])

        st.subheader(f"Market Analytics: {ticker}")
        df = yf.download(ticker, period="1mo")
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
            )])
            fig.update_layout(xaxis_rangeslider_visible=False, height=400,
                              plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                              font=dict(color="white"),
                              xaxis=dict(gridcolor='#333333'),
                              yaxis=dict(gridcolor='#333333'))
            st.plotly_chart(fig, use_container_width=True)

elif page == "Researcher":
    st.title("Market Research")
    selected_res = st.selectbox("Select Asset for Report",
                               options=["Custom Ticker..."] + STOCK_OPTIONS,
                               index=0)

    if selected_res == "Custom Ticker...":
        search_ticker = st.text_input("Enter Ticker").upper()
    else:
        search_ticker = parse_ticker(selected_res)

    if st.button("Generate Report"):
        if search_ticker:
            with st.spinner("Synthesizing ML Data..."):
                try:
                    engine = AlderMLModels(search_ticker)
                    report = engine.generate_detailed_report()
                    st.markdown("---")
                    st.text(report)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        else:
            st.error("Please provide a valid ticker.")

elif page == "Risk Manager":
    st.title("Risk Management")
    if st.button("Audit Current Portfolio"):
        with st.spinner("Calculating Systematic Risk..."):
            engine = AlderMLModels("SPY")
            report = engine.assess_portfolio_risk(st.session_state.portfolio['holdings'])
            st.markdown("---")
            st.text(report)

elif page == "Strategist":
    st.title("Strategic Growth Planner")
    tickers_to_scan = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "LLY", "V"]
    if st.button("Scan Market for Growth Leaders"):
        with st.spinner("Running Cross-Asset ML Growth Prediction..."):
            engine = AlderMLModels("SPY")
            leaders = engine.predict_growth_leaders(tickers_to_scan)
            st.markdown("---")
            st.subheader("Top Predicted Growth Assets")
            data = [{"Ticker": x['ticker'], "Expected Growth (%)": f"{x['growth']:.2f}%"} for x in leaders]
            st.table(pd.DataFrame(data))
            st.info("Growth is predicted using a RandomForestRegressor comparing current trends to 2-year historical cycles.")

elif page == "Portfolio Manager":
    st.title("Portfolio Management")

    # 1. User Portfolio Data
    portfolio = st.session_state.portfolio
    holdings = portfolio['holdings']
    balance = portfolio['balance']

    # Calculate Total AUM
    total_value = balance
    holdings_list = []
    for ticker, shares in holdings.items():
        try:
            price = yf.Ticker(ticker).info.get("currentPrice", 0)
            val = shares * price
            total_value += val
            holdings_list.append({"Ticker": ticker, "Shares": shares, "Price": price, "Value": val})
        except:
            pass

    # Layout: AUM and Cash
    c1, c2 = st.columns(2)
    c1.metric("Total Assets Under Management", f"${total_value:,.2f}")
    c2.metric("Liquid Cash Reserves", f"${balance:,.2f}")

    st.markdown("---")

    # 2. Detailed Overarching Overview (Institutional Narrative)
    st.subheader("Institutional Portfolio Overview")

    # Generate the comprehensive narrative
    # We use a mix of calculated data and professional finance templates to ensure a detailed, long-form output.
    asset_count = len(holdings)
    diversification = "diversified" if asset_count > 3 else "concentrated"
    risk_profile = "Aggressive" if asset_count < 3 else "Balanced"

    # Constructing a deep-dive analysis
    narrative = f"""
    STRATEGIC ASSET ALLOCATION ANALYSIS

    The current portfolio is structured with a total Assets Under Management (AUM) of ${total_value:,.2f}, reflecting a current liquidity ratio of {(balance/total_value*100):.2f}%. This allocation suggests a {'conservative' if (balance/total_value) > 0.3 else 'aggressive'} stance towards current market conditions.

    ASSET COMPOSITION & EXPOSURE

    The portfolio currently holds {asset_count} unique positions. This reflects a {diversification} strategy. A concentrated portfolio of this nature allows for higher alpha generation but exposes the account to significant idiosyncratic risk. By focusing on a small number of high-conviction assets, the portfolio is highly sensitive to the volatility of specific sectors.

    CURRENT HOLDINGS ANALYSIS
    """
    for h in holdings_list:
        weight = (h['Value'] / total_value) * 100
        narrative += f"\n- {h['Ticker']}: This position represents {weight:.2f}% of the total AUM. At a current market price of ${h['Price']:.2f}, this asset acts as a {'core' if weight > 20 else 'satellite'} holding. Its contribution to the overall portfolio volatility is proportional to its weight and the underlying asset's beta."

    narrative += f"""

    RISK ARCHITECTURE

    The portfolio's current risk profile is classified as {risk_profile}. The lack of extensive diversification across uncorrelated asset classes implies a high correlation with the broader equity market. To optimize the Sharpe ratio, it is recommended to introduce non-correlated assets such as commodities or treasury bonds to mitigate systemic drawdown.

    LIQUIDITY & CAPITAL EFFICIENCY

    With ${balance:,.2f} in liquid reserves, the fund maintains a {'strong' if (balance/total_value) > 0.2 else 'lean'} capital buffer. This liquidity provides the necessary agility to execute opportunistic trades during market corrections without the need to liquidate existing core positions at a loss.

    STRATEGIC RECOMMENDATIONS

    Based on the current distribution, the portfolio should consider the following adjustments:
    1. Sectoral Rebalancing: Evaluate the overlap between current holdings to ensure that the portfolio is not overly exposed to a single industry.
    2. Volatility Harvesting: Utilize the liquid reserves to initiate positions in assets identified as 'undervalued' by the Researcher Agent's ML pipeline.
    3. Risk Mitigation: Implement stop-loss triggers at a 10-15% drawdown level to protect the current AUM from catastrophic tail-risk events.

    CONCLUSION

    The overall health of the portfolio is {'Stable' if asset_count >= 2 else 'Fragile'}. The primary goal for the next quarter should be the optimization of the risk-adjusted return, moving from a purely growth-oriented posture to a more sustainable institutional framework.
    """

    st.text(narrative)

    st.markdown("---")

    # 3. Active Suggestions (if pipeline ran)
    if st.session_state.pipeline_results:
        res = st.session_state.pipeline_results['manager']
        st.subheader("Active ML Recommendation")
        st.info(f"Based on the current investigation of {st.session_state.current_ticker}, the ML Manager suggests: {res['value']}")
        st.write(res['detail'])
    else:
        st.warning("No active stock investigation in progress. Execute the pipeline on the Dashboard to see a specific recommendation.")

elif page == "Asset Portfolio":
    st.title("Asset Portfolio")
    portfolio = st.session_state.portfolio
    total_value = portfolio['balance']

    if not portfolio['holdings']:
        st.write("Portfolio currently contains no active holdings.")
    else:
        holdings_data = []
        for ticker, shares in portfolio['holdings'].items():
            current_price = yf.Ticker(ticker).info.get("currentPrice", 0)
            value = shares * current_price
            total_value += value
            holdings_data.append({"Ticker": ticker, "Shares": shares, "Price": current_price, "Value": value})
        st.table(pd.DataFrame(holdings_data))

    st.metric("Total AUM", f"${total_value:,.2f}")
    st.metric("Cash Liquidity", f"${portfolio['balance']:,.2f}")

    st.markdown("---")
    st.subheader("Trade Execution")
    with st.form("trade_form"):
        t_ticker = st.text_input("Ticker").upper()
        t_shares = st.number_input("Shares", min_value=1, step=1)
        trade_type = st.selectbox("Direction", ["Buy", "Sell"])
        submit_trade = st.form_submit_button("Execute Order")

        if submit_trade and t_ticker:
            price = yf.Ticker(t_ticker).info.get("currentPrice", 0)
            if price == 0:
                st.error("Asset not found.")
            else:
                cost = price * t_shares
                if trade_type == "Buy":
                    if portfolio['balance'] >= cost:
                        portfolio['balance'] -= cost
                        portfolio['holdings'][t_ticker] = portfolio['holdings'].get(t_ticker, 0) + t_shares
                        st.success(f"Order Filled: Bought {t_shares} shares of {t_ticker}")
                    else:
                        st.error("Insufficient Liquidity")
                else:
                    if portfolio['holdings'].get(t_ticker, 0) >= t_shares:
                        portfolio['balance'] += cost
                        portfolio['holdings'][t_ticker] -= t_shares
                        if portfolio['holdings'][t_ticker] == 0:
                            del portfolio['holdings'][t_ticker]
                        st.success(f"Order Filled: Sold {t_shares} shares of {t_ticker}")
                    else:
                        st.error("Insufficient Asset Position")
                db.save_portfolio(st.session_state.user_id, portfolio['balance'], portfolio['holdings'])
                st.session_state.portfolio = portfolio
                st.rerun()
