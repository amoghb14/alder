"""ALDER — multi-agent quantitative research and paper-trading terminal."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics
import database as db
import market_data as md
import talent
import ui
from agents import ManagerAgent, ResearcherAgent, RiskAgent, StrategistAgent
from models import AlderMLModels

st.set_page_config(
    page_title="ALDER | Quantitative Research Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    db.init_db()
    db.seed_test_account()
except db.DatabaseError as exc:
    st.error("Database unavailable")
    st.write(str(exc))
    st.stop()

ui.inject_theme()

SIGNAL_TONE = {"BUY": "up", "SELL": "down", "HOLD": "warn"}


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
def auth_screen() -> None:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(
            f"""
            <div style="padding-top:3.2rem">
              <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:1.6rem">
                <div class="brand-mark" style="width:38px;height:38px;font-size:16px">A</div>
                <div style="font-size:13px;letter-spacing:.22em;font-weight:600;color:#A5B4FC">ALDER</div>
              </div>
              <div style="font-size:2.35rem;font-weight:700;line-height:1.15;letter-spacing:-.03em">
                Research desk for<br>systematic equity strategies.
              </div>
              <div style="color:#8B98B0;font-size:14.5px;line-height:1.6;margin-top:1rem;max-width:30rem">
                Four cooperating agents — research, risk, strategy and allocation — run a
                local ML pipeline over live market data, evaluate it on a walk-forward
                backtest, and execute into a simulated portfolio.
              </div>
              <div style="display:flex;gap:.5rem;margin-top:1.5rem;flex-wrap:wrap">
                {ui.pill('7,400+ listed tickers', 'accent')}
                {ui.pill('3 local models', 'neutral')}
                {ui.pill('Walk-forward backtest', 'neutral')}
                {ui.pill('Talent radar', 'neutral')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            mode = st.radio("Access", ["Sign in", "Create account"], horizontal=True, label_visibility="collapsed")
            if mode == "Sign in":
                st.markdown("##### Welcome back")
                with st.form("login"):
                    username = st.text_input("Username", placeholder="amogh")
                    password = st.text_input("Password", type="password", placeholder="••••")
                    submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
                if submitted:
                    user = db.authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user["id"]
                        st.session_state.username = user["username"]
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                st.caption("Demo account — `amogh` / `abc`")
            else:
                st.markdown("##### Create your account")
                with st.form("register"):
                    new_user = st.text_input("Username")
                    new_email = st.text_input("Email")
                    new_password = st.text_input("Password", type="password")
                    register = st.form_submit_button("Create account", type="primary", width="stretch")
                if register:
                    ok, message = db.create_user(new_user, new_email, new_password)
                    (st.success if ok else st.error)(message)
                    if ok:
                        st.caption("You can sign in now.")


for key, default in (
    ("authenticated", False),
    ("user_id", None),
    ("username", None),
    ("pipeline", None),
    ("scan", None),
):
    st.session_state.setdefault(key, default)

if not st.session_state.authenticated:
    auth_screen()
    st.stop()

st.session_state.setdefault("portfolio", db.ensure_portfolio(st.session_state.user_id))
portfolio = st.session_state.portfolio


# --------------------------------------------------------------------------- #
# Shared data helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=900)
def cached_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    return md.history(symbol, period=period)


@st.cache_data(show_spinner=False, ttl=60)
def cached_quote(symbol: str) -> dict:
    return md.quote(symbol)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_backtest(symbol: str, cost_bps: float, threshold: float) -> dict:
    return analytics.backtest(symbol, period="2y", cost_bps=cost_bps, threshold=threshold)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_models(symbol: str) -> dict:
    return analytics.evaluate_models(symbol)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_talent(pages: int) -> dict:
    return talent.scan(pages=pages)


def price_chart(symbol: str, period: str = "6mo", candles: bool = True) -> None:
    frame = cached_history(symbol, period)
    if candles:
        figure = go.Figure(
            go.Candlestick(
                x=frame.index, open=frame["Open"], high=frame["High"],
                low=frame["Low"], close=frame["Close"], name=symbol,
                increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN,
            )
        )
        figure.update_layout(xaxis_rangeslider_visible=False)
    else:
        figure = go.Figure(
            go.Scatter(
                x=frame.index, y=frame["Close"], mode="lines", name=symbol,
                line=dict(color=ui.ACCENT, width=2),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
            )
        )
    frame["MA20"] = frame["Close"].rolling(20).mean()
    frame["MA50"] = frame["Close"].rolling(50).mean()
    for column, colour in (("MA20", "#38BDF8"), ("MA50", "#F59E0B")):
        figure.add_trace(
            go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column,
                       line=dict(color=colour, width=1.3))
        )
    ui.chart(figure, height=380)


def run_pipeline(symbol: str) -> dict:
    engine = AlderMLModels(symbol)
    quote = cached_quote(symbol)
    researcher = ResearcherAgent(engine).act()
    risk = RiskAgent(engine).act()
    strategist = StrategistAgent(engine).act(researcher["value"], quote["price"], risk["value"])
    manager = ManagerAgent(engine).act(strategist["value"], risk["value"])
    return {
        "symbol": symbol.upper(),
        "name": quote["name"],
        "price": quote["price"],
        "change_pct": quote["change_pct"],
        "researcher": researcher,
        "risk": risk,
        "strategist": strategist,
        "manager": manager,
    }


def money(value: float, currency: str = "$") -> str:
    return f"{currency}{value:,.2f}"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard() -> None:
    ui.page_header(
        "Command dashboard",
        "Run the four-agent pipeline on any listed security.",
        meta=[f"Signed in as {st.session_state.username}", "Data: Yahoo Finance (delayed)"],
    )

    with st.container(border=True):
        col_pick, col_button = st.columns([3, 1], vertical_alignment="bottom")
        with col_pick:
            symbol = ui.ticker_picker(
                "Security", "dash", default=st.session_state.get("current_symbol", "NVDA"),
                help_text="Search 7,400+ listed stocks by ticker or company name.",
            )
        with col_button:
            run = st.button("Run pipeline", type="primary", width="stretch")

    if run and symbol:
        try:
            with st.spinner(f"Fitting models on {symbol}…"):
                st.session_state.pipeline = run_pipeline(symbol)
            st.session_state.current_symbol = symbol.upper()
        except Exception as exc:
            st.error(f"Pipeline failed for {symbol}: {exc}")

    result = st.session_state.pipeline
    if not result:
        ui.empty_state("No analysis yet", "Pick a security above and run the pipeline to see signals, allocation and market analytics.")
        return

    signal = result["strategist"]["value"]
    change = result["change_pct"]
    right = ui.pill(signal, SIGNAL_TONE.get(signal, "neutral")) + " " + ui.pill(
        f"{change:+.2f}% today", ui.tone_for(change)
    )
    ui.company_header(result["symbol"], result["name"], right_html=right)
    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

    ui.kpis([
        {"label": "Last price", "value": money(result["price"]),
         "delta": f"{change:+.2f}% vs prior close",
         "delta_tone": "up" if change >= 0 else "down"},
        {"label": "Predicted next close", "value": money(result["researcher"]["value"]),
         "tone": "up"},
        {"label": "Volatility score", "value": f"{result['risk']['value']:.4f}",
         "delta": "expected |daily return|", "tone": "warn"},
        {"label": "Signal", "value": signal,
         "tone": SIGNAL_TONE.get(signal, "flat")},
        {"label": "Suggested allocation", "value": result["manager"]["value"],
         "delta": "of available capital", "tone": "flat"},
    ])

    ui.section("Market analytics")
    with st.container(border=True):
        price_chart(result["symbol"], "6mo")
        st.caption(result["manager"]["detail"])

    ui.section("Agent notes")
    columns = st.columns(4)
    for column, key, title in zip(
        columns,
        ["researcher", "risk", "strategist", "manager"],
        ["Researcher", "Risk manager", "Strategist", "Portfolio manager"],
    ):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(f"`{result[key]['metric']}` → **{result[key]['value']}**")
            st.caption(result[key]["detail"])


def page_research() -> None:
    ui.page_header("Equity research", "Generate a written note backed by the price model.")
    with st.container(border=True):
        col_pick, col_button = st.columns([3, 1], vertical_alignment="bottom")
        with col_pick:
            symbol = ui.ticker_picker("Security", "research", default="MSFT")
        with col_button:
            generate = st.button("Generate note", type="primary", width="stretch")

    if not symbol:
        return

    if generate:
        try:
            with st.spinner("Synthesising research note…"):
                engine = AlderMLModels(symbol)
                st.session_state.research_note = (symbol.upper(), engine.generate_detailed_report())
        except Exception as exc:
            st.error(f"Research failed: {exc}")

    note = st.session_state.get("research_note")
    if not note:
        ui.empty_state("No note generated", "Run a note to get the model's price target, trend read and strategic outlook.")
        return

    note_symbol, note_text = note
    try:
        quote = cached_quote(note_symbol)
        frame = cached_history(note_symbol, "1y")
        returns = {
            "1M": (frame["Close"].iloc[-1] / frame["Close"].iloc[-21] - 1) * 100,
            "3M": (frame["Close"].iloc[-1] / frame["Close"].iloc[-63] - 1) * 100,
            "1Y": (frame["Close"].iloc[-1] / frame["Close"].iloc[0] - 1) * 100,
        }
        price_text = money(quote["price"])
        change_pct = quote["change_pct"]
        right = ui.pill(price_text, "accent") + " " + ui.pill(
            f"{change_pct:+.2f}%", ui.tone_for(change_pct)
        )
        ui.company_header(note_symbol, quote["name"], right_html=right)
        ui.kpis([
            {"label": f"{label} return", "value": f"{value:+.2f}%", "tone": ui.tone_for(value)}
            for label, value in returns.items()
        ])
    except Exception:
        pass

    with st.container(border=True):
        price_chart(note_symbol, "1y", candles=False)

    ui.section("Research note")
    with st.container(border=True):
        ui.prose(note_text)


def page_risk() -> None:
    ui.page_header("Risk management", "Portfolio concentration, volatility and systematic risk audit.")
    holdings = portfolio.get("holdings") or {}

    frame = analytics.holdings_frame(holdings, cached_quote)
    cash = float(portfolio["balance"])
    total = cash + float(frame["Value"].sum()) if not frame.empty else cash
    concentration = analytics.concentration(frame)

    ui.kpis([
        {"label": "Total AUM", "value": money(total)},
        {"label": "Cash buffer", "value": f"{(cash / total * 100) if total else 0:.1f}%",
         "delta": money(cash), "tone": "flat"},
        {"label": "Positions", "value": str(concentration["positions"]), "tone": "flat"},
        {"label": "Largest position", "value": f"{concentration['largest']:.1f}%",
         "delta": "of invested capital", "tone": "warn"},
        {"label": "Concentration (HHI)", "value": f"{concentration['hhi']:.2f}",
         "delta": "1.00 = single position", "tone": "warn"},
    ])

    if frame.empty:
        ui.empty_state("No positions", "Open a position in the trade desk to unlock portfolio risk analytics.")
        return

    ui.section("Position risk")
    volatility_rows = []
    returns_matrix = analytics.portfolio_volatility(list(frame["Symbol"]))
    for _, row in frame.iterrows():
        series = returns_matrix[row["Symbol"]] if returns_matrix is not None and row["Symbol"] in returns_matrix else None
        ann_vol = float(series.dropna().std() * (252 ** 0.5) * 100) if series is not None and series.dropna().size > 5 else 0.0
        volatility_rows.append({
            "Symbol": row["Symbol"],
            "Weight %": row["Weight"],
            "Value": row["Value"],
            "Annualised vol %": round(ann_vol, 1),
            "Risk contribution": round(ann_vol * row["Weight"] / 100, 1),
        })
    risk_table = pd.DataFrame(volatility_rows).sort_values("Risk contribution", ascending=False)
    with st.container(border=True):
        st.dataframe(
            risk_table, width="stretch", hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn(format="$%.2f"),
                "Weight %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                "Annualised vol %": st.column_config.NumberColumn(format="%.1f%%"),
                "Risk contribution": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    if returns_matrix is not None and returns_matrix.shape[1] > 1:
        ui.section("Correlation matrix")
        with st.container(border=True):
            corr = returns_matrix.corr().round(2)
            heatmap = go.Figure(
                go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index, colorscale="Blues",
                    zmin=-1, zmax=1, text=corr.values, texttemplate="%{text}",
                    colorbar=dict(thickness=10, outlinewidth=0),
                )
            )
            ui.chart(heatmap, height=320)

    ui.section("Full risk audit")
    if st.button("Run multi-asset risk audit"):
        with st.spinner("Scoring every position with the risk model…"):
            try:
                engine = AlderMLModels("SPY")
                st.session_state.risk_audit = engine.assess_portfolio_risk(holdings)
            except Exception as exc:
                st.error(f"Risk audit failed: {exc}")
    if st.session_state.get("risk_audit"):
        with st.container(border=True):
            ui.prose(st.session_state.risk_audit)


def page_strategy() -> None:
    ui.page_header("Strategy signals", "Cross-sectional scan for expected growth leaders.")
    default_watchlist = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, LLY, TSLA, JPM"
    watchlist_text = st.text_area(
        "Watchlist (comma separated tickers)", value=st.session_state.get("watchlist", default_watchlist),
        height=70, help="Any listed ticker works — the universe covers 7,400+ stocks.",
    )
    watchlist = [t.strip().upper() for t in watchlist_text.replace("\n", ",").split(",") if t.strip()]

    if st.button(f"Scan {len(watchlist)} securities", type="primary"):
        st.session_state.watchlist = watchlist_text
        progress = st.progress(0.0, text="Fitting price model per security…")
        rows = []
        for index, ticker in enumerate(watchlist, start=1):
            try:
                engine = AlderMLModels(ticker)
                last_close = float(engine.data["Close"].iloc[-1])
                predicted = float(engine.train_researcher())
                rows.append({
                    "Symbol": ticker,
                    "Name": md.company_name(ticker),
                    "Last": last_close,
                    "Predicted": predicted,
                    "Expected %": (predicted - last_close) / last_close * 100,
                })
            except Exception:
                continue
            progress.progress(index / max(1, len(watchlist)), text=f"Scored {ticker}")
        progress.empty()
        st.session_state.scan = pd.DataFrame(rows).sort_values("Expected %", ascending=False) if rows else pd.DataFrame()

    scan = st.session_state.get("scan")
    if scan is None:
        ui.empty_state("No scan yet", "Run a scan to rank the watchlist by the price model's expected move.")
        return
    if scan.empty:
        st.warning("No securities could be scored — check the tickers.")
        return

    best = scan.iloc[0]
    ui.kpis([
        {"label": "Securities scored", "value": str(len(scan))},
        {"label": "Top idea", "value": str(best["Symbol"]), "delta": str(best["Name"]), "tone": "up"},
        {"label": "Expected move", "value": f"{best['Expected %']:+.2f}%", "tone": "up"},
        {"label": "Median expected", "value": f"{scan['Expected %'].median():+.2f}%", "tone": "flat"},
    ])

    ui.section("Ranked by expected move")
    with st.container(border=True):
        st.dataframe(
            scan, width="stretch", hide_index=True,
            column_config={
                "Last": st.column_config.NumberColumn(format="$%.2f"),
                "Predicted": st.column_config.NumberColumn(format="$%.2f"),
                "Expected %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.caption("Expected move is the RandomForest price target against the last close — a research input, not a return forecast.")

    if len(scan) > 1:
        with st.container(border=True):
            bar = go.Figure(go.Bar(
                x=scan["Symbol"], y=scan["Expected %"],
                marker_color=[ui.UP if v >= 0 else ui.DOWN for v in scan["Expected %"]],
            ))
            ui.chart(bar, height=300)


def page_portfolio() -> None:
    ui.page_header("Portfolio", "Allocation, exposure and performance of the simulated book.")
    holdings = portfolio.get("holdings") or {}
    frame = analytics.holdings_frame(holdings, cached_quote)
    cash = float(portfolio["balance"])
    invested = float(frame["Value"].sum()) if not frame.empty else 0.0
    total = cash + invested

    ui.kpis([
        {"label": "Total AUM", "value": money(total)},
        {"label": "Invested", "value": money(invested), "delta": f"{(invested / total * 100) if total else 0:.1f}% of AUM", "tone": "flat"},
        {"label": "Cash", "value": money(cash), "tone": "flat"},
        {"label": "Positions", "value": str(len(frame)), "tone": "flat"},
    ])

    if frame.empty:
        ui.empty_state("Portfolio is empty", "Head to the trade desk to open your first position — you start with $100,000 of simulated cash.")
        return

    left, right = st.columns([1.4, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("**Holdings**")
        st.dataframe(
            frame, width="stretch", hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Value": st.column_config.NumberColumn(format="$%.2f"),
                "Weight": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            },
        )
    with right, st.container(border=True):
        st.markdown("**Allocation**")
        donut = go.Figure(
            go.Pie(labels=frame["Symbol"], values=frame["Value"], hole=0.62,
                   marker=dict(colors=["#6366F1", "#0EA5E9", "#14B8A6", "#F59E0B", "#EC4899", "#8B5CF6", "#22C55E"]))
        )
        donut.update_traces(textinfo="percent", textfont_size=11)
        ui.chart(donut, height=300)

    ui.section("Portfolio narrative")
    with st.container(border=True):
        ui.prose(build_narrative(frame, cash, total))

    if st.session_state.pipeline:
        with st.container(border=True):
            st.markdown("**Active recommendation**")
            st.info(f"{st.session_state.pipeline['symbol']}: {st.session_state.pipeline['manager']['value']} — {st.session_state.pipeline['manager']['detail']}")


def build_narrative(frame: pd.DataFrame, cash: float, total: float) -> str:
    if frame.empty or total <= 0:
        return "No positions held."
    concentration = analytics.concentration(frame)
    top = frame.iloc[0]
    lines = [
        "POSITIONING",
        f"Assets under management total ${total:,.2f} with {cash / total * 100:.1f}% held in cash across "
        f"{len(frame)} position(s). The largest holding is {top['Symbol']} at {top['Weight']:.1f}% of invested capital.",
        "RISK ARCHITECTURE",
        f"Concentration (HHI) is {concentration['hhi']:.2f}; values above 0.25 indicate a book dominated by a "
        f"handful of names, where idiosyncratic risk outweighs sector diversification.",
        "CAPITAL EFFICIENCY",
        f"Deployable cash of ${cash:,.2f} represents {(cash / total * 100):.1f}% of the book — "
        f"{'a strong buffer for opportunistic entries' if cash / total > 0.2 else 'a lean reserve that leaves little room for drawdown buying'}.",
        "NEXT ACTIONS",
        "- Rebalance toward a target weight band if any position drifts past 25% of AUM.",
        "- Review correlation between the top two holdings before adding exposure.",
        "- Re-run the pipeline and the strategy lab after the next earnings window.",
    ]
    return "\n".join(lines)


def page_trade() -> None:
    ui.page_header("Trade desk", "Simulated execution at the latest quoted price.")
    left, right = st.columns([1, 1.2], gap="large")

    with left, st.container(border=True):
        st.markdown("**New order**")
        symbol = ui.ticker_picker("Security", "trade", default="AAPL")
        direction = st.radio("Direction", ["Buy", "Sell"], horizontal=True)
        shares = st.number_input("Shares", min_value=1, step=1, value=1)

        price = None
        if symbol:
            try:
                price = cached_quote(symbol)["price"]
            except Exception:
                price = None

        if price:
            st.caption(f"Estimated notional: **{money(price * shares)}** at {money(price)}")
        else:
            st.caption("Enter a valid symbol to price the order.")

        if st.button("Execute order", type="primary", width="stretch"):
            if not symbol or price is None:
                st.error("Unknown or untradeable symbol.")
            else:
                cost = price * shares
                book = st.session_state.portfolio
                if direction == "Buy":
                    if book["balance"] >= cost:
                        book["balance"] -= cost
                        book["holdings"][symbol] = book["holdings"].get(symbol, 0) + shares
                        db.save_portfolio(st.session_state.user_id, book["balance"], book["holdings"])
                        st.success(f"Filled: bought {shares} {symbol} at {money(price)}")
                        st.rerun()
                    else:
                        st.error(f"Insufficient cash — need {money(cost)}, have {money(book['balance'])}.")
                else:
                    owned = book["holdings"].get(symbol, 0)
                    if owned >= shares:
                        book["balance"] += cost
                        book["holdings"][symbol] = owned - shares
                        if book["holdings"][symbol] == 0:
                            del book["holdings"][symbol]
                        db.save_portfolio(st.session_state.user_id, book["balance"], book["holdings"])
                        st.success(f"Filled: sold {shares} {symbol} at {money(price)}")
                        st.rerun()
                    else:
                        st.error(f"Only {owned} share(s) of {symbol} held.")

    with right, st.container(border=True):
        st.markdown("**Open positions**")
        frame = analytics.holdings_frame(st.session_state.portfolio.get("holdings") or {}, cached_quote)
        if frame.empty:
            ui.empty_state("No positions", "Your fills will appear here.")
        else:
            st.dataframe(
                frame, width="stretch", hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Value": st.column_config.NumberColumn(format="$%.2f"),
                    "Weight": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                },
            )
            st.caption(f"Cash: {money(st.session_state.portfolio['balance'])}")


def page_lab() -> None:
    ui.page_header(
        "Strategy lab",
        "Walk-forward backtest of the signal with transaction costs, versus buy & hold.",
        meta=["Expanding-window refit", "No future data", f"Cost charged on every change"],
    )

    with st.container(border=True):
        col_symbol, col_cost, col_threshold, col_run = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
        with col_symbol:
            symbol = ui.ticker_picker("Security", "lab", default="MSFT")
        with col_cost:
            cost_bps = st.number_input("Cost (bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
        with col_threshold:
            threshold = st.slider("Probability threshold", 0.50, 0.75, 0.55, 0.01)
        with col_run:
            run = st.button("Run backtest", type="primary", width="stretch")

    if run and symbol:
        try:
            with st.spinner("Replaying history out-of-sample…"):
                st.session_state.backtest = cached_backtest(symbol.upper(), float(cost_bps), float(threshold))
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")

    result = st.session_state.get("backtest")
    if not result:
        ui.empty_state("No backtest yet", "Run the engine to see the out-of-sample equity curve, Sharpe and drawdown.")
        return

    metrics = result["metrics"]
    strategy, benchmark = metrics["strategy"], metrics["benchmark"]
    badges = " ".join([
        ui.pill(f"{metrics['days']} trading days", "neutral"),
        ui.pill(f"{metrics['trades']} trades", "neutral"),
        ui.pill(f"{metrics['cost_bps']:.0f} bps cost", "neutral"),
    ])
    ui.company_header(result["symbol"], md.company_name(result["symbol"]), right_html=badges)

    ui.kpis([
        {"label": "Strategy return", "value": f"{strategy['total_return'] * 100:+.1f}%",
         "tone": ui.tone_for(strategy["total_return"]),
         "delta": f"{metrics['excess_return']:+.1f}% vs buy & hold"},
        {"label": "Buy & hold", "value": f"{benchmark['total_return'] * 100:+.1f}%",
         "tone": ui.tone_for(benchmark["total_return"])},
        {"label": "Sharpe", "value": f"{strategy['sharpe']:.2f}", "tone": "flat",
         "delta": f"benchmark {benchmark['sharpe']:.2f}"},
        {"label": "Max drawdown", "value": f"{strategy['max_drawdown'] * 100:.1f}%", "tone": "warn",
         "delta": f"benchmark {benchmark['max_drawdown'] * 100:.1f}%"},
        {"label": "Hit rate", "value": f"{metrics['hit_rate']:.1f}%", "tone": "flat",
         "delta": f"{metrics['exposure']:.0f}% time in market"},
    ])

    ui.section("Equity curve")
    with st.container(border=True):
        curve = result["curve"]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=curve.index, y=curve["Strategy"], name="Strategy",
                                    line=dict(color=ui.ACCENT, width=2.2)))
        figure.add_trace(go.Scatter(x=curve.index, y=curve["Buy & Hold"], name="Buy & hold",
                                    line=dict(color="#64748B", width=1.6, dash="dot")))
        ui.chart(figure, height=380)

    ui.section("Metrics")
    with st.container(border=True):
        rows = []
        for label, key, is_pct in (
            ("Total return", "total_return", True),
            ("CAGR", "cagr", True),
            ("Volatility", "volatility", True),
            ("Sharpe", "sharpe", False),
            ("Sortino", "sortino", False),
            ("Max drawdown", "max_drawdown", True),
            ("Calmar", "calmar", False),
        ):
            rows.append({
                "Metric": label,
                "Strategy": f"{strategy[key] * 100:.2f}%" if is_pct else f"{strategy[key]:.2f}",
                "Buy & hold": f"{benchmark[key] * 100:.2f}%" if is_pct else f"{benchmark[key]:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    ui.section("Model scores (chronological hold-out)")
    try:
        scores = cached_models(result["symbol"])
        ui.kpis([
            {"label": "Price MAE", "value": f"${scores['researcher']['mae']:.2f}",
             "delta": f"RMSE ${scores['researcher']['rmse']:.2f}", "tone": "flat"},
            {"label": "Risk R²", "value": f"{scores['risk']['r2']:.2f}", "tone": "flat"},
            {"label": "Direction accuracy", "value": f"{scores['strategist']['accuracy']:.1f}%",
             "delta": f"F1 {scores['strategist']['f1']:.1f}%", "tone": "warn"},
            {"label": "Train / test rows", "value": f"{scores['rows']}",
             "delta": f"{scores['test_rows']} held out", "tone": "flat"},
        ])
        st.caption(
            "A ~50% directional accuracy is a coin flip: the pipeline is a research instrument, "
            "not a tradeable edge. The backtest above is the honest test of whether it survives costs."
        )
    except Exception as exc:
        st.caption(f"Model scoring unavailable: {exc}")


def page_talent() -> None:
    ui.page_header(
        "Talent radar",
        "Scan live software-engineering internship postings and rank the skills they demand.",
        meta=["Sources: Arbeitnow + Remotive (keyless APIs)"],
    )

    with st.container(border=True):
        col_pages, col_button = st.columns([3, 1], vertical_alignment="bottom")
        with col_pages:
            pages = st.slider("Pages to scan", 2, 20, 10,
                              help="Each Arbeitnow page holds up to 300 live postings.")
        with col_button:
            scan_now = st.button("Scan postings", type="primary", width="stretch")

    if scan_now:
        with st.spinner("Fetching and mining postings…"):
            st.session_state.talent = cached_talent(int(pages))

    data = st.session_state.get("talent")
    if not data:
        ui.empty_state("No scan yet", "Scan the boards to build a demand profile of the skills SWE internships ask for.")
        return

    postings = data["postings"]
    skills = data["skills"]
    if postings is None or postings.empty or skills.empty:
        st.warning("No postings matched. Try scanning more pages.")
        return

    top_skill = skills.iloc[0]
    ui.kpis([
        {"label": "Postings analysed", "value": str(len(postings)), "delta": f"fetched {data['fetched_at']}"},
        {"label": "Internships", "value": str(int(postings["is_internship"].sum())) if "is_internship" in postings else "—", "tone": "flat"},
        {"label": "Skills detected", "value": str(len(skills)), "tone": "flat"},
        {"label": "Most demanded", "value": top_skill["skill"],
         "delta": f"{top_skill['share']:.0f}% of postings", "tone": "up"},
    ])
    st.caption(
        "Live sources: " + ", ".join(data.get("sources") or ["none"]) +
        " · skills are mined from posting text against a 46-term taxonomy"
    )

    left, right = st.columns([1.15, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("**Skill demand**")
        top = skills.head(14).iloc[::-1]
        bar = go.Figure(go.Bar(
            x=top["share"], y=top["skill"], orientation="h",
            marker_color=ui.ACCENT, text=[f"{v:.0f}%" for v in top["share"]],
            textposition="outside",
        ))
        ui.chart(bar, height=420)

    with right, st.container(border=True):
        st.markdown("**Your profile vs. demand**")
        mine = st.text_area(
            "Your skills (comma separated)",
            value=st.session_state.get("my_skills", "Python, SQL, Pandas, scikit-learn, Git, Docker, AWS"),
            height=90,
        )
        st.session_state.my_skills = mine
        gap = talent.gap_analysis(skills, [s for s in mine.split(",")])
        st.metric("Coverage of top skills", f"{gap['coverage']:.0f}%")
        if gap["missing"]:
            st.markdown("Missing: " + " ".join(ui.pill(s, "warn") for s in gap["missing"][:6]), unsafe_allow_html=True)
        if not gap["rows"].empty:
            st.dataframe(gap["rows"], width="stretch", hide_index=True, height=250)

    ui.section("Postings")
    with st.container(border=True):
        display = postings.copy()
        if "skills" in display:
            display["skills"] = display["skills"].apply(lambda s: ", ".join(s[:5]) if isinstance(s, list) else s)
        columns = [c for c in ["title", "company", "location", "skills", "source", "url"] if c in display]
        st.dataframe(
            display[columns], width="stretch", hide_index=True, height=320,
            column_config={
                "title": st.column_config.TextColumn("Role", width="medium"),
                "company": st.column_config.TextColumn("Company", width="small"),
                "location": st.column_config.TextColumn("Location", width="small"),
                "skills": st.column_config.TextColumn("Skills", width="medium"),
                "source": st.column_config.TextColumn("Source", width="small"),
                "url": st.column_config.LinkColumn("Posting", display_text="Open"),
            },
        )


PAGES = {
    "Dashboard": page_dashboard,
    "Research": page_research,
    "Risk": page_risk,
    "Strategy": page_strategy,
    "Portfolio": page_portfolio,
    "Trade Desk": page_trade,
    "Strategy Lab": page_lab,
    "Talent Radar": page_talent,
}

page = ui.sidebar("Research terminal", st.session_state.username, "Authorized session")
if st.sidebar.button("Sign out", width="stretch"):
    for key in ("authenticated", "user_id", "username", "portfolio", "pipeline", "scan", "backtest"):
        st.session_state.pop(key, None)
    st.rerun()

PAGES.get(page, page_dashboard)()
