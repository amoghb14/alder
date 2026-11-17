"""ALDER design system.

One place for the visual language: a dark institutional palette, Inter
typography, card surfaces, navigation, metric tiles and the company avatar with
its monogram fallback. Keeping this out of ``app.py`` keeps page code readable
and makes the look consistent across every view.
"""

from __future__ import annotations

import html
import re
from typing import Dict, Iterable, List, Optional

import plotly.graph_objects as go
import streamlit as st

import market_data

ACCENT = "#6366F1"
UP = "#10B981"
DOWN = "#F43F5E"
WARN = "#F59E0B"
MUTED = "#8B98B0"
GRID = "rgba(148,163,184,0.12)"

PAGES: List[tuple] = [
    ("Dashboard", "Command"),
    ("Research", "Command"),
    ("Risk", "Command"),
    ("Strategy", "Command"),
    ("Portfolio", "Capital"),
    ("Trade Desk", "Capital"),
    ("Strategy Lab", "Research"),
    ("Talent Radar", "Research"),
]


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

        :root {
            --bg: #0A0E17;
            --surface: #111827;
            --surface-2: #151D2E;
            --border: #1F2A44;
            --text: #E8EDF7;
            --muted: #8B98B0;
            --accent: #6366F1;
            --up: #10B981;
            --down: #F43F5E;
        }

        html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
        .stApp { background: radial-gradient(1100px 600px at 15% -10%, #131B2E 0%, #0A0E17 55%) fixed; color: var(--text); }
        #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
        .block-container { padding: 2.2rem 2.6rem 4rem 2.6rem; max-width: 1500px; }

        /* ---------- sidebar ---------- */
        [data-testid="stSidebar"] {
            background: #0C121F; border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
        [data-testid="stSidebar"] .block-container { padding: 0; }
        .brand { display:flex; align-items:center; gap:.7rem; padding: 0 .2rem 1.1rem .2rem; }
        .brand-mark {
            width: 34px; height: 34px; border-radius: 10px; display:grid; place-items:center;
            background: linear-gradient(135deg, #6366F1, #0EA5E9); color:#fff; font-weight:700; font-size:15px;
            letter-spacing:.5px; box-shadow: 0 6px 18px rgba(99,102,241,.35);
        }
        .brand-text { line-height:1.05; }
        .brand-text b { font-size: 15px; letter-spacing:.14em; }
        .brand-text span { display:block; font-size: 10.5px; color: var(--muted); letter-spacing:.12em; text-transform: uppercase; margin-top:3px;}
        .nav-group { font-size: 10px; letter-spacing: .18em; text-transform: uppercase; color: #5A6683; margin: 1.1rem 0 .35rem .3rem; }

        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stButton > button:focus:not(:active) {
            background: transparent; border: 1px solid transparent; color: #B9C4D8;
            text-align: left; justify-content: flex-start; border-radius: 9px;
            font-weight: 500; font-size: 13.5px; padding: .5rem .7rem; box-shadow: none;
            width: 100%; transition: all .15s ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(99,102,241,.12); color: #E8EDF7; border-color: rgba(99,102,241,.35);
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(99,102,241,.18); color: #FFFFFF; border-color: rgba(99,102,241,.55);
            font-weight: 600;
        }
        .sidebar-user {
            margin-top: 1.2rem; padding: .7rem .8rem; border: 1px solid var(--border);
            border-radius: 12px; background: #0F1626; display:flex; gap:.6rem; align-items:center;
        }
        .avatar {
            width: 30px; height: 30px; border-radius: 50%; background: rgba(99,102,241,.25);
            color: #C7CCFF; display:grid; place-items:center; font-weight: 600; font-size: 12px;
        }
        .sidebar-user small { color: var(--muted); display:block; font-size: 10.5px; letter-spacing:.06em; }

        /* ---------- typography ---------- */
        h1, h2, h3, h4 { color: var(--text) !important; font-family:'Inter',sans-serif !important; letter-spacing:-.02em; }
        .page-title { font-size: 1.72rem; font-weight: 700; margin: 0; letter-spacing:-.03em; }
        .page-sub { color: var(--muted); font-size: 13.5px; margin-top: .35rem; }
        .page-head { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
            padding-bottom: 1.1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
        .page-meta { color: var(--muted); font-size: 12px; text-align:right; }
        .section-label { font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: #6B7894; margin: 1.6rem 0 .6rem 0; font-weight:600;}

        /* ---------- cards ---------- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(20,28,45,.9), rgba(15,21,35,.9));
            border: 1px solid var(--border) !important; border-radius: 16px !important;
        }
        .kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
        .kpi {
            background: linear-gradient(180deg, #141C2D, #101728); border: 1px solid var(--border);
            border-radius: 14px; padding: 14px 16px; position: relative; overflow: hidden;
        }
        .kpi:before { content:''; position:absolute; inset:0 auto 0 0; width:3px; background: var(--accent); opacity:.9; }
        .kpi.up:before { background: var(--up); } .kpi.down:before { background: var(--down); }
        .kpi.warn:before { background: var(--warn); } .kpi.flat:before { background: #3B4A68; }
        .kpi-label { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
        .kpi-value { font-family:'JetBrains Mono', ui-monospace, monospace; font-size: 1.5rem; font-weight: 600; margin-top: .45rem; line-height:1.1; }
        .kpi-delta { font-size: 12px; margin-top: .35rem; color: var(--muted); }
        .kpi-delta.up { color: var(--up); } .kpi-delta.down { color: var(--down); }

        .tile { position: relative; width: 46px; height: 46px; border-radius: 12px; overflow: hidden; flex: 0 0 auto;
            display:grid; place-items:center; font-weight: 700; font-size: 14px; letter-spacing:.02em; }
        .tile img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#F7F9FC; padding:7px; }
        .company { display:flex; align-items:center; gap: .8rem; }
        .company-name { font-size: 1.05rem; font-weight: 650; letter-spacing:-.01em; }
        .company-symbol { color: var(--muted); font-size: 12px; letter-spacing:.08em; font-family:'JetBrains Mono',monospace;}

        .pill { display:inline-flex; align-items:center; gap:.35rem; padding: 3px 9px; border-radius: 999px;
            font-size: 11px; font-weight: 600; letter-spacing:.04em; border:1px solid transparent; }
        .pill.neutral { background: rgba(148,163,184,.12); color:#C3CBDD; border-color: rgba(148,163,184,.25); }
        .pill.up { background: rgba(16,185,129,.12); color:#34D399; border-color: rgba(16,185,129,.3); }
        .pill.down { background: rgba(244,63,94,.12); color:#FB7185; border-color: rgba(244,63,94,.3); }
        .pill.warn { background: rgba(245,158,11,.12); color:#FBBF24; border-color: rgba(245,158,11,.3); }
        .pill.accent { background: rgba(99,102,241,.16); color:#A5B4FC; border-color: rgba(99,102,241,.35); }

        .empty { border: 1px dashed var(--border); border-radius: 16px; padding: 2.4rem 1.5rem; text-align:center; background: #0E1522; }
        .empty b { display:block; font-size: 15px; margin-bottom:.3rem; }
        .empty span { color: var(--muted); font-size: 13px; }

        .prose { color:#C9D3E4; font-size: 13.8px; line-height: 1.65; }
        .prose h4 { margin: 1.1rem 0 .35rem 0; font-size: 13px; letter-spacing:.1em; text-transform:uppercase; color:#A5B4FC; }
        .prose p { margin: .25rem 0 .6rem 0; }
        .prose li { margin-left: 1.1rem; }

        /* ---------- streamlit controls ---------- */
        .stButton > button {
            border-radius: 10px; font-weight: 600; font-size: 13px; padding: .5rem 1rem;
            border: 1px solid var(--border); background: #16203300; color: #D5DDEC;
        }
        .stButton > button[kind="primary"],
        [data-testid="stBaseButton-primaryFormSubmit"],
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #6366F1, #4F46E5) !important; color:#fff !important;
            border:none !important; box-shadow: 0 6px 18px rgba(79,70,229,.28);
        }
        [data-testid="stBaseButton-primaryFormSubmit"]:hover,
        [data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }
        .stButton > button:hover { border-color: rgba(99,102,241,.5); color: #fff; }
        /* text + number inputs (react-aria in current Streamlit, baseweb in older) */
        .react-aria-TextField > div, .react-aria-ComboBox > div, .react-aria-Select > div,
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background: #0F1626 !important; border: 1px solid var(--border) !important;
            border-radius: 10px !important; box-shadow: none !important;
        }
        .react-aria-TextField > div:focus-within, div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
        }
        .react-aria-TextField input, div[data-baseweb="input"] input, .stTextArea textarea {
            background: transparent !important; color: var(--text) !important; border: none !important;
            font-size: 13.5px !important;
        }
        .react-aria-TextField input::placeholder, .stTextArea textarea::placeholder { color: #62708C !important; }
        label, .stSelectbox label, .stTextInput label, .stNumberInput label { color: #9AA7BE !important; font-size: 12px !important; font-weight: 500 !important; }
        [data-testid="stForm"] { border: none !important; background: transparent !important; padding: 0 !important; }
        [data-testid="stMetricValue"] { font-family:'JetBrains Mono', monospace; }
        /* segmented control: hide the native radio dot, style the chip instead */
        div[role="radiogroup"] { gap: 8px !important; }
        div[role="radiogroup"] label {
            background: #0F1626; border: 1px solid var(--border); border-radius: 10px;
            padding: .42rem .85rem; transition: all .15s ease; color: #B9C4D8 !important;
        }
        div[role="radiogroup"] label > div > div:first-child,
        [data-testid="stRadioOption"] > div > div:first-child { display: none !important; }
        div[role="radiogroup"] label:hover { border-color: rgba(99,102,241,.5); }
        div[role="radiogroup"] > div[data-selected="true"] label,
        div[role="radiogroup"] label:has(input:checked) {
            background: rgba(99,102,241,.18) !important; border-color: rgba(99,102,241,.6) !important;
            color: #FFFFFF !important;
        }
        div[role="radiogroup"] label p { font-size: 13px !important; margin: 0 !important; }
        .stDataFrame, [data-testid="stTable"] { border-radius: 12px; overflow:hidden; border:1px solid var(--border); }
        [data-testid="stForm"] { border: 1px solid var(--border); border-radius: 16px; padding: 1.1rem 1.1rem .6rem 1.1rem; background: #0F1626; }
        div[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--border); }
        hr, [data-testid="stDivider"] { border-color: var(--border) !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; color: var(--muted); font-size: 13px; }
        .stTabs [aria-selected="true"] { color: #fff; background: rgba(99,102,241,.12); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def _inject(markup: str) -> None:
    """Render an HTML block.

    Streamlit runs the string through a Markdown parser, and four-space
    indentation (or a blank line) turns part of it into a <code> block — which is
    how raw div markup leaks onto the page. Collapsing whitespace between tags
    keeps the whole block as one inline HTML fragment.
    """
    compact = re.sub(r">\s+<", "><", (markup or "").strip())
    st.markdown(compact, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", meta: Optional[Iterable[str]] = None) -> None:
    meta_html = "<br>".join(html.escape(str(m)) for m in (meta or []))
    _inject(
        f'<div class="page-head"><div><div class="page-title">{html.escape(title)}</div>'
        f'<div class="page-sub">{html.escape(subtitle)}</div></div>'
        f'<div class="page-meta">{meta_html}</div></div>'
    )


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{html.escape(label)}</div>', unsafe_allow_html=True)


def kpis(items: List[Dict[str, str]]) -> None:
    """items: [{label, value, delta?, tone?}] — tone in up/down/warn/flat."""
    cards = []
    for item in items:
        tone = item.get("tone", "")
        delta = item.get("delta", "")
        delta_tone = item.get("delta_tone", "")
        delta_html = (
            f'<div class="kpi-delta {delta_tone}">{html.escape(delta)}</div>' if delta else ""
        )
        cards.append(
            f'<div class="kpi {tone}">'
            f'<div class="kpi-label">{html.escape(item.get("label", ""))}</div>'
            f'<div class="kpi-value">{html.escape(str(item.get("value", "")))}</div>'
            f"{delta_html}</div>"
        )
    _inject(f'<div class="kpi-grid">{"".join(cards)}</div>')


def company_tile(symbol: str, name: str, size: int = 46) -> str:
    """Avatar markup: real logo when reachable, monogram underneath as fallback."""
    symbol = (symbol or "").upper()
    name = name or market_data.company_name(symbol)
    colour = market_data.monogram_colour(symbol)
    initials = html.escape(market_data.monogram_text(symbol, name))
    logo = html.escape(market_data.logo_url(symbol))
    return (
        f'<div class="tile" style="width:{size}px;height:{size}px;background:rgba(255,255,255,.06);color:{colour}">'
        f'<span>{initials}</span>'
        f'<img src="{logo}" alt="{html.escape(symbol)}" loading="lazy" onerror="this.remove()">'
        f"</div>"
    )


def company_header(symbol: str, name: str, right_html: str = "") -> None:
    symbol = (symbol or "").upper()
    _inject(
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:1rem">'
        f'<div class="company">{company_tile(symbol, name)}'
        f'<div><div class="company-name">{html.escape(name)}</div>'
        f'<div class="company-symbol">{html.escape(symbol)}</div></div></div>'
        f"<div>{right_html}</div></div>"
    )


def pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="pill {tone}">{html.escape(str(text))}</span>'


def tone_for(value: float) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def empty_state(title: str, body: str, icon: str = "◎") -> None:
    _inject(
        f'<div class="empty">'
        f'<div style="font-size:1.6rem;color:#6366F1;margin-bottom:.4rem">{icon}</div>'
        f"<b>{html.escape(title)}</b><span>{html.escape(body)}</span></div>"
    )


def prose(text: str) -> None:
    """Render pre-formatted narrative with section headings as HTML."""
    blocks = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if line == upper and len(line) < 70:
            blocks.append(f"<h4>{html.escape(line)}</h4>")
        elif line.startswith("- "):
            blocks.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            blocks.append(f"<p>{html.escape(line)}</p>")
    st.markdown(f'<div class="prose">{"".join(blocks)}</div>', unsafe_allow_html=True)


def chart(fig: go.Figure, height: int = 380) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C9D3E4", size=12),
        hoverlabel=dict(bgcolor="#111827", font_size=12, bordercolor="#1F2A44"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _inject_sidebar(markup: str) -> None:
    compact = re.sub(r">\s+<", "><", (markup or "").strip())
    st.sidebar.markdown(compact, unsafe_allow_html=True)


def sidebar(brand_subtitle: str, user: str, status: str) -> str:
    """Render brand + grouped navigation; returns the selected page."""
    _inject_sidebar(
        '<div class="brand"><div class="brand-mark">A</div>'
        f'<div class="brand-text"><b>ALDER</b><span>{html.escape(brand_subtitle)}</span></div></div>'
    )

    if "page" not in st.session_state:
        st.session_state.page = PAGES[0][0]

    current_group = None
    for page, group in PAGES:
        if group != current_group:
            _inject_sidebar(f'<div class="nav-group">{html.escape(group)}</div>')
            current_group = group
        st.sidebar.button(
            page,
            key=f"nav_{page}",
            width="stretch",
            type="primary" if st.session_state.page == page else "secondary",
            on_click=_set_page,
            args=(page,),
        )

    _inject_sidebar(
        f'<div class="sidebar-user"><div class="avatar">{html.escape((user or "?")[:2].upper())}</div>'
        f'<div><b style="font-size:12.5px">{html.escape(user or "Guest")}</b>'
        f"<small>{html.escape(status)}</small></div></div>"
    )
    return st.session_state.page


def _set_page(page: str) -> None:
    st.session_state.page = page


def ticker_picker(label: str, key: str, default: str = "NVDA", help_text: str = "") -> str:
    """Search the full listed-stock universe by ticker or company name."""
    query = st.text_input(label, value=default, key=f"{key}_q", help=help_text,
                          placeholder="Ticker or company, e.g. PLTR, palantir, NIO")
    candidates = market_data.search_symbols(query, limit=25)
    if not candidates:
        return (query or "").upper().strip()

    labels = [f"{c['symbol']} — {c['name']}" for c in candidates]
    index = 0
    upper = (query or "").upper().strip()
    for i, candidate in enumerate(candidates):
        if candidate["symbol"] == upper:
            index = i
            break
    chosen = st.selectbox("Matching securities", labels, index=index, key=f"{key}_sel",
                          label_visibility="collapsed")
    return chosen.split(" — ")[0].strip()
