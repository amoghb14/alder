"""Robust market data layer for ALDER.

Everything the app needs about a security goes through here, so the UI never
talks to a network API directly:

* a full ticker universe (every NASDAQ / NYSE listed common stock) for instant
  local symbol search,
* quotes and daily history through yfinance,
* company logos with a built-in monogram fallback.

Every network call has a timeout, bounded retries with exponential backoff, a
short-lived in-process cache and a disk cache for slow-changing reference data.
Tradeable results are cached so a UI rerun never re-hits an API.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd

try:  # requests is a hard dependency of the app, but stay importable regardless
    import requests
except ImportError:  # pragma: no cover
    requests = None

import yfinance as yf

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
SYMBOL_DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
LOGO_URL = "https://assets.parqet.com/logos/symbol/{symbol}?format=png"

UNIVERSE_TTL = 7 * 24 * 3600  # one week
QUOTE_TTL = 45  # seconds
HISTORY_TTL = 15 * 60  # fifteen minutes


class MarketDataError(RuntimeError):
    """Raised when market data cannot be retrieved."""


# --------------------------------------------------------------------------- #
# Small TTL cache
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, tuple] = {}


def _cache_get(key: str, ttl: float):
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    return None


def _cache_set(key: str, value):
    _CACHE[key] = (time.time(), value)
    return value


def clear_cache() -> None:
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# HTTP with retries
# --------------------------------------------------------------------------- #
def http_get(url: str, timeout: int = 15, retries: int = 3, backoff: float = 0.8) -> str:
    """GET a URL, retrying transient failures with exponential backoff."""
    if requests is None:  # pragma: no cover
        raise MarketDataError(
            "The 'requests' package is required for network access. "
            "Run: pip install requests"
        )

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": _UA})
            if response.status_code == 200:
                return response.text
            last_error = MarketDataError(f"HTTP {response.status_code} from {url}")
            if response.status_code not in (408, 425, 429, 500, 502, 503, 504):
                break
        except Exception as exc:  # network error / timeout
            last_error = exc
        time.sleep(backoff * (2**attempt))
    raise MarketDataError(f"Could not reach {url}: {last_error}")


# --------------------------------------------------------------------------- #
# Ticker universe
# --------------------------------------------------------------------------- #
FALLBACK_UNIVERSE = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"), ("AMZN", "Amazon"),
    ("GOOGL", "Alphabet"), ("META", "Meta Platforms"), ("TSLA", "Tesla"),
    ("BRK-B", "Berkshire Hathaway"), ("LLY", "Eli Lilly"), ("AVGO", "Broadcom"),
    ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("UNH", "UnitedHealth"),
    ("XOM", "Exxon Mobil"), ("JNJ", "Johnson & Johnson"), ("WMT", "Walmart"),
    ("MA", "Mastercard"), ("PG", "Procter & Gamble"), ("HD", "Home Depot"),
    ("CVX", "Chevron"), ("ABBV", "AbbVie"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"),
    ("COST", "Costco"), ("ADBE", "Adobe"), ("MRK", "Merck"), ("CRM", "Salesforce"),
    ("AMD", "Advanced Micro Devices"), ("NFLX", "Netflix"), ("INTC", "Intel"),
    ("PLTR", "Palantir"), ("UBER", "Uber"), ("SNOW", "Snowflake"), ("SHOP", "Shopify"),
    ("COIN", "Coinbase"), ("SPY", "SPDR S&P 500 ETF"), ("QQQ", "Invesco QQQ Trust"),
]

_NAME_SUFFIX = re.compile(
    r"\s*(?:-\s*)?(?:Class\s+[A-Z]\b.*|Common\s+Stock.*|Ordinary\s+Shares.*|"
    r"American\s+Depositary.*|Depositary\s+Shares.*|Preferred\s+Stock.*|"
    r"Fixed\s+Rate.*|Notes?\b.*|Bonds?\b.*|Warrants?\b.*|Rights?\b.*|Units?\b.*|"
    r"Shares?\s+of\s+Beneficial.*)$",
    re.IGNORECASE,
)
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")


def _clean_company_name(raw: str) -> str:
    name = (raw or "").strip()
    cleaned = _NAME_SUFFIX.sub("", name).strip(" ,-.")
    return cleaned or name or "Unknown"


def _parse_symbol_directory(text: str) -> List[tuple]:
    rows: List[tuple] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("File Creation Time"):
            continue
        fields = line.split("|")
        if len(fields) < 8 or fields[0] != "Y":
            continue
        symbol, name, etf, test_issue = fields[1].strip(), fields[2], fields[5], fields[7]
        if etf != "N" or test_issue != "N" or not _SYMBOL_RE.match(symbol):
            continue
        rows.append((symbol, _clean_company_name(name)))
    return rows


def universe(force_refresh: bool = False) -> pd.DataFrame:
    """Every listed common stock (NASDAQ / NYSE / AMEX) as ``Symbol``, ``Name``.

    Cached on disk for a week; falls back to a bundled list of large caps if the
    directory cannot be downloaded, so symbol search always works.
    """
    cached = _cache_get("universe", UNIVERSE_TTL)
    if cached is not None and not force_refresh:
        return cached

    os.makedirs(CACHE_DIR, exist_ok=True)
    disk_path = os.path.join(CACHE_DIR, "universe.csv")

    if not force_refresh and os.path.exists(disk_path):
        age = time.time() - os.path.getmtime(disk_path)
        if age < UNIVERSE_TTL:
            try:
                frame = pd.read_csv(disk_path)
                if len(frame) > len(FALLBACK_UNIVERSE):
                    return _cache_set("universe", frame)
            except Exception:
                pass

    try:
        rows = _parse_symbol_directory(http_get(SYMBOL_DIRECTORY_URL, timeout=30))
        if len(rows) < 500:
            raise MarketDataError("symbol directory looked incomplete")
        frame = pd.DataFrame(rows, columns=["Symbol", "Name"]).drop_duplicates("Symbol")
        try:
            frame.to_csv(disk_path, index=False)
        except OSError:
            pass  # read-only filesystem: keep the in-memory copy
    except Exception:
        if os.path.exists(disk_path):
            try:
                frame = pd.read_csv(disk_path)
            except Exception:
                frame = pd.DataFrame(FALLBACK_UNIVERSE, columns=["Symbol", "Name"])
        else:
            frame = pd.DataFrame(FALLBACK_UNIVERSE, columns=["Symbol", "Name"])

    frame = frame.sort_values("Symbol").reset_index(drop=True)
    return _cache_set("universe", frame)


def company_name(symbol: str) -> str:
    symbol = (symbol or "").upper().strip()
    frame = universe()
    match = frame.loc[frame["Symbol"] == symbol, "Name"]
    if len(match):
        return str(match.iloc[0])
    for tick, name in FALLBACK_UNIVERSE:
        if tick == symbol:
            return name
    return symbol


def search_symbols(query: str, limit: int = 25) -> List[Dict[str, str]]:
    """Rank symbols for a free-text query — ticker or company name."""
    query = (query or "").strip()
    frame = universe()
    if not query:
        popular = [t for t, _ in FALLBACK_UNIVERSE[:limit]]
        return [
            {"symbol": sym, "name": company_name(sym)} for sym in popular if sym
        ]

    q = query.lower()
    q_upper = query.upper()
    named = frame["Name"].str.lower()
    symbols = frame["Symbol"]

    buckets = [
        symbols == q_upper,
        symbols.str.startswith(q_upper, na=False),
        named.str.startswith(q, na=False),
        named.str.contains(rf"\b{re.escape(q)}", na=False, regex=True),
        named.str.contains(re.escape(q), na=False, regex=True),
    ]

    seen: set = set()
    results: List[Dict[str, str]] = []
    for mask in buckets:
        for symbol, name in frame.loc[mask, ["Symbol", "Name"]].itertuples(index=False):
            if symbol in seen:
                continue
            seen.add(symbol)
            results.append({"symbol": symbol, "name": name})
            if len(results) >= limit:
                return results

    # Not in the directory but a plausible ticker (foreign listing, index, ETF)
    if not results and _SYMBOL_RE.match(q_upper):
        results.append({"symbol": q_upper, "name": company_name(q_upper)})
    return results


def is_known_symbol(symbol: str) -> bool:
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return False
    frame = universe()
    return bool((frame["Symbol"] == symbol).any())


# --------------------------------------------------------------------------- #
# Quotes and history
# --------------------------------------------------------------------------- #
def quote(symbol: str) -> Dict[str, Any]:
    """Latest price and daily change for a symbol.

    Returns ``{symbol, name, price, previous_close, change, change_pct, currency}``.
    Raises ``MarketDataError`` when the symbol has no tradeable data.
    """
    symbol = (symbol or "").upper().strip()
    if not symbol:
        raise MarketDataError("No symbol supplied")

    cached = _cache_get(f"quote:{symbol}", QUOTE_TTL)
    if cached is not None:
        return cached

    price = previous_close = None
    currency = "USD"
    ticker = yf.Ticker(symbol)
    try:
        fast = ticker.fast_info
        price = float(getattr(fast, "last_price", None) or 0) or None
        previous_close = float(getattr(fast, "previous_close", None) or 0) or None
        currency = getattr(fast, "currency", None) or "USD"
    except Exception:
        pass

    if price is None:
        try:
            hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
        except Exception as exc:
            raise MarketDataError(f"No market data for {symbol}: {exc}") from exc
        if hist is None or hist.empty:
            raise MarketDataError(f"No market data for {symbol}")
        closes = hist["Close"].dropna()
        if closes.empty:
            raise MarketDataError(f"No market data for {symbol}")
        price = float(closes.iloc[-1])
        previous_close = float(closes.iloc[-2]) if len(closes) > 1 else price

    previous_close = previous_close or price
    change = price - previous_close
    data = {
        "symbol": symbol,
        "name": company_name(symbol),
        "price": price,
        "previous_close": previous_close,
        "change": change,
        "change_pct": (change / previous_close * 100) if previous_close else 0.0,
        "currency": currency,
    }
    return _cache_set(f"quote:{symbol}", data)


def history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Daily OHLCV history with a flat column index and a ``Returns`` column."""
    symbol = (symbol or "").upper().strip()
    key = f"history:{symbol}:{period}:{interval}"
    cached = _cache_get(key, HISTORY_TTL)
    if cached is not None:
        return cached.copy()

    try:
        frame = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise MarketDataError(f"Could not download history for {symbol}: {exc}") from exc

    if frame is None or frame.empty:
        raise MarketDataError(f"No price history for {symbol}")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if frame.empty:
        raise MarketDataError(f"No usable price history for {symbol}")
    frame["Returns"] = frame["Close"].pct_change()
    return _cache_set(key, frame)


# --------------------------------------------------------------------------- #
# Branding
# --------------------------------------------------------------------------- #
def logo_url(symbol: str) -> str:
    return LOGO_URL.format(symbol=(symbol or "").upper())


def monogram_colour(text: str) -> str:
    palette = [
        "#6366F1", "#0EA5E9", "#14B8A6", "#F59E0B", "#EC4899",
        "#8B5CF6", "#22C55E", "#EF4444", "#06B6D4", "#F97316",
    ]
    return palette[sum(ord(c) for c in (text or "?")) % len(palette)]


def monogram_text(symbol: str, name: str = "") -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name or "") if w]
    if len(words) >= 2 and name.strip().upper() != (symbol or "").upper():
        return (words[0][0] + words[1][0]).upper()
    return (symbol or "?")[:2].upper()
