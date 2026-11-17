"""Persistence layer for ALDER.

Two interchangeable backends are supported and picked automatically:

* **PostgreSQL** when a ``DATABASE_URL`` is available (an environment variable
  or Streamlit's ``st.secrets``). This is what a cloud deployment such as
  Supabase / Streamlit Community Cloud uses.
* **A local SQLite file** next to this module otherwise. This is the default
  and needs no configuration at all, so the app runs out of the box.

Nothing here requires a ``.streamlit/secrets.toml``: when one is missing the
local SQLite backend is used instead of raising Streamlit's
"No secrets found" error.
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime

try:  # Optional: only needed when a PostgreSQL DATABASE_URL is configured.
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover - the SQLite backend has no such need
    psycopg2 = None

DB_FILE = os.environ.get(
    "ALDER_DB_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "alder_system.db"),
)

STARTING_BALANCE = 100000.0

_INTEGRITY_ERRORS = (sqlite3.IntegrityError,)
_DB_ERRORS = (sqlite3.Error,)
if psycopg2 is not None:
    _INTEGRITY_ERRORS += (psycopg2.IntegrityError,)
    _DB_ERRORS += (psycopg2.Error,)


class DatabaseError(RuntimeError):
    """Raised when the configured database cannot be reached or initialised."""


def database_url():
    """Return the configured PostgreSQL URL, or "" when there is none.

    Never raises: a missing ``secrets.toml`` (or a missing DATABASE_URL key)
    simply means "use the local SQLite database".
    """
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    try:
        import streamlit as st

        return str(st.secrets["DATABASE_URL"]).strip()
    except Exception:
        return ""


def is_postgres():
    return bool(database_url())


def _connect():
    """Open a connection. Returns ``(connection, is_postgres)``.

    Raises ``DatabaseError`` when the configured backend cannot be reached.
    """
    url = database_url()
    if url:
        if psycopg2 is None:
            raise DatabaseError(
                "DATABASE_URL is set but psycopg2 is not installed. "
                "Run: pip install psycopg2-binary"
            )
        try:
            return psycopg2.connect(url), True
        except Exception as exc:
            raise DatabaseError(
                f"Could not connect to PostgreSQL using DATABASE_URL: {exc}"
            ) from exc

    try:
        conn = sqlite3.connect(DB_FILE, timeout=20)
    except sqlite3.Error as exc:
        raise DatabaseError(
            f"Could not open the ALDER database at '{DB_FILE}': {exc}. "
            "Check that the folder is writable, or set ALDER_DB_FILE to a "
            "writable path."
        ) from exc
    conn.row_factory = sqlite3.Row
    return conn, False


def get_db_connection():
    """Backwards-compatible helper that returns a raw connection."""
    return _connect()[0]


def _q(sql, postgres):
    """SQLite uses ``?`` placeholders, PostgreSQL uses ``%s``."""
    return sql.replace("?", "%s") if postgres else sql


def _cursor(conn, postgres, dict_rows=False):
    if postgres and dict_rows:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    return conn.cursor()


def init_db():
    conn, postgres = _connect()
    try:
        cur = conn.cursor()
        try:
            id_column = (
                "SERIAL PRIMARY KEY" if postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
            )
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS users (
                    id {id_column},
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    user_id INTEGER PRIMARY KEY,
                    balance REAL NOT NULL,
                    holdings TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            trade_id = "SERIAL PRIMARY KEY" if postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS trades (
                    id {trade_id},
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    value REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    user_id INTEGER NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    PRIMARY KEY (user_id, snapshot_date)
                )
            """)
            conn.commit()
        finally:
            cur.close()
    except _DB_ERRORS as exc:
        raise DatabaseError(
            f"Could not initialise the ALDER database: {exc}"
        ) from exc
    finally:
        conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, email, password):
    hashed = hash_password(password)
    conn = None
    try:
        conn, postgres = _connect()
        cur = _cursor(conn, postgres)
        try:
            if postgres:
                cur.execute(
                    "INSERT INTO users (username, email, password) "
                    "VALUES (%s, %s, %s) RETURNING id",
                    (username, email, hashed),
                )
                user_id = cur.fetchone()[0]
            else:
                cur.execute(
                    "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (username, email, hashed),
                )
                user_id = cur.lastrowid
            # Create default portfolio
            cur.execute(
                _q(
                    "INSERT INTO portfolios (user_id, balance, holdings) VALUES (?, ?, ?)",
                    postgres,
                ),
                (user_id, STARTING_BALANCE, json.dumps({})),
            )
            conn.commit()
        finally:
            cur.close()
        return True, "Account created successfully."
    except _INTEGRITY_ERRORS:
        return False, "Username or Email already exists."
    except (DatabaseError, *_DB_ERRORS) as exc:
        return False, str(exc)
    finally:
        if conn is not None:
            conn.close()


def authenticate_user(username, password):
    conn, postgres = _connect()
    try:
        hashed = hash_password(password)
        cur = _cursor(conn, postgres, dict_rows=True)
        try:
            cur.execute(
                _q(
                    "SELECT * FROM users WHERE username = ? AND password = ?",
                    postgres,
                ),
                (username, hashed),
            )
            return cur.fetchone()
        finally:
            cur.close()
    finally:
        conn.close()


def get_portfolio(user_id):
    conn, postgres = _connect()
    try:
        cur = _cursor(conn, postgres, dict_rows=True)
        try:
            cur.execute(
                _q("SELECT * FROM portfolios WHERE user_id = ?", postgres), (user_id,)
            )
            row = cur.fetchone()
        finally:
            cur.close()
    finally:
        conn.close()
    if row:
        return {"balance": row["balance"], "holdings": json.loads(row["holdings"])}
    return None


def ensure_portfolio(user_id):
    """Return the user's portfolio, creating a default one if it is missing."""
    portfolio = get_portfolio(user_id)
    if portfolio is not None:
        return portfolio

    conn, postgres = _connect()
    try:
        cur = _cursor(conn, postgres)
        try:
            insert = "INSERT INTO portfolios (user_id, balance, holdings) VALUES (?, ?, ?)"
            if postgres:
                insert += " ON CONFLICT (user_id) DO NOTHING"
            else:
                insert = insert.replace("INSERT INTO", "INSERT OR IGNORE INTO")
            cur.execute(
                _q(insert, postgres),
                (user_id, STARTING_BALANCE, json.dumps({})),
            )
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()
    return get_portfolio(user_id) or {
        "balance": STARTING_BALANCE,
        "holdings": {},
    }


def save_portfolio(user_id, balance, holdings):
    conn, postgres = _connect()
    try:
        cur = _cursor(conn, postgres)
        try:
            cur.execute(
                _q(
                    "UPDATE portfolios SET balance = ?, holdings = ? WHERE user_id = ?",
                    postgres,
                ),
                (balance, json.dumps(holdings), user_id),
            )
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Trade log and portfolio history
# --------------------------------------------------------------------------- #
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def record_trade(user_id, symbol, side, shares, price, total_value, cash):
    """Append a fill to the trade log and refresh today's portfolio snapshot."""
    conn, postgres = _connect()
    try:
        cur = _cursor(conn, postgres)
        try:
            cur.execute(
                _q(
                    "INSERT INTO trades (user_id, symbol, side, shares, price, value, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    postgres,
                ),
                (user_id, symbol.upper(), side.upper(), float(shares), float(price),
                 float(shares) * float(price), _timestamp()),
            )
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()
    snapshot_portfolio(user_id, total_value, cash)


def trade_history(user_id, limit: int = 100):
    conn, postgres = _connect()
    try:
        cur = _cursor(conn, postgres, dict_rows=True)
        try:
            cur.execute(
                _q("SELECT * FROM trades WHERE user_id = ? ORDER BY id DESC", postgres),
                (user_id,),
            )
            rows = cur.fetchall() or []
        finally:
            cur.close()
    finally:
        conn.close()
    return [dict(row) for row in rows][:limit]


def snapshot_portfolio(user_id, total_value, cash, snapshot_date=None):
    """Upsert one portfolio valuation per day."""
    day = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    conn, postgres = _connect()
    try:
        cur = _cursor(conn, postgres)
        try:
            cur.execute(
                _q(
                    "INSERT INTO portfolio_snapshots (user_id, snapshot_date, total_value, cash) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT (user_id, snapshot_date) "
                    "DO UPDATE SET total_value = excluded.total_value, cash = excluded.cash",
                    postgres,
                ),
                (user_id, day, float(total_value), float(cash)),
            )
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


def portfolio_history(user_id, limit: int = 365):
    conn, postgres = _connect()
    try:
        cur = _cursor(conn, postgres, dict_rows=True)
        try:
            cur.execute(
                _q(
                    "SELECT snapshot_date, total_value, cash FROM portfolio_snapshots "
                    "WHERE user_id = ? ORDER BY snapshot_date ASC",
                    postgres,
                ),
                (user_id,),
            )
            rows = cur.fetchall() or []
        finally:
            cur.close()
    finally:
        conn.close()
    return [dict(row) for row in rows][-limit:]


# Seed the test account
def seed_test_account():
    create_user("amogh", "amogh@test.com", "abc")


if __name__ == "__main__":
    init_db()
    seed_test_account()
