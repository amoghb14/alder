import sqlite3
import hashlib
import json

DB_FILE = "alder_system.db"

def get_db_connection():
    # timeout=20 allows the process to wait for the lock to release before erroring
    conn = sqlite3.connect(DB_FILE, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    # Users table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    # Portfolios table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            user_id INTEGER PRIMARY KEY,
            balance REAL NOT NULL,
            holdings TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password):
    try:
        conn = get_db_connection()
        hashed = hash_password(password)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                       (username, email, hashed))
        user_id = cursor.lastrowid
        # Create default portfolio
        conn.execute("INSERT INTO portfolios (user_id, balance, holdings) VALUES (?, ?, ?)",
                     (user_id, 100000.0, json.dumps({})))
        conn.commit()
        conn.close()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username or Email already exists."
    except Exception as e:
        return False, str(e)

def authenticate_user(username, password):
    conn = get_db_connection()
    hashed = hash_password(password)
    user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                        (username, hashed)).fetchone()
    conn.close()
    return user

def get_portfolio(user_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM portfolios WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return {"balance": row["balance"], "holdings": json.loads(row["holdings"])}
    return None

def save_portfolio(user_id, balance, holdings):
    conn = get_db_connection()
    conn.execute("UPDATE portfolios SET balance = ?, holdings = ? WHERE user_id = ?",
                 (balance, json.dumps(holdings), user_id))
    conn.commit()
    conn.close()

# Seed the test account
def seed_test_account():
    try:
        create_user("amogh", "amogh@test.com", "abc")
    except:
        pass

if __name__ == "__main__":
    init_db()
    seed_test_account()
