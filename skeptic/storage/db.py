import sqlite3
import os
from contextlib import contextmanager

from skeptic import config


def _ensure_dir() -> None:
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)


def init_db() -> None:
    """Create tables if they don't exist."""
    _ensure_dir()
    with connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id       TEXT PRIMARY KEY,
                asset            TEXT NOT NULL,
                condition_id     TEXT NOT NULL,
                window_start_ts  INTEGER NOT NULL,
                buy_price_used   REAL NOT NULL,
                sell_price_used  REAL NOT NULL,
                capital_deployed REAL,
                up_price_open    REAL,
                down_price_open  REAL,
                up_price_m1      REAL,
                down_price_m1    REAL,
                fill_occurred    INTEGER DEFAULT 0,
                filled_outcome   TEXT,
                sell_order_placed INTEGER DEFAULT 0,
                sell_filled      INTEGER DEFAULT 0,
                realized_pnl     REAL,
                optimal_buy      REAL,
                optimal_sell     REAL,
                resolution_price REAL,
                notes            TEXT,
                created_at       INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id     TEXT PRIMARY KEY,
                session_id   TEXT REFERENCES sessions(session_id),
                token_id     TEXT NOT NULL,
                outcome      TEXT NOT NULL,
                side         TEXT NOT NULL,
                price        REAL NOT NULL,
                size         REAL NOT NULL,
                status       TEXT NOT NULL DEFAULT 'OPEN',
                placed_at    INTEGER,
                updated_at   INTEGER
            );

            CREATE TABLE IF NOT EXISTS price_snapshots (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT REFERENCES sessions(session_id),
                token_id     TEXT NOT NULL,
                outcome      TEXT NOT NULL,
                price        REAL NOT NULL,
                ts           INTEGER NOT NULL,
                minute_mark  INTEGER NOT NULL
            );
        """)


@contextmanager
def connect():
    _ensure_dir()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
