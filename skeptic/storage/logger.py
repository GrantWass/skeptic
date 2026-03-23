import time
from skeptic.models.session import TradingSession
from skeptic.models.order import Order
from skeptic.storage import db


def insert_session(session: TradingSession) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, asset, condition_id, window_start_ts,
                buy_price_used, sell_price_used, capital_deployed,
                up_price_open, down_price_open,
                fill_occurred, filled_outcome,
                sell_order_placed, sell_filled, realized_pnl,
                optimal_buy, optimal_sell, resolution_price,
                notes, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                session.session_id, session.asset, session.condition_id,
                session.window_start_ts, session.buy_price_used,
                session.sell_price_used, session.capital_deployed,
                session.up_price_open, session.down_price_open,
                int(session.fill_occurred), session.filled_outcome,
                int(session.sell_order_placed), int(session.sell_filled),
                session.realized_pnl, session.optimal_buy, session.optimal_sell,
                session.resolution_price, session.notes, session.created_at,
            ),
        )


def update_m1_prices(session_id: str, up_price_m1: float, down_price_m1: float) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE sessions SET up_price_m1=?, down_price_m1=? WHERE session_id=?",
            (up_price_m1, down_price_m1, session_id),
        )


def update_fill(session_id: str, filled_outcome: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE sessions SET fill_occurred=1, filled_outcome=? WHERE session_id=?",
            (filled_outcome, session_id),
        )


def update_sell_placed(session_id: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE sessions SET sell_order_placed=1 WHERE session_id=?",
            (session_id,),
        )


def update_sell_filled(session_id: str, pnl: float) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE sessions SET sell_filled=1, realized_pnl=? WHERE session_id=?",
            (pnl, session_id),
        )


def update_resolution(session_id: str, resolution_price: float, pnl: float | None = None) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE sessions SET resolution_price=?, realized_pnl=COALESCE(realized_pnl, ?) WHERE session_id=?",
            (resolution_price, pnl, session_id),
        )


def insert_order(order: Order, session_id: str) -> None:
    now = int(time.time())
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO orders (order_id, session_id, token_id, outcome, side, price, size, status, placed_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (order.order_id, session_id, order.token_id, order.outcome,
             order.side, order.price, order.size, order.status, now, now),
        )


def update_order_status(order_id: str, status: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE orders SET status=?, updated_at=? WHERE order_id=?",
            (status, int(time.time()), order_id),
        )


def insert_price_snapshot(session_id: str, token_id: str, outcome: str, price: float, minute_mark: int) -> None:
    now = int(time.time())
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO price_snapshots (session_id, token_id, outcome, price, ts, minute_mark)
            VALUES (?,?,?,?,?,?)
            """,
            (session_id, token_id, outcome, price, now, minute_mark),
        )
