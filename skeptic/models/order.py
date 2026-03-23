from dataclasses import dataclass, field


@dataclass
class Order:
    order_id: str
    token_id: str
    outcome: str        # "UP" or "DOWN"
    side: str           # "BUY" or "SELL"
    price: float
    size: float         # shares
    status: str = "OPEN"   # OPEN | FILLED | CANCELLED
    size_matched: float = 0.0
    placed_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class OrderPair:
    up_order: Order
    down_order: Order
    placed_at: float    # unix timestamp


@dataclass
class Fill:
    order_id: str
    outcome: str        # "UP" or "DOWN"
    price: float
    size: float
    ts: float
