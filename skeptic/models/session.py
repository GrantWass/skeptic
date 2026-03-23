from dataclasses import dataclass, field
import uuid
import time


@dataclass
class TradingSession:
    asset: str
    condition_id: str
    window_start_ts: int
    buy_price_used: float
    sell_price_used: float
    capital_deployed: float
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: int = field(default_factory=lambda: int(time.time()))

    # Prices at window open (minute 0)
    up_price_open: float | None = None
    down_price_open: float | None = None

    # Prices at minute-1 cutoff
    up_price_m1: float | None = None
    down_price_m1: float | None = None

    # Fill outcome
    fill_occurred: bool = False
    filled_outcome: str | None = None   # "UP" or "DOWN"

    # Sell leg
    sell_order_placed: bool = False
    sell_filled: bool = False
    realized_pnl: float | None = None

    # Research back-fill
    optimal_buy: float | None = None
    optimal_sell: float | None = None
    resolution_price: float | None = None   # 1.0 win / 0.0 loss for filled outcome

    notes: str = ""
