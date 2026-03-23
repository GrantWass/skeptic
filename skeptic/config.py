import os
from dotenv import load_dotenv

load_dotenv()

# --- API endpoints ---
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
WS_USER_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# --- Auth ---
PRIVATE_KEY: str = os.environ["PRIVATE_KEY"]
WALLET_ADDRESS: str = os.environ["WALLET_ADDRESS"]
CLOB_API_KEY: str = os.getenv("CLOB_API_KEY", "")
CLOB_SECRET: str = os.getenv("CLOB_SECRET", "")
CLOB_PASSPHRASE: str = os.getenv("CLOB_PASSPHRASE", "")
CHAIN_ID: int = 137  # Polygon

# --- Strategy parameters ---
BUY_PRICE: float | None = None   # set from research output (scripts/research.py)
SELL_PRICE: float | None = None  # set from research output (scripts/research.py)
POSITION_SIZE_PCT: float = 0.05  # fraction of total capital to deploy per window
MONITOR_SECS: int = 60        # seconds to monitor for fills before cancelling

# --- Window ---
WINDOW_SECS: int = 300        # 5-minute windows


# Available 5-min assets on Polymarket: BTC, ETH, SOL, DOGE, XRP, BNB, HYPE
# --- Assets to trade/research ---
ASSETS: list[str] = ["BTC", "ETH", "SOL", "DOGE", "XRP", "BNB", "HYPE"]

# --- Paths ---
CREDS_FILE = ".creds.json"
DB_PATH = "data/sessions.db"
REPORTS_DIR = "data/reports"
