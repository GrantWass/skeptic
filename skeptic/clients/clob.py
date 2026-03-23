"""
Wrapper around py-clob-client for order placement, cancellation, and balance queries.
Handles credential derivation and caching.
"""
import json
import logging
import os
import time

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    OrderArgs,
    OrderType,
    TradeParams,
    OpenOrderParams,
)
from py_clob_client.order_builder.constants import BUY, SELL

from skeptic import config
from skeptic.models.order import Order

logger = logging.getLogger(__name__)


def _load_or_derive_creds(client: ClobClient) -> ApiCreds:
    """Load cached API creds from disk, or derive and cache them."""
    if os.path.exists(config.CREDS_FILE):
        try:
            with open(config.CREDS_FILE) as f:
                data = json.load(f)
            creds = ApiCreds(
                api_key=data["api_key"],
                api_secret=data["api_secret"],
                api_passphrase=data["api_passphrase"],
            )
            logger.info("Loaded API creds from %s", config.CREDS_FILE)
            return creds
        except Exception as e:
            logger.warning("Failed to load cached creds: %s — re-deriving", e)

    creds = client.create_or_derive_api_creds()
    try:
        with open(config.CREDS_FILE, "w") as f:
            json.dump(
                {
                    "api_key": creds.api_key,
                    "api_secret": creds.api_secret,
                    "api_passphrase": creds.api_passphrase,
                },
                f,
                indent=2,
            )
        logger.info("Derived and cached API creds to %s", config.CREDS_FILE)
    except Exception as e:
        logger.warning("Could not cache creds: %s", e)

    return creds


def build_client() -> ClobClient:
    """Build and authenticate a ClobClient. Call once at startup."""
    client = ClobClient(
        host=config.CLOB_HOST,
        key=config.PRIVATE_KEY,
        chain_id=config.CHAIN_ID,
        signature_type=1,
        funder=config.WALLET_ADDRESS,
    )
    creds = _load_or_derive_creds(client)
    client.set_api_creds(creds)
    return client


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------

def place_limit_order(
    client: ClobClient,
    token_id: str,
    outcome: str,
    side: str,   # "BUY" or "SELL"
    price: float,
    size: float,
) -> Order:
    """
    Place a GTC limit order. Returns an Order dataclass.
    size = shares (not USDC).
    """
    clob_side = BUY if side == "BUY" else SELL
    args = OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side=clob_side,
    )
    signed = client.create_order(args)
    resp = client.post_order(signed, OrderType.GTC)
    order_id = resp.get("orderID") or resp.get("order_id", "")
    if not order_id:
        raise RuntimeError(f"Order placement failed: {resp}")
    logger.info("Placed %s %s order %s @ %.2f x %.4f shares", side, outcome, order_id, price, size)
    return Order(
        order_id=order_id,
        token_id=token_id,
        outcome=outcome,
        side=side,
        price=price,
        size=size,
        status="OPEN",
        placed_at=time.time(),
        updated_at=time.time(),
    )


def cancel_order(client: ClobClient, order_id: str) -> bool:
    """Cancel a single order. Returns True on success."""
    try:
        client.cancel(order_id)
        logger.info("Cancelled order %s", order_id)
        return True
    except Exception as e:
        logger.warning("Cancel %s failed: %s", order_id, e)
        return False


def cancel_orders(client: ClobClient, order_ids: list[str]) -> None:
    """Cancel multiple orders."""
    for oid in order_ids:
        cancel_order(client, oid)


def get_open_orders(client: ClobClient, market: str | None = None) -> list[dict]:
    """Return open orders, optionally filtered by market condition ID."""
    params = OpenOrderParams(market=market) if market else OpenOrderParams()
    return client.get_orders(params) or []


def get_usdc_balance(client: ClobClient) -> float:
    """Return the USDC balance available for trading."""
    try:
        balance = client.get_balance()
        return float(balance)
    except Exception as e:
        logger.error("get_usdc_balance failed: %s", e)
        return 0.0


def get_trades(client: ClobClient, asset_id: str) -> list[dict]:
    """Return trade history for a given token ID."""
    try:
        params = TradeParams(asset_id=asset_id)
        return client.get_trades(params) or []
    except Exception as e:
        logger.error("get_trades(%s) failed: %s", asset_id, e)
        return []
