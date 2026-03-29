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
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    OrderArgs,
    OrderType,
    TradeParams,
    OpenOrderParams,
)
from py_clob_client.order_builder.constants import BUY, SELL

from skeptic import config
from skeptic.models.order import Order

logger = logging.getLogger(__name__)


def _creds_file() -> str:
    """Return a wallet-specific creds filename so each address gets its own file."""
    addr = config.WALLET_ADDRESS.lower()[-8:]  # last 8 chars of address
    base, ext = os.path.splitext(config.CREDS_FILE)
    return f"{base}_{addr}{ext}"


def _load_or_derive_creds(client: ClobClient) -> ApiCreds:
    """Load cached API creds from disk, or derive and cache them."""
    path = _creds_file()
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            creds = ApiCreds(
                api_key=data["api_key"],
                api_secret=data["api_secret"],
                api_passphrase=data["api_passphrase"],
            )
            logger.info("Loaded API creds from %s", path)
            return creds
        except Exception as e:
            logger.warning("Failed to load cached creds: %s — re-deriving", e)

    creds = client.create_or_derive_api_creds()
    try:
        with open(path, "w") as f:
            json.dump(
                {
                    "wallet": config.WALLET_ADDRESS,
                    "api_key": creds.api_key,
                    "api_secret": creds.api_secret,
                    "api_passphrase": creds.api_passphrase,
                },
                f,
                indent=2,
            )
        logger.info("Derived and cached API creds to %s", path)
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


def presign_market_order(
    client: ClobClient,
    token_id: str,
    usdc_amount: float,
    price_cap: float = 0.99,
):
    """
    Sign a FOK market buy order without submitting it.
    Call this at window start (once per token) so the hot path only needs to POST.
    price_cap: max price willing to pay — set high to guarantee fill; actual fill
               is determined by the order book at POST time, not this cap.
    """
    args = MarketOrderArgs(
        token_id=token_id,
        amount=usdc_amount,
        side=BUY,
        price=price_cap,
        order_type=OrderType.FOK,
    )
    return client.create_market_order(args)


def _parse_fill(resp: dict, usdc_amount: float) -> tuple[float, float]:
    """Extract fill_price and fill_size from a CLOB order response.

    Polymarket returns makingAmount (USDC paid, 6 decimals) and takingAmount
    (shares received, 6 decimals). Fall back to 0 if absent so the caller can
    substitute the trigger price.
    """
    making = float(resp.get("makingAmount") or 0)  # USDC in micro-units
    taking = float(resp.get("takingAmount") or 0)  # shares in micro-units
    if making > 0 and taking > 0:
        fill_usdc  = making / 1e6
        fill_size  = taking / 1e6
        fill_price = round(fill_usdc / fill_size, 4)
        return fill_price, fill_size
    # Legacy fields
    fill_price = float(resp.get("price") or resp.get("avg_price") or 0)
    fill_size  = float(resp.get("size_matched") or resp.get("size") or 0)
    return fill_price, fill_size


def post_presigned_order(
    client: ClobClient,
    signed,
    token_id: str,
    outcome: str,
    usdc_amount: float,
) -> Order:
    """POST a pre-signed order. This is the only network call in the hot path."""
    resp = client.post_order(signed, OrderType.FOK)
    order_id = resp.get("orderID") or resp.get("order_id", "")
    if not order_id:
        raise RuntimeError(f"Market order failed: {resp}")
    fill_price, fill_size = _parse_fill(resp, usdc_amount)
    logger.info("Market order %s  %s  $%.2f USDC  fill_price=%.4f  size=%.4f",
                order_id, outcome, usdc_amount, fill_price, fill_size)
    return Order(
        order_id=order_id,
        token_id=token_id,
        outcome=outcome,
        side="BUY",
        price=fill_price,
        size=fill_size,
        status="FILLED",
        placed_at=time.time(),
        updated_at=time.time(),
    )


def place_market_order(
    client: ClobClient,
    token_id: str,
    outcome: str,
    usdc_amount: float,
) -> Order:
    """
    Place a FOK market buy. usdc_amount = USDC to spend.
    Price is auto-calculated from the live order book.
    Returns an Order with the actual fill price from the response.
    """
    args = MarketOrderArgs(
        token_id=token_id,
        amount=usdc_amount,
        side=BUY,
        price=0,  # auto-calculated from book
        order_type=OrderType.FOK,
    )
    signed = client.create_market_order(args)
    resp = client.post_order(signed, OrderType.FOK)
    order_id = resp.get("orderID") or resp.get("order_id", "")
    if not order_id:
        raise RuntimeError(f"Market order failed: {resp}")
    fill_price, fill_size = _parse_fill(resp, usdc_amount)
    logger.info("Market order %s  %s  $%.2f USDC  fill_price=%.4f  size=%.4f",
                order_id, outcome, usdc_amount, fill_price, fill_size)
    return Order(
        order_id=order_id,
        token_id=token_id,
        outcome=outcome,
        side="BUY",
        price=fill_price,
        size=fill_size,
        status="FILLED",
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
    """Return the USDC (collateral) balance available for trading."""
    try:
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        result = client.get_balance_allowance(params)
        return float(result.get("balance", 0)) / 1e6  # micro-USDC → USDC
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
