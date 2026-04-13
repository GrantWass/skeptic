"""
Wrapper around py-clob-client for order placement, cancellation, and balance queries.
Handles credential derivation and caching.
"""
import asyncio
import concurrent.futures
import json
import logging
import os
import time

import httpx
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
from py_clob_client.endpoints import POST_ORDER
from py_clob_client.headers.headers import create_level_2_headers
from py_clob_client.clob_types import RequestArgs
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.utilities import order_to_json

from skeptic import config
from skeptic.models.order import Order

logger = logging.getLogger(__name__)


# ── Dedicated signing executor ────────────────────────────────────────────────
# A single-thread executor ensures signing work never waits for other tasks.

_signing_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="clob-signer"
)


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
) -> tuple:
    """
    Sign a FOK market buy order and pre-serialize the body without submitting it.
    Call this at window start (once per token) so the hot path only needs to
    compute HMAC headers and POST.

    Returns (signed_order, serialized_body_bytes, body_dict, usdc_amount).
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
    signed = client.create_market_order(args)
    body = order_to_json(signed, client.creds.api_key, OrderType.FOK)
    serialized = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    return signed, serialized, body, usdc_amount


def _parse_fill(resp: dict, usdc_amount: float) -> tuple[float, float]:
    """Extract fill_price and fill_size from a CLOB order response.

    Polymarket returns makingAmount (USDC paid, 6 decimals) and takingAmount
    (shares received, 6 decimals). Fall back to 0 if absent so the caller can
    substitute the trigger price.
    """
    making = float(resp.get("makingAmount") or 0)
    taking = float(resp.get("takingAmount") or 0)
    # If values are too small, assume they're in normal units and need scaling up
    if 0 < making < 1e-3 and 0 < taking < 1e-3:
        fill_usdc = making * 1e6
        fill_size = taking * 1e6
    else:
        fill_usdc = making / 1e6 if making > 1000 else making
        fill_size = taking / 1e6 if taking > 1000 else taking
    if fill_usdc > 0 and fill_size > 0:
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


async def sign_and_post_async(
    http: httpx.AsyncClient,
    client: ClobClient,
    token_id: str,
    outcome: str,
    usdc_amount: float,
    price_cap: float = 0.85,
) -> tuple["Order", float, float]:
    """
    Sign and POST a FOK market order.
    Signing (EIP-712/ECDSA) runs in a thread pool to avoid blocking the event loop.

    Returns (order, sign_ms, post_ms) where sign_ms is the time spent signing
    and building headers, and post_ms is the HTTP round-trip time.
    """
    import asyncio
    _t0 = time.perf_counter()

    def _sign() -> tuple:
        args = MarketOrderArgs(
            token_id=token_id,
            amount=usdc_amount,
            side=BUY,
            price=price_cap,
            order_type=OrderType.FOK,
        )
        signed = client.create_market_order(args)
        body = order_to_json(signed, client.creds.api_key, OrderType.FOK)
        serialized = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        request_args = RequestArgs(
            method="POST",
            request_path=POST_ORDER,
            body=body,
            serialized_body=serialized,
        )
        headers = create_level_2_headers(client.signer, client.creds, request_args)
        return serialized, headers

    serialized, headers = await asyncio.get_event_loop().run_in_executor(_signing_executor, _sign)
    _t_signed = time.perf_counter()
    sign_ms = round((_t_signed - _t0) * 1000, 1)
    logger.info("ORDER TIMING  sign+headers=%.0fms", sign_ms)
    url = f"{client.host}{POST_ORDER}"
    resp = await http.post(url, content=serialized.encode(), headers=headers)
    _t_resp = time.perf_counter()
    post_ms = round((_t_resp - _t_signed) * 1000, 1)
    logger.info("ORDER TIMING  http_post=%.0fms  status=%d", post_ms, resp.status_code)
    if resp.status_code != 200:
        raise RuntimeError(f"Order POST {resp.status_code}: {resp.text}")
    data = resp.json()
    order_id = data.get("orderID") or data.get("order_id", "")
    if not order_id:
        raise RuntimeError(f"Market order failed: {data}")
    fill_price, fill_size = _parse_fill(data, usdc_amount)
    logger.info("Market order %s  %s  $%.2f USDC  fill_price=%.4f  size=%.4f  total=%.0fms",
                order_id, outcome, usdc_amount, fill_price, fill_size,
                (time.perf_counter() - _t0) * 1000)
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
    ), sign_ms, post_ms


async def post_presigned_order_async(
    http: httpx.AsyncClient,
    client: ClobClient,
    signed,
    token_id: str,
    outcome: str,
    usdc_amount: float,
) -> Order:
    """
    POST a pre-signed order using an async HTTP client — no thread handoff.
    Identical semantics to post_presigned_order but runs in the event loop.
    """
    body = order_to_json(signed, client.creds.api_key, OrderType.FOK)
    serialized = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    request_args = RequestArgs(
        method="POST",
        request_path=POST_ORDER,
        body=body,
        serialized_body=serialized,
    )
    headers = create_level_2_headers(client.signer, client.creds, request_args)
    url = f"{client.host}{POST_ORDER}"
    resp = await http.post(url, content=serialized.encode(), headers=headers)
    resp.raise_for_status()
    data = resp.json()
    order_id = data.get("orderID") or data.get("order_id", "")
    if not order_id:
        raise RuntimeError(f"Market order failed: {data}")
    fill_price, fill_size = _parse_fill(data, usdc_amount)
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


async def post_preserialized_order_async(
    http: httpx.AsyncClient,
    client: ClobClient,
    serialized_body: bytes,
    body_dict: dict,
    token_id: str,
    outcome: str,
    usdc_amount: float,
) -> Order:
    """
    Fastest possible order submission: body is already signed and serialized at
    window start. Hot path only computes HMAC headers (~0.3ms) and POSTs.

    Use with presign_market_order() which returns (signed, serialized_body, body_dict, amount).
    """
    request_args = RequestArgs(
        method="POST",
        request_path=POST_ORDER,
        body=body_dict,
        serialized_body=serialized_body.decode(),
    )
    headers = create_level_2_headers(client.signer, client.creds, request_args)
    url = f"{client.host}{POST_ORDER}"
    resp = await http.post(url, content=serialized_body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    order_id = data.get("orderID") or data.get("order_id", "")
    if not order_id:
        raise RuntimeError(f"Market order failed: {data}")
    fill_price, fill_size = _parse_fill(data, usdc_amount)
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
        print("FULL ERROR:", repr(e))
        return 0.0


def get_token_balance(client: ClobClient, token_id: str) -> float:
    """Return the conditional token balance (shares) for a given token ID."""
    try:
        params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
        result = client.get_balance_allowance(params)
        return float(result.get("balance", 0)) / 1e6
    except Exception as e:
        logger.error("get_token_balance(%s) failed: %s", token_id[:16], e)
        return 0.0


def get_trades(client: ClobClient, asset_id: str) -> list[dict]:
    """Return trade history for a given token ID."""
    try:
        params = TradeParams(asset_id=asset_id)
        return client.get_trades(params) or []
    except Exception as e:
        logger.error("get_trades(%s) failed: %s", asset_id, e)
        return []
