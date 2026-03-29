"""
On-chain CTF (Conditional Token Framework) helpers.

Uses eth_account + eth_abi + httpx to sign and broadcast transactions
directly via Polygon JSON-RPC — no web3.py required.
"""
import logging
import os

import httpx
from eth_abi import encode
from eth_account import Account
from eth_utils import keccak
from py_clob_client.config import get_contract_config

from skeptic import config

log = logging.getLogger(__name__)

# Override with POLYGON_RPC env var (e.g. Alchemy/Infura/QuickNode URL)
_POLYGON_RPC_FALLBACKS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://rpc.ankr.com/polygon",
    "https://1rpc.io/matic",
]
POLYGON_RPC: str = (
    os.environ.get("POLYGON_RPC")
    or _POLYGON_RPC_FALLBACKS[0]
)

# keccak256("redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
_REDEEM_SELECTOR: bytes = keccak(
    text="redeemPositions(address,bytes32,bytes32,uint256[])"
)[:4]


def redeem_positions(condition_id: str, side: str) -> str | None:
    """
    Redeem a resolved winning position on-chain via the CTF contract.

    condition_id: hex string from Market.condition_id
    side: "UP" (outcome index 0, indexSet=1) or "DOWN" (outcome index 1, indexSet=2)

    Returns the transaction hash on success, None on failure.
    """
    cfg = get_contract_config(config.CHAIN_ID)
    ctf_address  = cfg.conditional_tokens
    collateral   = cfg.collateral

    cid_hex = condition_id.removeprefix("0x").zfill(64)
    cid_bytes    = bytes.fromhex(cid_hex)
    parent_bytes = bytes(32)                          # parentCollectionId = 0
    index_sets   = [1] if side == "UP" else [2]       # UP=outcome0, DOWN=outcome1

    calldata = _REDEEM_SELECTOR + encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [collateral, parent_bytes, cid_bytes, index_sets],
    )

    account = Account.from_key(config.PRIVATE_KEY)

    rpcs = ([POLYGON_RPC] + [r for r in _POLYGON_RPC_FALLBACKS if r != POLYGON_RPC])

    with httpx.Client(timeout=15) as http:
        for rpc in rpcs:
            try:
                nonce     = _rpc_int(http, rpc, "eth_getTransactionCount", [account.address, "latest"])
                gas_price = _rpc_int(http, rpc, "eth_gasPrice", [])

                tx = {
                    "to":       ctf_address,
                    "data":     "0x" + calldata.hex(),
                    "gas":      200_000,
                    "gasPrice": gas_price,
                    "nonce":    nonce,
                    "chainId":  config.CHAIN_ID,
                    "value":    0,
                }
                signed = account.sign_transaction(tx)
                raw_tx = signed.raw_transaction if hasattr(signed, "raw_transaction") else signed.rawTransaction

                result = _rpc(http, rpc, "eth_sendRawTransaction", ["0x" + raw_tx.hex()])
                if "error" in result:
                    log.error("Redeem failed  cond=…%s  side=%s  rpc=%s  error=%s",
                              condition_id[-8:], side, rpc, result["error"])
                    continue

                tx_hash = result.get("result", "")
                log.info("Redeemed  cond=…%s  side=%s  tx=%s…",
                         condition_id[-8:], side, tx_hash[:20])
                return tx_hash

            except Exception as exc:
                log.warning("Redeem attempt failed  rpc=%s  %s — trying next", rpc, exc)

    log.error("Redeem failed for all RPCs  cond=…%s  side=%s", condition_id[-8:], side)
    return None


# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def _rpc(http: httpx.Client, url: str, method: str, params: list) -> dict:
    resp = http.post(url, json={
        "jsonrpc": "2.0", "method": method, "params": params, "id": 1,
    })
    resp.raise_for_status()
    return resp.json()


def _rpc_int(http: httpx.Client, url: str, method: str, params: list) -> int:
    return int(_rpc(http, url, method, params)["result"], 16)
