# Funding Your Wallet with MATIC for CTF Redemptions

Polymarket runs on Polygon. Redeeming winning positions requires a small amount of MATIC
(Polygon's native gas token) in your wallet. Each redemption costs roughly **0.05–0.15 MATIC**
(~$0.03–$0.09 at typical prices). Funding with **1–2 MATIC** covers many sessions.

---

## Your Wallet Address

Your wallet address is in `.env` as `WALLET_ADDRESS`. This is the address you need to fund.

---

## Option 1 — Bridge from Ethereum (cheapest long-term)

1. Go to [https://wallet.polygon.technology/polygon/bridge](https://wallet.polygon.technology/polygon/bridge)
2. Connect the same wallet (MetaMask, etc.)
3. Bridge ETH → MATIC or USDC → MATIC on the Polygon side
4. Wait ~10–30 minutes for the bridge to finalize

## Option 2 — Buy MATIC directly on Coinbase / Kraken / Binance

1. Buy MATIC on any exchange
2. Withdraw to your wallet address, **selecting Polygon network** (not Ethereum)
3. Arrives in ~5 minutes

> ⚠️ Always select **Polygon (MATIC)** as the withdrawal network, not Ethereum.
> Sending on the wrong network results in lost funds.

## Option 3 — Swap inside a Polygon wallet (fastest if you have USDC on Polygon)

Polymarket deposits leave USDC on Polygon. You can swap a small amount to MATIC:

1. Open [https://app.uniswap.org](https://app.uniswap.org) and switch network to Polygon
2. Swap ~$1–2 of USDC → MATIC
3. Confirm in your wallet

---

## Verify the Balance

After funding, check your MATIC balance:

```bash
# Quick check via public RPC
curl -s -X POST https://polygon-bor-rpc.publicnode.com \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","method":"eth_getBalance","params":["YOUR_WALLET_ADDRESS","latest"],"id":1}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(int(r['result'],16)/1e18, 'MATIC')"
```

---

## Re-enable Auto-Redemption

Once funded, uncomment these lines in [skeptic/executor/high_buy.py](../skeptic/executor/high_buy.py) (around line 241):

```python
# 5. Redeem any won trades on-chain
if not self.dry_run and self._trades:
    await self._redeem_won_trades()
```

Or run `scripts/test_redeem.py` manually after each session to redeem in batch.
