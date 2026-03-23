from dataclasses import dataclass


@dataclass
class Token:
    token_id: str    # CLOB token ID (clobTokenIds entry)
    outcome: str     # "UP" or "DOWN"


@dataclass
class Market:
    condition_id: str
    slug: str
    asset: str         # e.g. "BTC"
    start_ts: int      # unix timestamp of window open
    end_ts: int        # unix timestamp of window close
    up_token: Token
    down_token: Token
    active: bool
