"""
S3 storage helpers for uploading collected data (prices, orderbook, coin_prices).

Bucket layout:
  {S3_PREFIX}/prices/prices_YYYYMMDD.csv
  {S3_PREFIX}/orderbook/orderbook_YYYYMMDD.csv
  {S3_PREFIX}/coin_prices/{SYMBOL}_1s.csv
"""
import logging
from datetime import datetime, timezone
from pathlib import Path


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

log = logging.getLogger(__name__)


def _client():
    import boto3
    return boto3.client("s3")


def _bucket() -> str:
    from skeptic import config
    if not config.S3_BUCKET:
        raise ValueError(
            "S3_BUCKET environment variable is not set. "
            "Add it to your .env file to enable S3 storage."
        )
    return config.S3_BUCKET


def _prefix() -> str:
    from skeptic import config
    return config.S3_PREFIX.rstrip("/")


# ── Low-level helpers ─────────────────────────────────────────────────────────

def upload_file(local_path: str | Path, s3_key: str) -> None:
    """Upload a single local file to S3."""
    s3 = _client()
    bucket = _bucket()
    log.info("s3 upload: %s → s3://%s/%s", local_path, bucket, s3_key)
    s3.upload_file(str(local_path), bucket, s3_key)


def list_keys(prefix: str) -> list[str]:
    """Return all S3 keys whose prefix matches (handles pagination)."""
    s3 = _client()
    bucket = _bucket()
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ── Directory sync helpers ─────────────────────────────────────────────────────

def sync_prices(local_dir: str | Path, today_only: bool = False) -> int:
    """Upload prices_*.csv files from local_dir to S3. Returns file count.

    If today_only=True, only uploads today's UTC-dated file (faster periodic sync).
    """
    prefix = f"{_prefix()}/prices"
    pattern = f"prices_{_today_utc()}.csv" if today_only else "prices_*.csv"
    count = 0
    for f in sorted(Path(local_dir).glob(pattern)):
        upload_file(f, f"{prefix}/{f.name}")
        count += 1
    return count


def sync_orderbook(local_dir: str | Path, today_only: bool = False) -> int:
    """Upload orderbook_*.csv files from local_dir to S3. Returns file count.

    If today_only=True, only uploads today's UTC-dated file (faster periodic sync).
    """
    prefix = f"{_prefix()}/orderbook"
    pattern = f"orderbook_{_today_utc()}.csv" if today_only else "orderbook_*.csv"
    count = 0
    for f in sorted(Path(local_dir).glob(pattern)):
        upload_file(f, f"{prefix}/{f.name}")
        count += 1
    return count


def sync_coin_prices(local_dir: str | Path, today_only: bool = False) -> int:
    """Upload all *_1s.csv files from local_dir to S3. Returns file count.

    today_only has no effect for coin_prices (files aren't date-stamped).
    """
    prefix = f"{_prefix()}/coin_prices"
    count = 0
    for f in sorted(Path(local_dir).glob("*.csv")):
        upload_file(f, f"{prefix}/{f.name}")
        count += 1
    return count


def sync_file(local_path: str | Path, data_type: str) -> None:
    """Upload a single file to the correct S3 prefix for data_type.

    data_type must be one of: 'prices', 'orderbook', 'coin_prices'.
    """
    local_path = Path(local_path)
    key = f"{_prefix()}/{data_type}/{local_path.name}"
    upload_file(local_path, key)
