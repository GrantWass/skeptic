"""
S3 storage helpers for uploading collected data (prices, orderbook, coin_prices).

Bucket layout:
  {S3_PREFIX}/prices/prices_YYYYMMDD.csv
  {S3_PREFIX}/orderbook/orderbook_YYYYMMDD.csv
  {S3_PREFIX}/coin_prices/{SYMBOL}_1s.csv
"""
import logging
import fnmatch
import io
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


def is_s3_uri(path: str | Path) -> bool:
    return str(path).startswith("s3://")


def _split_s3_uri(uri: str | Path) -> tuple[str, str]:
    s = str(uri)
    if not s.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    no_scheme = s[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def s3_uri_for_data_type(data_type: str) -> str:
    """Return canonical S3 URI for one top-level data type."""
    if data_type not in {"prices", "orderbook", "coin_prices", "models", "reports"}:
        raise ValueError(f"Unsupported data_type: {data_type}")
    pfx = _prefix()
    if pfx:
        return f"s3://{_bucket()}/{pfx}/{data_type}"
    return f"s3://{_bucket()}/{data_type}"


def default_data_location(data_type: str, local_fallback: str) -> str:
    """Prefer S3 location when configured; otherwise return local fallback path."""
    try:
        return s3_uri_for_data_type(data_type)
    except Exception:
        return local_fallback


# -- Low-level helpers ---------------------------------------------------------

def upload_file(local_path: str | Path, s3_key: str) -> bool:
    """Upload a single local file to S3.

    Returns True when an upload occurs, False when skipped as unchanged.
    """
    s3 = _client()
    bucket = _bucket()
    local_path = Path(local_path)

    # Fast unchanged check: if key exists and object size matches local file size,
    # treat it as already synced.
    try:
        head = s3.head_object(Bucket=bucket, Key=s3_key)
        remote_size = int(head.get("ContentLength", -1))
        local_size = local_path.stat().st_size
        if remote_size == local_size:
            log.info("s3 skip unchanged: %s", local_path)
            return False
    except Exception:
        # Missing object (or head failure): proceed with upload.
        pass

    log.info("s3 upload: %s -> s3://%s/%s", local_path, bucket, s3_key)
    s3.upload_file(str(local_path), bucket, s3_key)
    return True


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


def list_csv_paths(base: str | Path, pattern: str = "*.csv") -> list[str]:
    """List CSV paths from either local filesystem dir or S3 URI prefix."""
    if is_s3_uri(base):
        bucket, key_prefix = _split_s3_uri(base)
        key_prefix = key_prefix.rstrip("/")
        if key_prefix:
            key_prefix = f"{key_prefix}/"

        s3 = _client()
        paginator = s3.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                name = Path(key).name
                if fnmatch.fnmatch(name, pattern):
                    out.append(f"s3://{bucket}/{key}")
        return sorted(out)

    return sorted(str(p) for p in Path(base).glob(pattern))


def path_exists(path: str | Path) -> bool:
    """Check existence for local paths and s3://bucket/key URIs."""
    if is_s3_uri(path):
        bucket, key = _split_s3_uri(path)
        if not key:
            return False
        s3 = _client()
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return Path(path).exists()


def read_csv(path: str | Path, **kwargs):
    """Read CSV from local filesystem or S3 URI into a pandas DataFrame."""
    import pandas as pd

    if is_s3_uri(path):
        bucket, key = _split_s3_uri(path)
        s3 = _client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return pd.read_csv(io.StringIO(body), **kwargs)

    return pd.read_csv(path, **kwargs)


# -- Directory sync helpers ----------------------------------------------------

def sync_prices(local_dir: str | Path, today_only: bool = False) -> int:
    """Upload prices_*.csv files from local_dir to S3. Returns file count.

    If today_only=True, only uploads today's UTC-dated file (faster periodic sync).
    """
    prefix = f"{_prefix()}/prices"
    pattern = f"prices_{_today_utc()}.csv" if today_only else "prices_*.csv"
    count = 0
    for f in sorted(Path(local_dir).glob(pattern)):
        if upload_file(f, f"{prefix}/{f.name}"):
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
        if upload_file(f, f"{prefix}/{f.name}"):
            count += 1
    return count


def sync_coin_prices(local_dir: str | Path, today_only: bool = False) -> int:
    """Upload all *_1s.csv files from local_dir to S3. Returns file count.

    today_only has no effect for coin_prices (files aren't date-stamped).
    """
    prefix = f"{_prefix()}/coin_prices"
    count = 0
    for f in sorted(Path(local_dir).glob("*.csv")):
        if upload_file(f, f"{prefix}/{f.name}"):
            count += 1
    return count


def sync_file(local_path: str | Path, data_type: str) -> None:
    """Upload a single file to the correct S3 prefix for data_type.

    data_type must be one of: 'prices', 'orderbook', 'coin_prices'.
    """
    local_path = Path(local_path)
    key = f"{_prefix()}/{data_type}/{local_path.name}"
    upload_file(local_path, key)
