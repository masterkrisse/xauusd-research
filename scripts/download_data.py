"""
Download XAUUSD historical 1-minute OHLC data from Dukascopy's public data feed.
No account or API key required.

Resamples to 15-minute bars and saves as a UTC-timestamped CSV ready for run_backtest.py.

Usage:
    uv run python scripts/download_data.py --from 2018-01-01 --to 2024-12-31

Output:
    data/xauusd_15m_<from>_<to>.csv

Dukascopy binary format (bi5 = LZMA-compressed):
    Each daily file contains 1-minute BID candles for that calendar day.
    Each record is 24 bytes (big-endian):
        uint32  : seconds from midnight UTC
        uint32  : open  price * 1000
        uint32  : close price * 1000   (NOTE: Dukascopy field order is OCLH, not OHLC)
        uint32  : low   price * 1000
        uint32  : high  price * 1000
        float32 : volume (tick count)

    XAUUSD price divisor: 1000 (prices have 3 decimal places)
    So actual price = stored_int / 100000 * ... wait:
    Actually Dukascopy stores XAUUSD with divisor 100000 giving 5 decimal places,
    but XAUUSD is quoted to 2 decimal places so prices will have trailing zeros.
    Verified: 2000.00 would be stored as 200000000 (2000.00 * 100000).

URL pattern:
    https://datafeed.dukascopy.com/datafeed/XAUUSD/{YYYY}/{MM:02d}/{DD:02d}/BID_candles_min_1.bi5
    MM is 0-indexed: January = 00, December = 11.
"""

import argparse
import logging
import lzma
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"
PRICE_DIVISOR = 1_000.0     # Dukascopy stores price * 1000 as uint32 for XAUUSD
RECORD_SIZE = 24             # bytes per candle record
RECORD_FORMAT = ">IIIIIf"    # big-endian: uint32 x5 (ms,o,h,lo,c), float32 (vol)
RESAMPLE_TF = "15min"

REQUEST_TIMEOUT = 15         # seconds
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0          # seconds between retries
MAX_WORKERS = 4              # parallel day downloads; >4 triggers Dukascopy rate limits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Download helpers ──────────────────────────────────────────────────────────

def _build_url(d: date) -> str:
    # Dukascopy month is 0-indexed
    return BASE_URL.format(year=d.year, month=d.month - 1, day=d.day)


def _fetch_day(d: date, session: requests.Session) -> bytes | None:
    """
    Fetch the bi5 file for one calendar day.
    Returns raw bytes or None if the day has no data (weekend / holiday).
    """
    url = _build_url(d)
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return None   # no data for this day (weekend / holiday)
            if resp.status_code == 200:
                if len(resp.content) == 0:
                    return None
                return resp.content
            logger.warning(
                "HTTP %d for %s (attempt %d/%d)",
                resp.status_code, d, attempt, RETRY_ATTEMPTS,
            )
        except requests.RequestException as exc:
            logger.warning("Request error for %s: %s (attempt %d/%d)", d, exc, attempt, RETRY_ATTEMPTS)
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF * attempt)
    return None


def _decode_bi5(raw_bytes: bytes, d: date) -> list[dict]:
    """
    Decompress and decode a Dukascopy bi5 file into a list of candle dicts.

    Each record is 24 bytes (big-endian):
        [0:4]   uint32  seconds from midnight UTC
        [4:8]   uint32  open  * 1000
        [8:12]  uint32  close * 1000   (Dukascopy OCLH order, not OHLC)
        [12:16] uint32  low   * 1000
        [16:20] uint32  high  * 1000
        [20:24] float32 volume
    """
    try:
        decompressed = lzma.decompress(raw_bytes)
    except lzma.LZMAError as exc:
        logger.error("LZMA decompression failed for %s: %s", d, exc)
        return []

    n_records = len(decompressed) // RECORD_SIZE
    if len(decompressed) % RECORD_SIZE != 0:
        logger.warning(
            "%s: decompressed size %d is not a multiple of %d — truncating",
            d, len(decompressed), RECORD_SIZE,
        )

    midnight_utc = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    candles = []

    for i in range(n_records):
        offset = i * RECORD_SIZE
        # Dukascopy field order is: seconds, open, close, low, high, volume
        secs, o, c, lo, h, vol = struct.unpack_from(RECORD_FORMAT, decompressed, offset)

        ts = midnight_utc + timedelta(seconds=int(secs))
        candles.append({
            "datetime": ts,
            "open":  o  / PRICE_DIVISOR,
            "high":  h  / PRICE_DIVISOR,
            "low":   lo / PRICE_DIVISOR,
            "close": c  / PRICE_DIVISOR,
        })

    return candles


# ── Download + resample ───────────────────────────────────────────────────────

def _fetch_and_decode(d: date) -> tuple[date, list[dict]]:
    """Fetch and decode one day. Returns (date, candles_list). Thread-safe."""
    session = requests.Session()
    session.headers["User-Agent"] = "xauusd-research-downloader/1.0"
    try:
        raw = _fetch_day(d, session)
        if raw is None:
            return d, []
        return d, _decode_bi5(raw, d)
    finally:
        session.close()


def download_range(start: date, end: date) -> pd.DataFrame:
    """
    Download all 1-minute candles for [start, end] inclusive using parallel workers.
    Returns a UTC-aware DataFrame indexed by datetime.
    """
    total_days = (end - start).days + 1
    all_dates = [start + timedelta(days=i) for i in range(total_days)]

    logger.info(
        "Downloading XAUUSD 1m from %s to %s (%d days, %d workers)",
        start, end, total_days, MAX_WORKERS,
    )

    results: dict[date, list[dict]] = {}
    completed = 0
    lock = Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_and_decode, d): d for d in all_dates}
        for future in as_completed(futures):
            d, candles = future.result()
            with lock:
                results[d] = candles
                completed += 1
                if completed % 100 == 0:
                    fetched = sum(1 for c in results.values() if c)
                    logger.info(
                        "Progress: %d/%d days completed | %d with data",
                        completed, total_days, fetched,
                    )

    # Reassemble in chronological order
    all_candles: list[dict] = []
    fetched = skipped = 0
    for d in all_dates:
        candles = results.get(d, [])
        if candles:
            all_candles.extend(candles)
            fetched += 1
        else:
            skipped += 1

    if not all_candles:
        raise RuntimeError(
            "No candles downloaded. Check network connection and date range."
        )

    logger.info(
        "Download complete | Days fetched=%d skipped=%d | 1m candles=%d",
        fetched, skipped, len(all_candles),
    )

    df = pd.DataFrame(all_candles)
    df = df.set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)  # already UTC-aware
    df = df.sort_index()
    df = df.astype(float)
    return df


def resample_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute OHLC data to 15-minute bars.
    Uses standard OHLC resampling (first open, max high, min low, last close).
    Drops bars with no trades (NaN after resample).
    """
    df_15m = df_1m.resample(RESAMPLE_TF, label="left", closed="left").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }).dropna()

    # Validate: high >= low, high >= open, high >= close
    bad = (
        (df_15m["high"] < df_15m["low"]) |
        (df_15m["high"] < df_15m["open"]) |
        (df_15m["high"] < df_15m["close"])
    )
    if bad.any():
        logger.warning(
            "%d resampled bars have OHLC integrity issues — dropping them.",
            bad.sum(),
        )
        df_15m = df_15m[~bad]

    return df_15m


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download XAUUSD 15m historical data from Dukascopy."
    )
    parser.add_argument(
        "--from", dest="date_from", required=True,
        help="Start date inclusive, format YYYY-MM-DD",
    )
    parser.add_argument(
        "--to", dest="date_to", required=True,
        help="End date inclusive, format YYYY-MM-DD",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--keep-1m", action="store_true",
        help="Also save the raw 1-minute data alongside the 15m output.",
    )
    args = parser.parse_args()

    try:
        start = date.fromisoformat(args.date_from)
        end   = date.fromisoformat(args.date_to)
    except ValueError as exc:
        logger.error("Invalid date format: %s", exc)
        sys.exit(1)

    if start > end:
        logger.error("--from date must be before --to date")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────
    df_1m = download_range(start, end)

    if args.keep_1m:
        path_1m = out_dir / f"xauusd_1m_{start}_{end}.csv"
        df_1m.to_csv(path_1m)
        logger.info("1m data saved → %s  (%d rows)", path_1m, len(df_1m))

    # ── Resample to 15m ───────────────────────────────────────────────────────
    logger.info("Resampling to 15m …")
    df_15m = resample_to_15m(df_1m)
    logger.info("Resampled: %d 1m bars → %d 15m bars", len(df_1m), len(df_15m))

    # ── Validate output ───────────────────────────────────────────────────────
    price_range = (df_15m["close"].min(), df_15m["close"].max())
    logger.info(
        "Price range: %.2f – %.2f | First bar: %s | Last bar: %s",
        price_range[0], price_range[1],
        df_15m.index[0].strftime("%Y-%m-%d %H:%M"),
        df_15m.index[-1].strftime("%Y-%m-%d %H:%M"),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / f"xauusd_15m_{start}_{end}.csv"
    df_15m.index.name = "datetime"
    df_15m.to_csv(out_path)
    logger.info("15m data saved → %s  (%d rows)", out_path, len(df_15m))

    print(f"\nReady to backtest:")
    print(f"  uv run python run_backtest.py {out_path}")


if __name__ == "__main__":
    main()
