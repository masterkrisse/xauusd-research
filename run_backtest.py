"""
Entry point for the Asian Range → London Breakout backtest.

Usage:
    python run_backtest.py <data_file.csv> [results.json]

CSV format requirements:
    - Column named 'datetime' (or 'timestamp') in format: YYYY-MM-DD HH:MM:SS
    - Columns: open, high, low, close  (case-insensitive)
    - Timestamps must represent 15-minute bars in UTC.
    - If the datetime column has no timezone, it is assumed to be UTC.

Example:
    python run_backtest.py data/xauusd_15m_2018_2024.csv results/is_2018_2021.json

IS / OOS comparison:
    Run twice with date-filtered CSVs, then compare expectancy_r values:

        is_summary  = compute_results(run_backtest(is_df,  params), params)
        oos_summary = compute_results(run_backtest(oos_df, params), params)

        # Pass criterion: OOS expectancy >= 60% of IS (or 80% given selection bias)
        oos_ratio = oos_summary.expectancy_r / is_summary.expectancy_r
        oos_summary.passed_oos_expectancy = oos_ratio

    Use R-based metrics for IS/OOS comparison, not dollar PnL.
    Dollar metrics are not comparable across windows unless you manually set
    oos starting equity to the IS ending equity.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.strategies.asian_range_breakout.engine import run_backtest
from src.strategies.asian_range_breakout.params import StrategyParams
from src.strategies.asian_range_breakout.results import compute_results, to_json

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def load_ohlcv(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Normalises column names to lowercase.
    Accepts 'datetime' or 'timestamp' as the index column name.
    Localises the index to UTC if it is timezone-naive.

    Returns a DataFrame with a UTC-aware DatetimeIndex and
    columns: open, high, low, close.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    # Accept 'datetime' or 'timestamp' as the time column
    if "datetime" in df.columns:
        df = df.set_index(pd.to_datetime(df["datetime"])).drop(columns=["datetime"])
    elif "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"])).drop(columns=["timestamp"])
    else:
        # Try the first column
        first_col = df.columns[0]
        logger.warning(
            "No 'datetime' or 'timestamp' column found. "
            "Attempting to parse first column '%s' as datetime.",
            first_col,
        )
        df = df.set_index(pd.to_datetime(df[first_col])).drop(columns=[first_col])

    df.index.name = "datetime"

    # Localise to UTC if timezone-naive
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        logger.info("Index had no timezone — assumed UTC.")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()

    # Keep only required columns
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[required].astype(float)

    logger.info(
        "Loaded %d rows | %s → %s",
        len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
    )
    return df


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results.json"

    # ── Parameters ────────────────────────────────────────────────────────────
    # Baseline v1.0 — do not change defaults before the first IS backtest.
    # See params.py for full documentation of each parameter.
    params = StrategyParams(
        candle_minutes=15,
        london_window_duration_hours=2.0,
        time_exit_hours_after_london_open=5.0,
        min_range_pct=0.0015,
        max_range_pct=0.0080,
        spread_price=0.30,
        slippage_price=0.20,
        stop_buffer_floor_pct=0.0005,
        tp_r_multiplier=1.5,
        risk_pct=0.01,
        initial_equity=100_000.0,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_ohlcv(data_path)

    # ── Run backtest ──────────────────────────────────────────────────────────
    trade_results = run_backtest(df, params)

    # ── Compute and save results ──────────────────────────────────────────────
    summary = compute_results(trade_results, params)
    json_str = to_json(summary)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_str)

    logger.info("Results written to %s", output_path)
    print(json_str)


if __name__ == "__main__":
    main()
