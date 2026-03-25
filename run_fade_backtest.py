"""
Entry point for the London False Breakout Fade backtest.

Usage:
    uv run python run_fade_backtest.py <data_file.csv> [results.json]

Example:
    uv run python run_fade_backtest.py data/xauusd_15m_2018-01-01_2021-12-31.csv results/fade_is.json
    uv run python run_fade_backtest.py data/xauusd_15m_2022-01-01_2024-12-31.csv results/fade_oos.json

IS / OOS comparison:
    After both runs, compare expectancy_r values:
        oos_ratio = oos.expectancy_r / is.expectancy_r
        Pass criterion: oos_ratio >= 0.60 (60% retention)
"""

import logging
import sys
from pathlib import Path

from run_backtest import load_ohlcv   # reuse the CSV loader
from src.strategies.london_fade.engine import run_fade_backtest
from src.strategies.london_fade.params import FadeParams
from src.strategies.london_fade.results import compute_fade_results, to_json

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fade_backtest.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path   = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "fade_results.json"

    # ── Parameters ────────────────────────────────────────────────────────────
    # Baseline v1.0 — no optimisation before first IS run.
    params = FadeParams(
        candle_minutes=15,
        london_window_duration_hours=2.0,
        time_exit_hours_after_london_open=5.0,
        min_range_pct=0.0015,
        max_range_pct=0.0080,
        min_overshoot_pct=0.0005,
        spread_price=0.30,
        slippage_price=0.20,
        stop_buffer_floor_pct=0.0005,
        risk_pct=0.01,
        initial_equity=100_000.0,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_ohlcv(data_path)

    # ── Run backtest ──────────────────────────────────────────────────────────
    trade_results = run_fade_backtest(df, params)

    # ── Compute and save results ──────────────────────────────────────────────
    summary = compute_fade_results(trade_results, params)
    json_str = to_json(summary)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_str)

    logger.info("Results written to %s", output_path)
    print(json_str)


if __name__ == "__main__":
    main()
