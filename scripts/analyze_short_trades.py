"""
Detailed analysis of short trades from the Prior Day Sweep-Rejection Fade strategy.

Enriches each trade with:
  - Prior day range (pdr_pct)
  - Overshoot proxy (stop_distance ≈ overshoot + buffer)
  - TP distance and R:R (structural, varies per trade)
  - 10-session and 20-session trend at time of signal
    Trend = sign of (price_at_signal - price_N_sessions_ago)
    "Session close" = last 15m close before 17:00 UTC each day
  - Day of week (Monday gap risk)
  - Gold price level at entry

Runs both IS and OOS datasets.
Outputs: structured analysis printed to stdout.
"""

import sys
from pathlib import Path
from collections import defaultdict

# allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from run_backtest import load_ohlcv
from src.strategies.prior_day_fade.engine import run_pd_fade_backtest
from src.strategies.prior_day_fade.params import PDFadeParams
from src.strategies.asian_range_breakout.execution import (
    EXIT_TP, EXIT_SL, EXIT_SL_GAP, EXIT_TIME,
)

# ── Params (identical to production run) ──────────────────────────────────────
PARAMS = PDFadeParams(
    candle_minutes=15,
    signal_offset_hours=14.0,
    signal_window_end_hours=23.0,
    time_exit_hours=23.5,
    min_pdr_pct=0.0030,
    max_pdr_pct=0.0200,
    min_overshoot_pct=0.0002,
    spread_price=0.30,
    slippage_price=0.20,
    stop_buffer_floor_pct=0.0003,
    risk_pct=0.01,
    initial_equity=100_000.0,
)

_SESSION_HOUR = 17   # 17:00 UTC


def build_session_closes(df: pd.DataFrame) -> pd.Series:
    """
    Build a series of 'session closes': the last 15m close before 17:00 UTC
    for each UTC calendar date present in df.
    Index: date (not timestamp); value: close price.
    """
    closes = {}
    for d in sorted(df.index.normalize().unique()):
        window_end = d + pd.Timedelta(hours=_SESSION_HOUR)
        window = df[df.index < window_end]
        if not window.empty:
            closes[d.date()] = float(window.iloc[-1]["close"])
    return pd.Series(closes)


def trend_n_sessions_ago(session_closes: pd.Series, entry_date, n: int) -> float | None:
    """
    Return (current_close - close_n_sessions_ago) / close_n_sessions_ago.
    entry_date: the date of the trade entry (the prior session close is the reference).
    Returns None if insufficient history.
    """
    dates = list(session_closes.index)
    try:
        idx = dates.index(entry_date)
    except ValueError:
        return None
    if idx < n:
        return None
    current  = session_closes.iloc[idx]
    past     = session_closes.iloc[idx - n]
    return (current - past) / past


def enrich_trades(trades, session_closes):
    """
    For each TradeResult add a dict of contextual fields.
    Returns list of enriched dicts.
    """
    enriched = []
    for t in trades:
        entry_ts = t.setup.entry_timestamp
        entry_date = entry_ts.date()

        pdr_high = t.setup.asian_range.high
        pdr_low  = t.setup.asian_range.low
        pdr_range_pct = t.setup.asian_range.range_pct

        stop_dist = t.setup.stop_distance
        tp_dist   = abs(t.setup.entry_price - t.setup.effective_tp)
        rr        = tp_dist / stop_dist if stop_dist > 0 else 0.0

        t10 = trend_n_sessions_ago(session_closes, entry_date, 10)
        t20 = trend_n_sessions_ago(session_closes, entry_date, 20)

        enriched.append({
            "entry_date":   entry_date,
            "year":         entry_ts.year,
            "month":        entry_ts.month,
            "dow":          entry_ts.weekday(),   # 0=Mon, 4=Fri
            "direction":    t.setup.direction,
            "entry_price":  t.setup.entry_price,
            "pdr_high":     pdr_high,
            "pdr_low":      pdr_low,
            "pdr_range_pct": pdr_range_pct,
            "stop_dist":    stop_dist,
            "tp_dist":      tp_dist,
            "rr":           rr,
            "realized_r":   t.realized_r,
            "exit_reason":  t.exit_reason,
            "win":          t.realized_r > 0,
            "trend_10":     t10,
            "trend_20":     t20,
        })
    return enriched


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def pct_str(v, total):
    return f"{v}/{total} ({100*v/total:.1f}%)" if total else "0/0"


def analyze(label, trades_enriched, direction):
    d = direction
    dir_label = "LONG" if d == 1 else "SHORT"
    t = [x for x in trades_enriched if x["direction"] == d]
    if not t:
        print(f"\n  No {dir_label} trades in {label}.")
        return

    n = len(t)
    wins = [x for x in t if x["win"]]
    losses = [x for x in t if not x["win"]]
    wr = len(wins) / n
    exp = mean([x["realized_r"] for x in t])
    avg_rr = mean([x["rr"] for x in t])
    tp_count  = sum(1 for x in t if x["exit_reason"] == EXIT_TP)
    sl_count  = sum(1 for x in t if x["exit_reason"] in (EXIT_SL, EXIT_SL_GAP))
    slgap_count = sum(1 for x in t if x["exit_reason"] == EXIT_SL_GAP)
    time_count = sum(1 for x in t if x["exit_reason"] == EXIT_TIME)

    print(f"\n{'='*60}")
    print(f"  {dir_label} trades in {label}  (n={n})")
    print(f"{'='*60}")
    print(f"  WinRate     : {wr:.1%}")
    print(f"  Expectancy  : {exp:+.4f}R")
    print(f"  Avg R:R     : {avg_rr:.2f}:1  (structural, varies by wick overshoot)")
    print(f"  TP/SL/SL_GAP/TIME : {tp_count}/{sl_count}({slgap_count} gap)/{time_count}")

    # ── By year ────────────────────────────────────────────────────────────────
    print(f"\n  By year:")
    by_year = defaultdict(list)
    for x in t:
        by_year[x["year"]].append(x)
    print(f"  {'Year':<6} {'N':>4} {'WR':>7} {'Exp':>8} {'AvgRR':>7} {'TP/SL/TIME':>12}")
    for yr in sorted(by_year):
        yt = by_year[yr]
        yw = [x for x in yt if x["win"]]
        yr_exp = mean([x["realized_r"] for x in yt])
        yr_rr  = mean([x["rr"] for x in yt])
        yr_tp  = sum(1 for x in yt if x["exit_reason"] == EXIT_TP)
        yr_sl  = sum(1 for x in yt if x["exit_reason"] in (EXIT_SL, EXIT_SL_GAP))
        yr_te  = sum(1 for x in yt if x["exit_reason"] == EXIT_TIME)
        print(f"  {yr:<6} {len(yt):>4} {len(yw)/len(yt):>7.1%} {yr_exp:>+8.4f} {yr_rr:>7.2f} {yr_tp:>3}/{yr_sl:>3}/{yr_te:>3}")

    # ── By trend_10 (uptrend vs downtrend) ────────────────────────────────────
    has_trend = [x for x in t if x["trend_10"] is not None]
    if has_trend:
        up   = [x for x in has_trend if x["trend_10"] > 0.0]
        flat = [x for x in has_trend if x["trend_10"] == 0.0]
        down = [x for x in has_trend if x["trend_10"] < 0.0]

        print(f"\n  By 10-session trend (price vs 10 sessions ago):")
        for regime_label, subset in [("Uptrend (>0)", up), ("Downtrend (<0)", down)]:
            if not subset:
                continue
            sw = [x for x in subset if x["win"]]
            s_exp = mean([x["realized_r"] for x in subset])
            s_tp  = sum(1 for x in subset if x["exit_reason"] == EXIT_TP)
            s_sl  = sum(1 for x in subset if x["exit_reason"] in (EXIT_SL, EXIT_SL_GAP))
            s_te  = sum(1 for x in subset if x["exit_reason"] == EXIT_TIME)
            print(f"    {regime_label:<20}: n={len(subset):>3}  WR={len(sw)/len(subset):.1%}  "
                  f"Exp={s_exp:+.4f}R  TP/SL/TIME={s_tp}/{s_sl}/{s_te}")

    # ── By 20-session trend ────────────────────────────────────────────────────
    has_t20 = [x for x in t if x["trend_20"] is not None]
    if has_t20:
        up20   = [x for x in has_t20 if x["trend_20"] > 0.0]
        down20 = [x for x in has_t20 if x["trend_20"] < 0.0]

        print(f"\n  By 20-session trend:")
        for regime_label, subset in [("Uptrend (>0)", up20), ("Downtrend (<0)", down20)]:
            if not subset:
                continue
            sw = [x for x in subset if x["win"]]
            s_exp = mean([x["realized_r"] for x in subset])
            print(f"    {regime_label:<20}: n={len(subset):>3}  WR={len(sw)/len(subset):.1%}  "
                  f"Exp={s_exp:+.4f}R")

    # ── By day of week ─────────────────────────────────────────────────────────
    print(f"\n  By day of week (entry):")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for dow in range(5):
        subset = [x for x in t if x["dow"] == dow]
        if not subset:
            continue
        sw = [x for x in subset if x["win"]]
        sg = [x for x in subset if x["exit_reason"] == EXIT_SL_GAP]
        s_exp = mean([x["realized_r"] for x in subset])
        print(f"    {dow_names[dow]}: n={len(subset):>3}  WR={len(sw)/len(subset):.1%}  "
              f"Exp={s_exp:+.4f}R  SL_GAP={len(sg)}")

    # ── By PDR range size ─────────────────────────────────────────────────────
    print(f"\n  By prior day range (pdr_pct):")
    tight  = [x for x in t if x["pdr_range_pct"] < 0.006]
    medium = [x for x in t if 0.006 <= x["pdr_range_pct"] < 0.010]
    wide   = [x for x in t if x["pdr_range_pct"] >= 0.010]
    for rl, subset in [("Tight  (<0.6%)", tight), ("Medium (0.6-1.0%)", medium), ("Wide  (>=1.0%)", wide)]:
        if not subset:
            continue
        sw = [x for x in subset if x["win"]]
        s_exp = mean([x["realized_r"] for x in subset])
        s_rr  = mean([x["rr"] for x in subset])
        print(f"    {rl}: n={len(subset):>3}  WR={len(sw)/len(subset):.1%}  "
              f"Exp={s_exp:+.4f}R  AvgRR={s_rr:.2f}")

    # ── By R:R band ───────────────────────────────────────────────────────────
    print(f"\n  By structural R:R at entry:")
    rr_bands = [
        ("RR < 2",     [x for x in t if x["rr"] < 2.0]),
        ("RR 2-4",     [x for x in t if 2.0 <= x["rr"] < 4.0]),
        ("RR 4-8",     [x for x in t if 4.0 <= x["rr"] < 8.0]),
        ("RR >= 8",    [x for x in t if x["rr"] >= 8.0]),
    ]
    for rl, subset in rr_bands:
        if not subset:
            continue
        sw = [x for x in subset if x["win"]]
        s_exp = mean([x["realized_r"] for x in subset])
        s_tp  = sum(1 for x in subset if x["exit_reason"] == EXIT_TP)
        print(f"    {rl:<10}: n={len(subset):>3}  WR={len(sw)/len(subset):.1%}  "
              f"Exp={s_exp:+.4f}R  TP_hits={s_tp}")


def run_and_analyze(filepath, label):
    print(f"\n\n{'#'*65}")
    print(f"  DATASET: {label}")
    print(f"  FILE   : {filepath}")
    print(f"{'#'*65}")

    df = load_ohlcv(filepath)
    session_closes = build_session_closes(df)

    import logging
    logging.disable(logging.CRITICAL)   # suppress engine logs for clean output
    trades = run_pd_fade_backtest(df, PARAMS)
    logging.disable(logging.NOTSET)

    if not trades:
        print("  No trades.")
        return

    enriched = enrich_trades(trades, session_closes)
    n_long  = sum(1 for x in enriched if x["direction"] ==  1)
    n_short = sum(1 for x in enriched if x["direction"] == -1)
    print(f"\n  Total trades: {len(enriched)}  (LONG={n_long}  SHORT={n_short})")

    analyze(label, enriched, direction=1)
    analyze(label, enriched, direction=-1)


if __name__ == "__main__":
    run_and_analyze("data/xauusd_15m_2018-01-01_2021-12-31.csv", "IS 2018-2021")
    run_and_analyze("data/xauusd_15m_2022-01-01_2024-12-31.csv", "OOS 2022-2024")
