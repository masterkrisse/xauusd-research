"""
Parameters for the Round Number Intraday Rejection strategy.

Round number levels: every $50 increment (e.g. $1900, $1950, $2000 …).
These are structural constants — not parameters to optimise.

Signal definition (per 15-minute candle, during active hours):
  SHORT:  candle high comes within `touch_proximity_price` of a $50 level
          AND the candle closes at least `min_rejection_price` below that level.
  LONG:   candle low comes within `touch_proximity_price` of a $50 level
          AND the candle closes at least `min_rejection_price` above that level.

Entry:    Open of the next 15-minute candle after the signal candle.
Stop:     Beyond the wick extreme + stop_buffer_price (fixed USD buffer).
TP:       1R (symmetric, tests win-rate edge cleanly).
Time exit: 16:30 UTC (session + 23.5h) — avoids NorAm session close thin tape.

Signal window: 07:00–16:00 UTC  (London open through NorAm mid-session).
               One trade per session maximum.

All time offsets are relative to the 17:00 UTC session boundary:
  signal_start_hours = 14.0   → 07:00 UTC
  signal_end_hours   = 23.0   → 16:00 UTC
  time_exit_hours    = 23.5   → 16:30 UTC
"""

from dataclasses import dataclass


@dataclass
class RNParams:
    candle_minutes: int  = 15

    # ── Level definition ──────────────────────────────────────────────────────
    level_spacing: float = 50.0          # $50 round-number increments (structural)

    # ── Signal thresholds ─────────────────────────────────────────────────────
    touch_proximity_price: float = 1.50  # wick must reach within $1.50 of level
    min_rejection_price:   float = 1.00  # close must be ≥ $1.00 away from level

    # ── Session offsets (hours from 17:00 UTC session start) ──────────────────
    signal_start_hours: float = 14.0    # 07:00 UTC
    signal_end_hours:   float = 23.0    # 16:00 UTC
    time_exit_hours:    float = 23.5    # 16:30 UTC

    # ── Execution ─────────────────────────────────────────────────────────────
    tp_r_multiplier:     float = 1.0
    stop_buffer_price:   float = 0.50   # buffer beyond wick extreme
    spread_price:        float = 0.30
    slippage_price:      float = 0.20
    risk_pct:            float = 0.01
    initial_equity:      float = 100_000.0

    def validate(self) -> None:
        assert self.level_spacing > 0
        assert self.touch_proximity_price > 0
        assert self.min_rejection_price > 0
        assert self.touch_proximity_price >= self.min_rejection_price, (
            "touch_proximity must be >= min_rejection (wick must reach the level "
            "before the close can be away from it)"
        )
        assert 0.0 < self.risk_pct <= 0.05
        assert self.initial_equity > 0
