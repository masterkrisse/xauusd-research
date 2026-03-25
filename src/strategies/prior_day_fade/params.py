"""
Parameters for the Prior Day High/Low Sweep-and-Rejection Fade strategy.

Session definition (17:00 UTC close convention):
  A "trading session" runs from 17:00 UTC on day D to 17:00 UTC on day D+1.
  This aligns with the CME/NYMEX metals settlement convention (approximately)
  and cleanly separates the prior day's London+NY price action from the
  subsequent Asian+London+NY session where the signal fires.

  Prior session (for PDH/PDL):  [session_start - 24h, session_start)
  Current session (for signal):  [session_start,  session_start + ~23h)

Signal window within the current session:
  signal_offset_hours       : hours after session_start to begin scanning.
                              Default 14 = 07:00 UTC (London open).
  signal_window_end_hours   : hours after session_start to stop scanning.
                              Default 23 = 16:00 UTC (NY afternoon).
  time_exit_hours           : hard position exit from session_start.
                              Default 23.5 = 16:30 UTC.

Signal (wick rejection, single candle):
  SHORT fade: candle.high > PDH  AND  candle.close < PDH
              overshoot >= min_overshoot_pct  (wick is meaningful, not 1 tick)
  LONG  fade: candle.low  < PDL  AND  candle.close > PDL
              overshoot >= min_overshoot_pct

Stop: beyond the wick extreme (failed rejection invalidation).
  SHORT: candle.high + stop_buffer
  LONG : candle.low  - stop_buffer

Take-profit: opposite prior day boundary (full-range structural target).
  SHORT: PDL  (price expected to return all the way to prior day's low)
  LONG : PDH  (price expected to return all the way to prior day's high)

Rationale for full-range TP:
  The prior day breakout backtest showed 66% of prior-day-level breaks fail.
  If a wick sweeps PDH and rejects, the hypothesis is that the whole breakout
  attempt failed and price should revert into the prior range.  PDL is the
  natural structural target for a complete reversal.  This produces R:R of
  roughly 3:1 to 10:1 (stop = small overshoot; TP = full prior day range).
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PDFadeParams:
    # ── Candle timeframe ──────────────────────────────────────────────────────
    candle_minutes: int = 15

    # ── Session definition (17:00 UTC close convention) ───────────────────────
    # Hours after session_start (17:00 UTC) when the signal window opens.
    # 14h = 07:00 UTC the following calendar day (London open).
    signal_offset_hours: float = 14.0
    # Hours after session_start when the signal window closes.
    # 23h = 16:00 UTC the following calendar day.
    signal_window_end_hours: float = 23.0
    # Hours after session_start for the hard time exit.
    time_exit_hours: float = 23.5

    # ── Prior day range filter ────────────────────────────────────────────────
    # Skip days where the prior session range is outside these bounds.
    min_pdr_pct: float = 0.0030    # 0.30% of price  (~$6 on $2000 gold)
    max_pdr_pct: float = 0.0200    # 2.00% of price  (~$40) — spike/news days

    # ── Wick overshoot filter ─────────────────────────────────────────────────
    # The wick must pierce the prior day boundary by at least this fraction.
    # Filters trivial 1-tick sweeps.
    # 0.02% on $2000 gold ≈ $0.40
    min_overshoot_pct: float = 0.0002

    # ── Execution costs ───────────────────────────────────────────────────────
    spread_price: float = 0.30
    slippage_price: float = 0.20

    # ── Stop buffer beyond wick extreme ───────────────────────────────────────
    stop_buffer_floor_pct: float = 0.0003

    # ── Risk ─────────────────────────────────────────────────────────────────
    risk_pct: float = 0.01
    initial_equity: float = 100_000.0

    # ── Minimum candles for a valid session ───────────────────────────────────
    min_session_candles: int = 8    # lower than daily min; sessions can start thin

    def validate(self) -> None:
        errors: list[str] = []

        if self.candle_minutes not in (1, 5, 15, 30, 60):
            errors.append(f"candle_minutes={self.candle_minutes} not a standard timeframe")

        if not (0.0 <= self.signal_offset_hours < self.signal_window_end_hours < self.time_exit_hours):
            errors.append(
                f"session time order invalid: "
                f"signal_offset={self.signal_offset_hours}h  "
                f"window_end={self.signal_window_end_hours}h  "
                f"time_exit={self.time_exit_hours}h"
            )

        if self.time_exit_hours > 24.0:
            errors.append(f"time_exit_hours={self.time_exit_hours} exceeds one session (24h)")

        if not (0.0 < self.min_pdr_pct < self.max_pdr_pct < 1.0):
            errors.append(
                f"PDR filter order invalid: "
                f"0 < min({self.min_pdr_pct}) < max({self.max_pdr_pct}) < 1"
            )

        if self.min_overshoot_pct < 0.0:
            errors.append(f"min_overshoot_pct must be >= 0, got {self.min_overshoot_pct}")

        if self.spread_price <= 0.0:
            errors.append(f"spread_price must be > 0, got {self.spread_price}")

        if self.slippage_price < 0.0:
            errors.append(f"slippage_price must be >= 0, got {self.slippage_price}")

        if not (0.0 < self.risk_pct <= 0.05):
            errors.append(f"risk_pct={self.risk_pct} outside safe range (0, 0.05]")

        if self.initial_equity <= 0.0:
            errors.append(f"initial_equity must be > 0, got {self.initial_equity}")

        if errors:
            raise ValueError(
                "PDFadeParams validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info(
            "[PDF PARAMS] Validated. "
            "signal=%02.0fh-%02.0fh after 17:00 UTC  exit=%02.1fh  "
            "pdr=[%.2f%%, %.2f%%]  overshoot>=%.3f%%  "
            "spread=$%.2f  slippage=$%.2f  risk=%.1f%%",
            self.signal_offset_hours, self.signal_window_end_hours,
            self.time_exit_hours,
            self.min_pdr_pct * 100, self.max_pdr_pct * 100,
            self.min_overshoot_pct * 100,
            self.spread_price, self.slippage_price,
            self.risk_pct * 100,
        )
