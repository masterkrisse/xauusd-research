"""
Parameters for the NY Morning Initial Balance Breakout strategy.

Session logic:
  NY open is 09:30 America/New_York local time.
  In UTC this is 14:30 (EST, Nov-Mar) or 13:30 (EDT, Mar-Nov).
  We derive it correctly from zoneinfo, not hardcoded UTC offsets.

Initial balance (IB):
  First ib_duration_minutes of NY trading.
  Default: 30 minutes = 2 candles at 15-minute resolution.

Signal window:
  From IB close through ib_signal_window_hours after NY open.
  A close beyond IB high (long) or IB low (short) within this window triggers entry.

Time exit:
  time_exit_hours_after_ny_open after NY open.
  Must be after the signal window.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NYIBParams:
    # ── Candle timeframe ──────────────────────────────────────────────────────
    candle_minutes: int = 15

    # ── NY session ────────────────────────────────────────────────────────────
    # Duration of the initial balance period (from NY open)
    ib_duration_minutes: int = 30
    # How long after NY open to scan for breakout signals
    ib_signal_window_hours: float = 2.5
    # Hard time exit: close any open trade this many hours after NY open
    time_exit_hours_after_ny_open: float = 4.0

    # ── IB range filters ──────────────────────────────────────────────────────
    # Skip days where the IB is too tight (noise) or too wide (news spike, unfillable)
    min_ib_range_pct: float = 0.0008     # 0.08% of price ~ $1.60 on $2000 gold
    max_ib_range_pct: float = 0.0060     # 0.60% of price ~ $12.00

    # ── Execution costs ───────────────────────────────────────────────────────
    spread_price: float = 0.30
    slippage_price: float = 0.20

    # ── Stop buffer beyond IB boundary ───────────────────────────────────────
    # Applied on top of the IB low/high as stop placement
    stop_buffer_floor_pct: float = 0.0003

    # ── Take-profit ───────────────────────────────────────────────────────────
    # Fixed R multiple from entry (same as breakout strategy baseline)
    tp_r_multiplier: float = 1.5

    # ── Risk ─────────────────────────────────────────────────────────────────
    risk_pct: float = 0.01
    initial_equity: float = 100_000.0

    # ── Minimum candles for a valid day ──────────────────────────────────────
    min_day_candles: int = 20

    def validate(self) -> None:
        errors: list[str] = []

        if self.candle_minutes not in (1, 5, 15, 30, 60):
            errors.append(f"candle_minutes={self.candle_minutes} not a standard timeframe")

        if self.ib_duration_minutes < self.candle_minutes:
            errors.append(
                f"ib_duration_minutes={self.ib_duration_minutes} must be >= candle_minutes"
            )

        if self.ib_duration_minutes % self.candle_minutes != 0:
            errors.append(
                f"ib_duration_minutes={self.ib_duration_minutes} must be a multiple "
                f"of candle_minutes={self.candle_minutes}"
            )

        if not (0.0 < self.min_ib_range_pct < self.max_ib_range_pct < 1.0):
            errors.append(
                f"IB range filter order invalid: "
                f"0 < min({self.min_ib_range_pct}) < max({self.max_ib_range_pct}) < 1"
            )

        if self.spread_price <= 0.0:
            errors.append(f"spread_price must be > 0, got {self.spread_price}")

        if self.slippage_price < 0.0:
            errors.append(f"slippage_price must be >= 0, got {self.slippage_price}")

        if self.tp_r_multiplier <= 0.0:
            errors.append(f"tp_r_multiplier must be > 0, got {self.tp_r_multiplier}")

        if not (0.0 < self.risk_pct <= 0.05):
            errors.append(f"risk_pct={self.risk_pct} outside safe range (0, 0.05]")

        if self.initial_equity <= 0.0:
            errors.append(f"initial_equity must be > 0, got {self.initial_equity}")

        if self.time_exit_hours_after_ny_open <= self.ib_signal_window_hours:
            errors.append(
                f"time_exit ({self.time_exit_hours_after_ny_open}h) must be after "
                f"signal window ({self.ib_signal_window_hours}h)"
            )

        if errors:
            raise ValueError(
                "NYIBParams validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info(
            "[NY IB PARAMS] Validated. ib=%dmin  range=[%.3f%%, %.3f%%]  "
            "signal_window=%.1fh  time_exit=%.1fh  "
            "spread=$%.2f  slippage=$%.2f  tp=%.1fR  risk=%.1f%%",
            self.ib_duration_minutes,
            self.min_ib_range_pct * 100, self.max_ib_range_pct * 100,
            self.ib_signal_window_hours,
            self.time_exit_hours_after_ny_open,
            self.spread_price, self.slippage_price,
            self.tp_r_multiplier,
            self.risk_pct * 100,
        )
