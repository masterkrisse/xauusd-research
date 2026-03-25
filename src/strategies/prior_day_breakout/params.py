"""
Parameters for the Prior Day High/Low Breakout strategy.

Prior day (PD) is the completed UTC calendar day immediately before the
current trading date.  PD_HIGH and PD_LOW are computed from all 15-minute
candles whose timestamp falls in [00:00, 24:00) UTC of that day.

Signal window:
  A fixed UTC time window during which the signal is allowed to fire.
  Default: 07:00-19:45 UTC.  This spans London open through the bulk of
  the NY session and excludes the thin Asian night / Sunday open hours.
  No DST adjustment needed: the window is intentionally wide enough to
  cover London (07:00 or 08:00 UTC depending on BST) through NY afternoon.

Stop logic (failed breakout invalidation):
  Once price closes above PD_HIGH (long), the level becomes support.
  If price falls back below PD_HIGH, the breakout has failed.
  Stop: PD_HIGH - stop_buffer  (long)
        PD_LOW  + stop_buffer  (short)
  This is a tight stop relative to the trade distance — much tighter than
  using the full prior day range as the stop.

One trade per day maximum.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PDBreakoutParams:
    # ── Candle timeframe ──────────────────────────────────────────────────────
    candle_minutes: int = 15

    # ── Signal window (UTC, fixed — intentionally wide to avoid DST bugs) ─────
    signal_window_start_utc: int = 7    # hour, inclusive
    signal_window_end_utc: int = 20     # hour, exclusive (last candle at 19:45)

    # Hard time exit for open positions (UTC hour, same day)
    time_exit_utc_hour: int = 21

    # ── Prior day range filters ───────────────────────────────────────────────
    # Skip if prior day's range is abnormally tight (low-volatility / holiday)
    min_pdr_pct: float = 0.0030     # 0.30% on $2000 gold ≈ $6
    # Skip if prior day's range is abnormally wide (spike day — stop too large)
    max_pdr_pct: float = 0.0200     # 2.00% on $2000 gold ≈ $40

    # ── Execution costs ───────────────────────────────────────────────────────
    spread_price: float = 0.30
    slippage_price: float = 0.20

    # ── Stop: buffer beyond the breakout level ────────────────────────────────
    # Long stop : PD_HIGH - stop_buffer  (invalidation if price returns inside)
    # Short stop: PD_LOW  + stop_buffer
    stop_buffer_floor_pct: float = 0.0003   # 0.03% minimum buffer

    # ── Take-profit ───────────────────────────────────────────────────────────
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

        if not (0 <= self.signal_window_start_utc < self.signal_window_end_utc <= 23):
            errors.append(
                f"signal window invalid: "
                f"{self.signal_window_start_utc}:00-{self.signal_window_end_utc}:00 UTC"
            )

        if self.time_exit_utc_hour <= self.signal_window_end_utc:
            errors.append(
                f"time_exit_utc_hour={self.time_exit_utc_hour} must be after "
                f"signal_window_end_utc={self.signal_window_end_utc}"
            )

        if not (0.0 < self.min_pdr_pct < self.max_pdr_pct < 1.0):
            errors.append(
                f"PDR filter order invalid: "
                f"0 < min({self.min_pdr_pct}) < max({self.max_pdr_pct}) < 1"
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

        if errors:
            raise ValueError(
                "PDBreakoutParams validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info(
            "[PD PARAMS] Validated. signal=%02d:00-%02d:00 UTC  "
            "pdr=[%.2f%%, %.2f%%]  spread=$%.2f  slippage=$%.2f  tp=%.1fR  risk=%.1f%%",
            self.signal_window_start_utc, self.signal_window_end_utc,
            self.min_pdr_pct * 100, self.max_pdr_pct * 100,
            self.spread_price, self.slippage_price,
            self.tp_r_multiplier, self.risk_pct * 100,
        )
