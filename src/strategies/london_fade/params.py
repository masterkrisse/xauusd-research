"""
Parameters for the London False Breakout Fade strategy.

Structural differences from the breakout strategy:
  - No tp_r_multiplier: take-profit is at the range midpoint (structural, not parametric).
  - Adds min_overshoot_pct: wick must pierce beyond the range by at least this amount
    to qualify as a meaningful breakout attempt worth fading.
  - Stop is placed beyond the breakout wick extreme, not the opposite range boundary.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FadeParams:
    # ── Candle timeframe ──────────────────────────────────────────────────────
    candle_minutes: int = 15

    # ── Session ───────────────────────────────────────────────────────────────
    london_window_duration_hours: float = 2.0
    time_exit_hours_after_london_open: float = 5.0

    # ── Asian range filters (same logic as breakout strategy) ─────────────────
    min_range_pct: float = 0.0015
    max_range_pct: float = 0.0080

    # ── Fade-specific signal filter ───────────────────────────────────────────
    # Minimum distance the wick must penetrate beyond the Asian range boundary
    # to qualify as a meaningful false breakout.  Filters trivial 1-tick wicks.
    # 0.05% on $2000 gold ≈ $1.
    min_overshoot_pct: float = 0.0005

    # ── Execution costs ───────────────────────────────────────────────────────
    spread_price: float = 0.30
    slippage_price: float = 0.20

    # ── Stop buffer ───────────────────────────────────────────────────────────
    # Buffer beyond the wick extreme for stop placement.
    # Same derivation as breakout strategy.
    stop_buffer_floor_pct: float = 0.0005

    # ── Risk ─────────────────────────────────────────────────────────────────
    risk_pct: float = 0.01
    initial_equity: float = 100_000.0

    # ── Minimum candles for a valid day ──────────────────────────────────────
    min_day_candles: int = 20

    def validate(self) -> None:
        errors: list[str] = []

        if self.candle_minutes not in (1, 5, 15, 30, 60):
            errors.append(f"candle_minutes={self.candle_minutes} not a standard timeframe")

        if not (0.0 < self.min_range_pct < self.max_range_pct < 1.0):
            errors.append(
                f"Range filter order invalid: "
                f"0 < min({self.min_range_pct}) < max({self.max_range_pct}) < 1 must hold"
            )

        if self.spread_price <= 0.0:
            errors.append(f"spread_price must be > 0, got {self.spread_price}")

        if self.slippage_price < 0.0:
            errors.append(f"slippage_price must be >= 0, got {self.slippage_price}")

        if self.min_overshoot_pct < 0.0:
            errors.append(f"min_overshoot_pct must be >= 0, got {self.min_overshoot_pct}")

        if not (0.0 < self.risk_pct <= 0.05):
            errors.append(f"risk_pct={self.risk_pct} outside safe range (0, 0.05]")

        if self.initial_equity <= 0.0:
            errors.append(f"initial_equity must be > 0, got {self.initial_equity}")

        if self.time_exit_hours_after_london_open <= self.london_window_duration_hours:
            errors.append(
                f"time_exit ({self.time_exit_hours_after_london_open}h) must be after "
                f"signal window ({self.london_window_duration_hours}h)"
            )

        if errors:
            raise ValueError(
                "FadeParams validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info(
            "[FADE PARAMS] Validated. range=[%.3f%%, %.3f%%]  overshoot>=%.3f%%  "
            "spread=$%.2f  slippage=$%.2f  risk=%.1f%%",
            self.min_range_pct * 100, self.max_range_pct * 100,
            self.min_overshoot_pct * 100,
            self.spread_price, self.slippage_price,
            self.risk_pct * 100,
        )
