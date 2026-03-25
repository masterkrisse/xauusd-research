"""
Strategy parameters for Asian Range → London Breakout (XAUUSD baseline v1.0).

All prices are in USD (XAUUSD spot, e.g. 2050.30).
All percentages are decimals (0.01 == 1%).
All times are in UTC.

Pip convention for XAUUSD:
  1 pip = $0.01 (minimum price increment)
  30 pips = $0.30 spread assumption
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    # ── Candle timeframe ──────────────────────────────────────────────────────
    candle_minutes: int = 15

    # ── Session configuration ─────────────────────────────────────────────────
    # London open is DST-dependent; see session.py for per-date resolution.
    # These values are the UTC offsets for the two regimes:
    #   london_open_winter_utc_hour = 8  (GMT == UTC, Oct–Mar)
    #   london_open_summer_utc_hour = 7  (BST == UTC+1, Mar–Oct)
    # Asian session always starts at 00:00 UTC.
    london_window_duration_hours: float = 2.0    # Hours after London open to accept signals
    time_exit_hours_after_london_open: float = 5.0  # Hard close (13:00 UTC in winter)

    # ── Range filters ─────────────────────────────────────────────────────────
    # Minimum: range must be large enough for positive expectancy after costs.
    # Derivation: range > (2 * effective_spread) / (TP_R - 1)
    # At spread=0.30, slippage=0.20, TP_R=1.5: required = (2*0.50)/(0.5) = $2.00
    # On $2000 gold: $2.00 / $2000 = 0.10%. Use 0.15% as conservative floor.
    min_range_pct: float = 0.0015   # 0.15% — reject if range too tight
    max_range_pct: float = 0.0080   # 0.80% — reject if already volatile

    # ── Execution costs ───────────────────────────────────────────────────────
    spread_price: float = 0.30      # $0.30 (30 pips) round-trip spread assumption
    slippage_price: float = 0.20    # $0.20 next-candle-open fill slippage

    # ── Stop buffer ───────────────────────────────────────────────────────────
    # Buffer beyond the Asian range boundary for stop placement.
    # Computed per-trade as: max(1.5 * spread_price, stop_buffer_floor_pct * entry)
    # This ensures the buffer is economically meaningful relative to spread.
    stop_buffer_floor_pct: float = 0.0005  # 0.05% of entry price as absolute floor

    # ── Take profit ───────────────────────────────────────────────────────────
    tp_r_multiplier: float = 1.5    # Target 1.5R from effective entry

    # ── Risk per trade ────────────────────────────────────────────────────────
    risk_pct: float = 0.01          # 1% of current equity per trade
    initial_equity: float = 100_000.0

    # ── Minimum candles for a valid day ──────────────────────────────────────
    # A full 15m session (00:00–23:45 UTC) has 96 candles.
    # Asian session alone (00:00–08:00) has 32 candles.
    # We require at least 20 to detect a valid range.
    min_day_candles: int = 20

    def validate(self) -> None:
        """
        Raise ValueError if any parameter combination is logically invalid.
        Log a warning for combinations that are legal but likely to produce
        poor results.
        """
        errors: list[str] = []

        if self.candle_minutes not in (1, 5, 15, 30, 60):
            errors.append(
                f"candle_minutes={self.candle_minutes} is not a recognised timeframe. "
                "Expected one of: 1, 5, 15, 30, 60."
            )

        if not (0.0 < self.min_range_pct < self.max_range_pct < 1.0):
            errors.append(
                f"Range filter order invalid: "
                f"0 < min_range_pct({self.min_range_pct}) "
                f"< max_range_pct({self.max_range_pct}) < 1 must hold."
            )

        if self.spread_price <= 0.0:
            errors.append(f"spread_price must be > 0, got {self.spread_price}")

        if self.slippage_price < 0.0:
            errors.append(f"slippage_price must be >= 0, got {self.slippage_price}")

        if self.tp_r_multiplier <= 0.0:
            errors.append(f"tp_r_multiplier must be > 0, got {self.tp_r_multiplier}")

        if self.tp_r_multiplier == 1.0:
            errors.append(
                "tp_r_multiplier == 1.0 means TP == breakeven after costs. "
                "This produces negative expectancy. Use a value > 1.0."
            )

        if not (0.0 < self.risk_pct <= 0.05):
            errors.append(
                f"risk_pct={self.risk_pct} is outside the safe research range (0, 0.05]. "
                "Values above 5% are dangerous for backtesting due to compounding effects."
            )

        if self.initial_equity <= 0.0:
            errors.append(f"initial_equity must be > 0, got {self.initial_equity}")

        if self.london_window_duration_hours <= 0.0:
            errors.append(
                f"london_window_duration_hours must be > 0, got "
                f"{self.london_window_duration_hours}"
            )

        if self.time_exit_hours_after_london_open <= self.london_window_duration_hours:
            errors.append(
                f"time_exit ({self.time_exit_hours_after_london_open}h after London open) "
                f"must be later than the signal window close "
                f"({self.london_window_duration_hours}h after London open)."
            )

        if self.min_day_candles < 10:
            errors.append(
                f"min_day_candles={self.min_day_candles} is too low to detect "
                "session anomalies reliably. Minimum recommended: 20."
            )

        if errors:
            raise ValueError(
                "StrategyParams validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # ── Warnings (do not raise) ───────────────────────────────────────────
        effective_spread = self.spread_price + self.slippage_price
        # For positive expectancy at TP_R on a minimum-size range:
        # win_amount = min_range * TP_R  |  loss_amount = min_range (approx)
        # Expected value > 0 when: TP_R * (1 - effective_spread/range) - 1 > 0
        # Conservative check at $2000 reference price
        ref_price = 2000.0
        min_range_dollars = self.min_range_pct * ref_price
        if self.tp_r_multiplier > 1.0:
            required_min_range = (2.0 * effective_spread) / (self.tp_r_multiplier - 1.0)
            if min_range_dollars < required_min_range:
                logger.warning(
                    "[PARAMS] min_range_pct may be too small for positive expectancy. "
                    "At $%s gold: min_range=$%.2f, required=$%.2f "
                    "(spread+slippage=%.2f, TP_R=%.1f). "
                    "Consider raising min_range_pct or lowering spread assumptions.",
                    ref_price, min_range_dollars, required_min_range,
                    effective_spread, self.tp_r_multiplier,
                )

        logger.info(
            "[PARAMS] Validation passed. "
            "candle=%dm  range=[%.3f%%, %.3f%%]  spread=$%.2f  slippage=$%.2f  "
            "TP_R=%.1f  risk=%.1f%%  equity=$%.0f",
            self.candle_minutes,
            self.min_range_pct * 100, self.max_range_pct * 100,
            self.spread_price, self.slippage_price,
            self.tp_r_multiplier,
            self.risk_pct * 100,
            self.initial_equity,
        )
