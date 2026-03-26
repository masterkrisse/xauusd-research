"""
Parameters for the Short-Only Macro-Filtered Fade strategy.

Two independent components, each with its own structural rationale:

  1. Macro filter (new):
       20-session rolling MA of session closes.
       Slope = MA(now) - MA(5 sessions ago).
       Slope < 0 → bearish macro regime → SHORT signals enabled.
       MA period 20 ≈ one calendar month (structural choice, not optimised).
       Slope lookback 5 ≈ one trading week (natural comparison unit).

  2. Prior-day fade SHORT signal (unchanged from prior_day_fade):
       PDH sweep-rejection: candle.high > PDH AND candle.close < PDH.
       Stop: wick extreme + buffer.
       TP: PDL (full prior-day range reversal).
       All prior_day_fade parameters are carried over identically.

No parameter is new relative to prior research — the macro filter constants
(20, 5) are structural and match widely-used institutional conventions.
The PDR thresholds and costs are unchanged.
"""

from dataclasses import dataclass


@dataclass
class SOFParams:
    candle_minutes: int = 15

    # ── Macro filter ──────────────────────────────────────────────────────────
    ma_period:       int   = 20     # sessions for rolling MA (~1 calendar month)
    slope_lookback:  int   = 5      # sessions for slope measurement (~1 week)

    # ── Prior-day range filter ─────────────────────────────────────────────────
    min_pdr_pct:     float = 0.0030  # minimum PDR as fraction of price
    max_pdr_pct:     float = 0.0200  # maximum PDR as fraction of price

    # ── Signal ────────────────────────────────────────────────────────────────
    signal_offset_hours:     float = 14.0   # signal window start: +14h = 07:00 UTC
    signal_window_end_hours: float = 23.0   # signal window end:   +23h = 16:00 UTC
    time_exit_hours:         float = 23.5   # hard time exit:      +23.5h = 16:30 UTC
    min_overshoot_pct:       float = 0.0002 # minimum wick overshoot

    # ── Execution ─────────────────────────────────────────────────────────────
    spread_price:            float = 0.30
    slippage_price:          float = 0.20
    stop_buffer_floor_pct:   float = 0.0003
    risk_pct:                float = 0.01
    initial_equity:          float = 100_000.0

    # ── Data quality gate ─────────────────────────────────────────────────────
    min_session_candles: int = 8

    def validate(self) -> None:
        assert self.ma_period >= 5
        assert self.slope_lookback >= 1
        assert self.slope_lookback < self.ma_period
        assert 0.0 < self.min_pdr_pct < self.max_pdr_pct < 0.10
        assert 0.0 < self.risk_pct <= 0.05
        assert self.initial_equity > 0
