"""
Parameters for the Asia–London Session Reversal strategy.

All time offsets are relative to the 17:00 UTC session boundary.

Session map (from 17:00 UTC start):
  +  0h 00m  → 17:00 UTC  Asian session opens
  + 13h 45m  → 06:45 UTC  Last Asian candle close captured
  + 14h 00m  → 07:00 UTC  London open — entry candle
  + 18h 00m  → 11:00 UTC  Hard time exit

No parameters are tunable knobs:
  - min_asian_move_pct is the only threshold; its sensitivity is tested, not optimised.
  - stop_buffer_pct is a fixed cost buffer, not a signal parameter.
  - All session times follow the CME/NYMEX 17:00 UTC convention.
"""

from dataclasses import dataclass


@dataclass
class ALParams:
    candle_minutes: int   = 15

    # ── Session time offsets (hours from 17:00 UTC session start) ─────────────
    asian_end_hours: float        = 13.75   # 06:45 UTC: last Asian candle
    london_open_hours: float      = 14.0    # 07:00 UTC: entry
    london_exit_hours: float      = 18.0    # 11:00 UTC: time exit

    # ── Signal filter ─────────────────────────────────────────────────────────
    min_asian_move_pct: float     = 0.0020  # 0.20% net Asian session move required

    # ── Execution costs ───────────────────────────────────────────────────────
    spread_price: float           = 0.30    # full spread in USD per oz
    slippage_price: float         = 0.20    # one-way slippage in USD per oz
    stop_buffer_pct: float        = 0.0003  # buffer beyond Asian extreme (~$0.60 at $2000)

    # ── Risk ──────────────────────────────────────────────────────────────────
    risk_pct: float               = 0.01    # 1% equity per trade
    initial_equity: float         = 100_000.0

    # ── Data quality gate ────────────────────────────────────────────────────
    min_asian_candles: int        = 30      # ~7.5 hours minimum; blocks holiday sessions

    @property
    def min_session_candles(self) -> int:
        """Alias for compatibility with shared result utilities."""
        return self.min_asian_candles

    def validate(self) -> None:
        assert self.candle_minutes > 0
        assert 0.0 < self.min_asian_move_pct < 0.10
        assert self.spread_price >= 0.0
        assert self.slippage_price >= 0.0
        assert 0.0 < self.risk_pct <= 0.05
        assert self.initial_equity > 0.0
        assert self.min_asian_candles > 0
