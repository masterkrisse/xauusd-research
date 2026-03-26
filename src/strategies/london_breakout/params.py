"""
Parameters for the London Opening Range Breakout strategy.

Structural rationale:
  - London open (07:00 UTC) is the highest-volume transition in gold markets.
  - The first hour [07:00, 08:00) establishes a price range driven by initial
    order-flow positioning.
  - A close outside that range after 08:00 UTC signals directional commitment.
  - The hypothesis: institutional order flow at the London open creates directional
    persistence, not mean-reversion.

TP is structural: LOR_size * tp_lor_multiplier from the breakout level.
  - "Measured move": if the opening range was X wide, target X * multiplier beyond.
  - Adaptive to volatility — no fixed dollar target.

All time offsets are measured from 17:00 UTC session start.
  - +14h = 07:00 UTC (London open)
  - +15h = 08:00 UTC (end of LOR window)
  - +20h = 13:00 UTC (end of signal window)
  - +21h = 14:00 UTC (time exit — before NY afternoon)

No parameter here is optimised. Values reflect natural market structure:
  - LOR window 1 hour (standard ORB definition)
  - Signal window 5 hours (London session: 08:00–13:00 UTC)
  - tp_lor_multiplier = 1.0 (measured move, 1:1 extension of range)
  - min_lor_pct 0.10% (filter out thin Asian-carryover sessions)
  - max_lor_pct 1.50% (filter out extreme gap/news sessions)
"""

from dataclasses import dataclass


@dataclass
class LBParams:
    candle_minutes: int = 15

    # ── Opening Range window ──────────────────────────────────────────────────
    lor_start_hours:  float = 14.0   # +14h = 07:00 UTC (London open)
    lor_end_hours:    float = 15.0   # +15h = 08:00 UTC (end of LOR)
    min_lor_pct:      float = 0.0010 # 0.10% min LOR size (filter thin sessions)
    max_lor_pct:      float = 0.0150 # 1.50% max LOR size (filter extreme sessions)
    min_lor_candles:  int   = 3      # min candles needed in LOR window

    # ── Signal window (breakout detection) ───────────────────────────────────
    signal_start_hours: float = 15.0  # +15h = 08:00 UTC
    signal_end_hours:   float = 20.0  # +20h = 13:00 UTC

    # ── TP: measured move from breakout level ─────────────────────────────────
    tp_lor_multiplier: float = 1.0    # TP = breakout_level ± LOR_size * multiplier

    # ── Stop ──────────────────────────────────────────────────────────────────
    stop_buffer_floor_pct: float = 0.0003  # min stop buffer as fraction of price

    # ── Time exit ─────────────────────────────────────────────────────────────
    time_exit_hours: float = 21.0  # +21h = 14:00 UTC

    # ── Execution costs ───────────────────────────────────────────────────────
    spread_price:   float = 0.30
    slippage_price: float = 0.20

    # ── Risk management ───────────────────────────────────────────────────────
    risk_pct:        float = 0.01
    initial_equity:  float = 100_000.0

    def validate(self) -> None:
        assert self.lor_start_hours < self.lor_end_hours
        assert self.lor_end_hours <= self.signal_start_hours
        assert self.signal_start_hours < self.signal_end_hours
        assert self.signal_end_hours < self.time_exit_hours
        assert 0.0 < self.min_lor_pct < self.max_lor_pct < 0.10
        assert self.tp_lor_multiplier > 0.0
        assert 0.0 < self.risk_pct <= 0.05
        assert self.initial_equity > 0
