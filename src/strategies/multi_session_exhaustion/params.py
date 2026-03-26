"""
Parameters for the Multi-Session Trend Exhaustion strategy.

Session convention: 17:00 UTC boundary (CME/NYMEX metals).
  Session close = last 15-min candle close strictly before the next 17:00 UTC.
  Session high/low = rolling extremes of all candles within the session.

Signal:
  Three consecutive sessions all closing in the same direction.
  N = 3 is structural (behavioural hypothesis), not a tuned parameter.

Stop:
  Placed at the extreme of the most recent session (S[-1]):
    SHORT: stop = S[-1].high + stop_buffer_pct * entry_price
    LONG : stop = S[-1].low  - stop_buffer_pct * entry_price
  Rationale: if the trend resumes beyond the prior session's wick, exhaustion
  is disproven.

TP:
  S[-2].close — the session close from two sessions ago.
  For SHORT: this is below S[-1].close, capturing the retrace of the final leg.
  For LONG : this is above S[-1].close.
  Rationale: "price retraces to where the run was two sessions ago."

Time exit:
  Next 17:00 UTC (one full session).  The trade is closed at the open of the
  first candle at or after the next 17:00 UTC boundary.
"""

from dataclasses import dataclass


@dataclass
class MSEParams:
    candle_minutes: int  = 15
    n_sessions:     int  = 3             # consecutive closes — structural, not optimised

    # ── Stop buffer ───────────────────────────────────────────────────────────
    stop_buffer_pct: float = 0.0003      # fraction of entry price (~$0.60 at $2000)

    # ── Execution costs ───────────────────────────────────────────────────────
    spread_price:    float = 0.30
    slippage_price:  float = 0.20

    # ── Risk ──────────────────────────────────────────────────────────────────
    risk_pct:        float = 0.01
    initial_equity:  float = 100_000.0

    def validate(self) -> None:
        assert self.n_sessions >= 2
        assert 0.0 < self.risk_pct <= 0.05
        assert self.initial_equity > 0
        assert self.spread_price >= 0
        assert self.slippage_price >= 0
