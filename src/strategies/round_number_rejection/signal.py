"""
Signal detection for Round Number Intraday Rejection.

For each 15-minute candle in the active window (07:00–16:00 UTC):

  SHORT signal (rejection from above):
    candle.high >= (nearest $50 level) - touch_proximity_price
    AND candle.close <= (nearest $50 level) - min_rejection_price

  LONG signal (rejection from below):
    candle.low <= (nearest $50 level) + touch_proximity_price
    AND candle.close >= (nearest $50 level) + min_rejection_price

The "nearest $50 level" is evaluated independently for the high and low:
  - For SHORT: nearest level >= candle.high - touch_proximity_price
               (any $50 level the high wick was able to reach)
  - For LONG:  nearest level <= candle.low + touch_proximity_price

When both long and short conditions fire on the same candle (extremely rare —
would require a candle range straddling two $50 levels), neither signal is taken
to avoid ambiguity.

Only the first signal in the session's active window is used (one trade/session).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .params import RNParams

logger = logging.getLogger(__name__)


@dataclass
class RNSignal:
    direction: int            # +1 LONG (wick below level), -1 SHORT (wick above)
    round_level: float        # the $50 level that was tested
    wick_extreme: float       # candle high (SHORT) or candle low (LONG)
    close_price: float        # candle close
    touch_distance: float     # distance between wick and round level (USD)
    rejection_size: float     # distance between close and round level (USD)
    signal_candle_ts: pd.Timestamp


def nearest_level_above(price: float, spacing: float) -> float:
    """Smallest multiple of spacing that is >= price."""
    import math
    return math.ceil(price / spacing) * spacing


def nearest_level_below(price: float, spacing: float) -> float:
    """Largest multiple of spacing that is <= price."""
    import math
    return math.floor(price / spacing) * spacing


def detect_rn_signal(
    candle: pd.Series,
    ts: pd.Timestamp,
    params: RNParams,
) -> Optional[RNSignal]:
    """
    Inspect a single candle for a round-number rejection signal.

    Returns RNSignal or None.
    """
    h  = float(candle["high"])
    lo = float(candle["low"])
    c  = float(candle["close"])
    sp = params.level_spacing

    short_signal: Optional[RNSignal] = None
    long_signal:  Optional[RNSignal] = None

    # ── SHORT: wick above a round level, close below ──────────────────────────
    # Find the smallest $50 level that the high wick could have touched.
    level_short = nearest_level_above(h - params.touch_proximity_price, sp)
    touch_dist_short = level_short - h   # distance from high to level (should be small)
    reject_dist_short = level_short - c  # distance from close below the level

    if (touch_dist_short >= 0                             # level is above or at the high
            and touch_dist_short <= params.touch_proximity_price   # wick reached within threshold
            and reject_dist_short >= params.min_rejection_price):  # close is far enough below
        short_signal = RNSignal(
            direction=-1,
            round_level=level_short,
            wick_extreme=h,
            close_price=c,
            touch_distance=round(touch_dist_short, 4),
            rejection_size=round(reject_dist_short, 4),
            signal_candle_ts=ts,
        )

    # ── LONG: wick below a round level, close above ───────────────────────────
    level_long = nearest_level_below(lo + params.touch_proximity_price, sp)
    touch_dist_long  = lo - level_long   # distance from low to level (should be small, negative means below)
    reject_dist_long = c - level_long    # distance close is above the level

    # touch_dist_long: lo >= level_long - proximity  →  level_long - lo <= proximity
    wick_to_level_long = level_long - lo   # positive when low is below level

    if (wick_to_level_long >= 0                              # level is above or at the low
            and wick_to_level_long <= params.touch_proximity_price
            and reject_dist_long >= params.min_rejection_price):
        long_signal = RNSignal(
            direction=1,
            round_level=level_long,
            wick_extreme=lo,
            close_price=c,
            touch_distance=round(wick_to_level_long, 4),
            rejection_size=round(reject_dist_long, 4),
            signal_candle_ts=ts,
        )

    # ── Conflict: both fire on same candle ───────────────────────────────────
    if short_signal and long_signal:
        logger.debug(
            "[RN SIGNAL] Conflicting long+short on %s (candle spans two levels). Skip.",
            ts,
        )
        return None

    sig = short_signal or long_signal
    if sig:
        logger.info(
            "[RN SIGNAL] %s | %s | Level=%.0f | Wick=%.3f | Close=%.3f | "
            "Touch=%.3f | Reject=%.3f",
            ts.strftime("%Y-%m-%d %H:%M UTC"),
            "SHORT" if sig.direction == -1 else "LONG",
            sig.round_level, sig.wick_extreme, sig.close_price,
            sig.touch_distance, sig.rejection_size,
        )
    return sig
