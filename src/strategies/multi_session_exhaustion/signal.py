"""
Session indexing and signal detection for Multi-Session Trend Exhaustion.

Session index:
  One entry per 17:00 UTC boundary.  Each entry stores the session's close,
  high, and low — the three statistics needed to generate and price signals.

  session_close = close of the last 15-min bar strictly before the next 17:00 UTC
  session_high  = max(high)  of all bars in [session_start, next_session_start)
  session_low   = min(low)   of all bars in [session_start, next_session_start)

Signal (N=3 default):
  Examine the most-recent N session closes.
  If close[i] > close[i-1] for all i in [1..N] → SHORT (three consecutive up-closes)
  If close[i] < close[i-1] for all i in [1..N] → LONG  (three consecutive down-closes)
  Otherwise → no signal.

Trade construction:
  entry_ts  = first candle at or after the next 17:00 UTC session boundary
  tp_level  = close of the session two periods ago (S[-2])   — retrace target
  sl_level  = S[-1].high + buffer (SHORT) | S[-1].low - buffer (LONG)

Validation:
  - tp_level must be on the correct side of entry_price after costs.
  - If tp_level == sl_level or stop_distance <= 0, skip.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .params import MSEParams

logger = logging.getLogger(__name__)

_SESSION_HOUR_UTC = 17


@dataclass
class SessionInfo:
    session_start: pd.Timestamp   # 17:00 UTC (inclusive)
    close: float
    high: float
    low: float


@dataclass
class MSESignal:
    direction: int                # +1 LONG, -1 SHORT
    session_start: pd.Timestamp   # 17:00 UTC entry session (the NEW session)
    s_minus1: SessionInfo         # most recent closed session
    s_minus2: SessionInfo         # two sessions ago
    tp_level: float               # = s_minus2.close
    sl_level_raw: float           # stop before buffer — for logging


def build_session_index(df: pd.DataFrame) -> List[SessionInfo]:
    """
    Build a chronological list of SessionInfo for all 17:00 UTC sessions in df.

    Each session covers [session_start, next_session_start).
    """
    dates   = sorted(df.index.normalize().unique())
    sessions: List[SessionInfo] = []

    for d in dates:
        s_start = d + pd.Timedelta(hours=_SESSION_HOUR_UTC)
        s_end   = s_start + pd.Timedelta(hours=24)

        # Candles strictly within this session
        mask   = (df.index >= s_start) & (df.index < s_end)
        candles = df[mask]
        if candles.empty:
            continue

        # Session close = last bar's close
        session_close = float(candles.iloc[-1]["close"])
        session_high  = float(candles["high"].max())
        session_low   = float(candles["low"].min())

        sessions.append(SessionInfo(
            session_start=s_start,
            close=session_close,
            high=session_high,
            low=session_low,
        ))

    return sessions


def detect_mse_signal(
    sessions: List[SessionInfo],
    idx: int,
    params: MSEParams,
) -> Optional[MSESignal]:
    """
    Check for an N-consecutive exhaustion signal at position idx in the session list.

    sessions[idx] is the session that JUST closed (S[-1]).
    The signal, if any, is to be entered at sessions[idx].session_start + 24h
    (i.e., the START of the next session, idx+1).

    Args:
        sessions: full session index list
        idx:      index of the most-recently-closed session (S[-1])
        params:   strategy parameters

    Returns:
        MSESignal with the entry session timestamp, or None.
    """
    n = params.n_sessions
    if idx < n - 1:
        return None   # not enough history

    # Gather the closes of the N sessions ending at idx
    window = [sessions[idx - (n - 1 - i)] for i in range(n)]
    # window[0] = S[-(n)], window[-1] = S[-1]
    closes = [s.close for s in window]

    all_up   = all(closes[i] > closes[i - 1] for i in range(1, n))
    all_down = all(closes[i] < closes[i - 1] for i in range(1, n))

    if not all_up and not all_down:
        return None

    direction = -1 if all_up else 1
    s_minus1  = window[-1]   # most recent closed
    s_minus2  = window[-2]   # two sessions ago

    tp_level = s_minus2.close   # retrace to this level

    logger.info(
        "[MSE SIGNAL] %s | %s | Closes: %s → %.2f | S[-2].close=%.2f (TP) | "
        "S[-1].H=%.2f L=%.2f (stop basis)",
        s_minus1.session_start.strftime("%Y-%m-%d"),
        "SHORT" if direction == -1 else "LONG",
        " > ".join(f"{c:.2f}" for c in closes) if direction == -1
            else " < ".join(f"{c:.2f}" for c in closes),
        s_minus1.close,
        tp_level,
        s_minus1.high, s_minus1.low,
    )

    return MSESignal(
        direction=direction,
        session_start=s_minus1.session_start + pd.Timedelta(hours=24),
        s_minus1=s_minus1,
        s_minus2=s_minus2,
        tp_level=tp_level,
        sl_level_raw=(s_minus1.high if direction == -1 else s_minus1.low),
    )
