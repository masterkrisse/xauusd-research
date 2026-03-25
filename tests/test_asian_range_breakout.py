"""
Unit tests for the Asian Range → London Breakout strategy components.

Covers:
  - DST detection (BST vs GMT)
  - Session boundary calculation
  - Asian range computation and filter blocking
  - Breakout signal detection (long, short, no signal, gap-open rejection)
  - Trade setup: stop/TP levels, position sizing, R consistency
  - Trade simulation: TP hit, SL hit, time exit, gap-through stop
  - Parameter validation
  - Results summary computation
"""

import math
from datetime import date, datetime, timezone, timedelta

import pandas as pd
import pytest

from src.strategies.asian_range_breakout.engine import _validate_dataframe
from src.strategies.asian_range_breakout.execution import (
    compute_trade_setup,
    simulate_trade,
    EXIT_TP,
    EXIT_SL,
    EXIT_SL_GAP,
    EXIT_TIME,
)
from src.strategies.asian_range_breakout.params import StrategyParams
from src.strategies.asian_range_breakout.results import compute_results
from src.strategies.asian_range_breakout.session import _is_bst, get_session_boundaries
from src.strategies.asian_range_breakout.signal import (
    AsianRange,
    BreakoutSignal,
    compute_asian_range,
    detect_breakout,
)

_UTC = timezone.utc

# ── Helpers ───────────────────────────────────────────────────────────────────

def _params(**overrides) -> StrategyParams:
    """Return default StrategyParams, with optional overrides."""
    p = StrategyParams()
    for k, v in overrides.items():
        object.__setattr__(p, k, v)
    return p


def _make_candles(
    timestamps: list,
    opens: list,
    highs: list,
    lows: list,
    closes: list,
) -> pd.DataFrame:
    """Build a UTC-aware 15m OHLC DataFrame from raw lists."""
    idx = pd.DatetimeIndex(timestamps, tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


def _asian_range_fixture(high: float = 2010.0, low: float = 2005.0) -> AsianRange:
    """Return a pre-built valid AsianRange for use in execution tests."""
    return AsianRange(
        high=high,
        low=low,
        range_price=high - low,
        range_pct=(high - low) / 2007.5,
        reference_price=2007.5,
        candle_count=30,
        valid=True,
        block_reason=None,
    )


# ── DST detection ─────────────────────────────────────────────────────────────

class TestBSTDetection:
    def test_winter_date_is_not_bst(self):
        # January is always GMT
        assert _is_bst(date(2023, 1, 15)) is False

    def test_summer_date_is_bst(self):
        # July is always BST
        assert _is_bst(date(2023, 7, 15)) is True

    def test_day_before_spring_forward_is_not_bst(self):
        # 2023 BST starts: last Sunday of March = March 26
        assert _is_bst(date(2023, 3, 25)) is False

    def test_day_of_spring_forward_is_bst(self):
        assert _is_bst(date(2023, 3, 26)) is True

    def test_day_of_fall_back_is_not_bst(self):
        # 2023 BST ends: last Sunday of October = October 29
        assert _is_bst(date(2023, 10, 29)) is False

    def test_day_before_fall_back_is_bst(self):
        assert _is_bst(date(2023, 10, 28)) is True


# ── Session boundaries ────────────────────────────────────────────────────────

class TestSessionBoundaries:
    def test_winter_london_open_is_0800_utc(self):
        s = get_session_boundaries(date(2023, 1, 15), 2.0, 5.0, 15)
        assert s.london_open.hour == 8
        assert s.london_open.minute == 0

    def test_summer_london_open_is_0700_utc(self):
        s = get_session_boundaries(date(2023, 7, 15), 2.0, 5.0, 15)
        assert s.london_open.hour == 7
        assert s.london_open.minute == 0

    def test_window_close_is_2h_after_london_open(self):
        s = get_session_boundaries(date(2023, 1, 15), 2.0, 5.0, 15)
        assert (s.london_window_close - s.london_open).total_seconds() == 2 * 3600

    def test_time_exit_is_5h_after_london_open_winter(self):
        s = get_session_boundaries(date(2023, 1, 15), 2.0, 5.0, 15)
        # 08:00 + 5h = 13:00 UTC
        assert s.time_exit.hour == 13

    def test_time_exit_is_5h_after_london_open_summer(self):
        s = get_session_boundaries(date(2023, 7, 15), 2.0, 5.0, 15)
        # 07:00 + 5h = 12:00 UTC
        assert s.time_exit.hour == 12

    def test_asia_start_is_midnight_utc(self):
        s = get_session_boundaries(date(2023, 3, 15), 2.0, 5.0, 15)
        assert s.asia_start.hour == 0
        assert s.asia_start.minute == 0

    def test_all_timestamps_are_utc_aware(self):
        s = get_session_boundaries(date(2023, 6, 1), 2.0, 5.0, 15)
        for attr in ("asia_start", "london_open", "london_window_close", "time_exit"):
            ts = getattr(s, attr)
            assert ts.tzinfo is not None, f"{attr} is not timezone-aware"


# ── Asian range ───────────────────────────────────────────────────────────────

class TestAsianRange:
    def _session(self, d: date = date(2023, 1, 15)):
        return get_session_boundaries(d, 2.0, 5.0, 15)

    def _make_asian_candles(self, high_val: float = 2010.0, low_val: float = 2002.0):
        """
        32 candles from 00:00 to 07:45 UTC (winter day).
        The range of the resulting DataFrame is [low_val, high_val].
        Most candles stay near the midpoint; one candle sets the high and one the low.
        """
        times = pd.date_range("2023-01-15 00:00", periods=32, freq="15min", tz="UTC")
        mid = (high_val + low_val) / 2.0
        opens  = [mid] * 32
        highs  = [mid + 0.01] * 32   # keep all other candles near midpoint
        lows   = [mid - 0.01] * 32
        closes = [mid] * 32
        highs[10] = high_val          # candle 10 sets the session high
        lows[20]  = low_val           # candle 20 sets the session low
        return _make_candles(times.tolist(), opens, highs, lows, closes)

    def test_valid_range_computed_correctly(self):
        candles = self._make_asian_candles(2012.0, 2001.0)
        session = self._session()
        params = _params()
        ar = compute_asian_range(candles, session, params)
        assert ar.valid is True
        assert ar.high == pytest.approx(2012.0)
        assert ar.low == pytest.approx(2001.0)
        assert ar.range_price == pytest.approx(11.0)

    def test_range_too_tight_is_blocked(self):
        # High and low are only $0.50 apart — below min_range_pct of 0.15% on $2000
        candles = self._make_asian_candles(2006.0, 2005.5)
        session = self._session()
        params = _params(min_range_pct=0.0015)
        ar = compute_asian_range(candles, session, params)
        assert ar.valid is False
        assert ar.block_reason is not None
        assert "tight" in ar.block_reason.lower()

    def test_range_too_wide_is_blocked(self):
        # Range of $30 on $2000 = 1.5%, above max of 0.80%
        candles = self._make_asian_candles(2030.0, 2000.0)
        session = self._session()
        params = _params(max_range_pct=0.0080)
        ar = compute_asian_range(candles, session, params)
        assert ar.valid is False
        assert "wide" in ar.block_reason.lower()

    def test_london_open_candle_excluded_from_asian_range(self):
        """Candle that opens at london_open (08:00) must NOT be in the Asian range."""
        session = self._session(date(2023, 1, 15))
        # 33 candles: 00:00 to 08:00 (inclusive)
        times = pd.date_range("2023-01-15 00:00", periods=33, freq="15min", tz="UTC")
        opens = [2005.0] * 33
        highs = [2010.0] * 32 + [2025.0]   # 08:00 candle has very high high
        lows = [2002.0] * 33
        closes = [2006.0] * 33
        candles = _make_candles(times.tolist(), opens, highs, lows, closes)

        params = _params()
        ar = compute_asian_range(candles, session, params)
        # Asian range high should NOT include the 2025 from the 08:00 candle
        assert ar.high == pytest.approx(2010.0), (
            f"Expected 2010.0 but got {ar.high}. "
            "The 08:00 candle (London open) was incorrectly included in the Asian range."
        )

    def test_no_candles_returns_invalid(self):
        session = self._session()
        empty = _make_candles([], [], [], [], [])
        ar = compute_asian_range(empty, session, _params())
        assert ar.valid is False


# ── Breakout detection ────────────────────────────────────────────────────────

class TestBreakoutDetection:
    def _ar(self, high: float = 2010.0, low: float = 2002.0) -> AsianRange:
        return _asian_range_fixture(high=high, low=low)

    def test_long_breakout_detected(self):
        ar = self._ar(high=2010.0, low=2002.0)
        # Candle opens at 2010.0 (at boundary), closes above
        candles = _make_candles(
            ["2023-01-15 08:00+00:00"],
            [2010.0], [2013.0], [2009.0], [2012.0],
        )
        signal = detect_breakout(candles, ar)
        assert signal is not None
        assert signal.direction == 1
        assert signal.breakout_level == pytest.approx(2010.0)

    def test_short_breakout_detected(self):
        ar = self._ar(high=2010.0, low=2002.0)
        candles = _make_candles(
            ["2023-01-15 08:00+00:00"],
            [2002.0], [2003.0], [1999.0], [2000.0],
        )
        signal = detect_breakout(candles, ar)
        assert signal is not None
        assert signal.direction == -1
        assert signal.breakout_level == pytest.approx(2002.0)

    def test_no_breakout_returns_none(self):
        ar = self._ar(high=2010.0, low=2002.0)
        # Candle stays within range
        candles = _make_candles(
            ["2023-01-15 08:00+00:00"],
            [2005.0], [2009.5], [2002.5], [2007.0],
        )
        assert detect_breakout(candles, ar) is None

    def test_gap_open_above_range_rejected_for_long(self):
        """Candle opens ABOVE ASIA_HIGH — does not qualify as a long signal."""
        ar = self._ar(high=2010.0, low=2002.0)
        candles = _make_candles(
            ["2023-01-15 08:00+00:00"],
            [2015.0],   # opens above ASIA_HIGH
            [2018.0], [2014.0], [2016.0],
        )
        assert detect_breakout(candles, ar) is None

    def test_gap_open_below_range_rejected_for_short(self):
        """Candle opens BELOW ASIA_LOW — does not qualify as a short signal."""
        ar = self._ar(high=2010.0, low=2002.0)
        candles = _make_candles(
            ["2023-01-15 08:00+00:00"],
            [1998.0],   # opens below ASIA_LOW
            [1999.5], [1996.0], [1997.0],
        )
        assert detect_breakout(candles, ar) is None

    def test_first_qualifying_candle_is_returned(self):
        """When two consecutive candles both qualify, only the first is returned."""
        ar = self._ar(high=2010.0, low=2002.0)
        candles = _make_candles(
            ["2023-01-15 08:00+00:00", "2023-01-15 08:15+00:00"],
            [2008.0, 2012.0],
            [2014.0, 2016.0],
            [2007.0, 2011.0],
            [2012.0, 2015.0],
        )
        signal = detect_breakout(candles, ar)
        assert signal is not None
        assert str(signal.signal_candle_ts) == "2023-01-15 08:00:00+00:00"


# ── Trade setup ───────────────────────────────────────────────────────────────

class TestTradeSetup:
    def _signal(self, direction: int = 1) -> BreakoutSignal:
        level = 2010.0 if direction == 1 else 2002.0
        return BreakoutSignal(
            direction=direction,
            breakout_level=level,
            signal_candle_ts=pd.Timestamp("2023-01-15 08:00", tz="UTC"),
            signal_candle_open=2008.0,
            signal_candle_close=2011.0 if direction == 1 else 2001.0,
        )

    def _entry_candle(self, open_price: float = 2011.5) -> pd.Series:
        return pd.Series(
            {"open": open_price, "high": open_price + 2, "low": open_price - 1, "close": open_price + 1}
        )

    def test_long_setup_entry_is_above_candle_open(self):
        params = _params(spread_price=0.30, slippage_price=0.20)
        signal = self._signal(direction=1)
        entry_candle = self._entry_candle(2011.5)
        ar = _asian_range_fixture(high=2010.0, low=2002.0)
        setup = compute_trade_setup(signal, entry_candle, pd.Timestamp("2023-01-15 08:15", tz="UTC"), ar, 100_000, params)
        assert setup is not None
        # Effective entry = open + half_spread + slippage = 2011.5 + 0.15 + 0.20 = 2011.85
        assert setup.entry_price == pytest.approx(2011.85, abs=1e-4)

    def test_long_sl_is_below_asia_low(self):
        params = _params(spread_price=0.30, slippage_price=0.20, stop_buffer_floor_pct=0.0)
        signal = self._signal(direction=1)
        ar = _asian_range_fixture(high=2010.0, low=2002.0)
        entry_candle = self._entry_candle(2011.5)
        setup = compute_trade_setup(signal, entry_candle, pd.Timestamp("2023-01-15 08:15", tz="UTC"), ar, 100_000, params)
        assert setup is not None
        # sl_gross = 2002.0 - 1 * max(1.5*0.30, 0) = 2002.0 - 0.45 = 2001.55
        assert setup.sl_gross < 2002.0

    def test_sl_hit_gives_minus_one_r(self):
        """A stop-loss exit should produce realized_r ≈ -1.0."""
        params = _params(spread_price=0.30, slippage_price=0.20)
        signal = self._signal(direction=1)
        ar = _asian_range_fixture(high=2010.0, low=2002.0)
        entry_candle = self._entry_candle(2011.5)
        entry_ts = pd.Timestamp("2023-01-15 08:15", tz="UTC")
        setup = compute_trade_setup(signal, entry_candle, entry_ts, ar, 100_000, params)
        assert setup is not None

        # Simulate a candle that hits the SL
        sl_candle_ts = pd.Timestamp("2023-01-15 08:30", tz="UTC")
        # Candle low must be <= sl_gross
        sl_candle = _make_candles(
            [sl_candle_ts],
            [setup.sl_gross + 0.10],        # opens above SL
            [setup.sl_gross + 0.30],
            [setup.sl_gross - 0.50],        # low breaches SL
            [setup.sl_gross + 0.05],
        )
        time_exit = pd.Timestamp("2023-01-15 13:00", tz="UTC")
        result = simulate_trade(setup, sl_candle, time_exit, 100_000, params)
        assert result.exit_reason == EXIT_SL
        assert result.realized_r == pytest.approx(-1.0, abs=0.01)

    def test_tp_hit_gives_tp_r(self):
        """A TP exit should produce realized_r ≈ tp_r_multiplier."""
        params = _params(spread_price=0.30, slippage_price=0.20, tp_r_multiplier=1.5)
        signal = self._signal(direction=1)
        ar = _asian_range_fixture(high=2010.0, low=2002.0)
        entry_candle = self._entry_candle(2011.5)
        entry_ts = pd.Timestamp("2023-01-15 08:15", tz="UTC")
        setup = compute_trade_setup(signal, entry_candle, entry_ts, ar, 100_000, params)
        assert setup is not None

        tp_candle_ts = pd.Timestamp("2023-01-15 08:30", tz="UTC")
        tp_candle = _make_candles(
            [tp_candle_ts],
            [setup.entry_price + 0.10],
            [setup.tp_gross + 0.50],        # high breaches TP
            [setup.entry_price],
            [setup.tp_gross + 0.30],
        )
        time_exit = pd.Timestamp("2023-01-15 13:00", tz="UTC")
        result = simulate_trade(setup, tp_candle, time_exit, 100_000, params)
        assert result.exit_reason == EXIT_TP
        assert result.realized_r == pytest.approx(1.5, abs=0.01)

    def test_time_exit_fires_before_sl_tp(self):
        params = _params(spread_price=0.30, slippage_price=0.20, tp_r_multiplier=1.5)
        signal = self._signal(direction=1)
        ar = _asian_range_fixture(high=2010.0, low=2002.0)
        entry_candle = self._entry_candle(2011.5)
        entry_ts = pd.Timestamp("2023-01-15 08:15", tz="UTC")
        setup = compute_trade_setup(signal, entry_candle, entry_ts, ar, 100_000, params)
        assert setup is not None

        # Candle that is at time_exit and also hits SL
        time_exit = pd.Timestamp("2023-01-15 13:00", tz="UTC")
        exit_candle = _make_candles(
            [time_exit],           # candle opens AT time_exit
            [setup.sl_gross - 1],  # also a gap-through stop
            [setup.sl_gross + 0.5],
            [setup.sl_gross - 2],
            [setup.sl_gross],
        )
        result = simulate_trade(setup, exit_candle, time_exit, 100_000, params)
        # TIME exit takes priority over SL in the same candle
        assert result.exit_reason == EXIT_TIME

    def test_gap_through_stop_produces_sl_gap(self):
        params = _params(spread_price=0.30, slippage_price=0.20)
        signal = self._signal(direction=1)
        ar = _asian_range_fixture(high=2010.0, low=2002.0)
        entry_candle = self._entry_candle(2011.5)
        entry_ts = pd.Timestamp("2023-01-15 08:15", tz="UTC")
        setup = compute_trade_setup(signal, entry_candle, entry_ts, ar, 100_000, params)
        assert setup is not None

        gap_ts = pd.Timestamp("2023-01-15 08:30", tz="UTC")
        # Candle opens BELOW sl_gross (gap-down through stop)
        gap_candle = _make_candles(
            [gap_ts],
            [setup.sl_gross - 2.0],   # gap open below SL
            [setup.sl_gross - 1.5],
            [setup.sl_gross - 3.0],
            [setup.sl_gross - 2.5],
        )
        time_exit = pd.Timestamp("2023-01-15 13:00", tz="UTC")
        result = simulate_trade(setup, gap_candle, time_exit, 100_000, params)
        assert result.exit_reason == EXIT_SL_GAP
        # Fill should be at gap open (worse than SL), so realized_r < -1
        assert result.realized_r < -1.0

    def test_position_size_risks_correct_dollar_amount(self):
        equity = 50_000.0
        params = _params(risk_pct=0.01)
        signal = self._signal(direction=1)
        ar = _asian_range_fixture(high=2010.0, low=2002.0)
        entry_candle = self._entry_candle(2011.5)
        entry_ts = pd.Timestamp("2023-01-15 08:15", tz="UTC")
        setup = compute_trade_setup(signal, entry_candle, entry_ts, ar, equity, params)
        assert setup is not None
        expected_risk = equity * params.risk_pct
        assert setup.risk_amount == pytest.approx(expected_risk, rel=1e-6)
        # Verify: position_size * stop_distance == risk_amount
        assert setup.position_size * setup.stop_distance == pytest.approx(expected_risk, rel=1e-4)


# ── Parameter validation ──────────────────────────────────────────────────────

class TestParamValidation:
    def test_default_params_are_valid(self):
        StrategyParams().validate()  # should not raise

    def test_inverted_range_filter_raises(self):
        with pytest.raises(ValueError, match="Range filter"):
            _params(min_range_pct=0.01, max_range_pct=0.005).validate()

    def test_zero_spread_raises(self):
        with pytest.raises(ValueError, match="spread_price"):
            _params(spread_price=0.0).validate()

    def test_excessive_risk_pct_raises(self):
        with pytest.raises(ValueError, match="risk_pct"):
            _params(risk_pct=0.10).validate()

    def test_tp_r_of_one_raises(self):
        with pytest.raises(ValueError, match="tp_r_multiplier"):
            _params(tp_r_multiplier=1.0).validate()

    def test_time_exit_before_window_raises(self):
        with pytest.raises(ValueError, match="time_exit"):
            _params(
                london_window_duration_hours=3.0,
                time_exit_hours_after_london_open=2.0,
            ).validate()


# ── DataFrame validation ──────────────────────────────────────────────────────

class TestDataFrameValidation:
    def _good_df(self) -> pd.DataFrame:
        times = pd.date_range("2023-01-15 00:00", periods=96, freq="15min", tz="UTC")
        n = len(times)
        return pd.DataFrame(
            {"open": [2005.0] * n, "high": [2010.0] * n,
             "low": [2000.0] * n, "close": [2007.0] * n},
            index=times,
        )

    def test_valid_df_does_not_raise(self):
        _validate_dataframe(self._good_df(), StrategyParams())

    def test_missing_column_raises(self):
        df = self._good_df().drop(columns=["close"])
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_dataframe(df, StrategyParams())

    def test_naive_index_raises(self):
        df = self._good_df()
        df.index = df.index.tz_localize(None)
        with pytest.raises(ValueError, match="timezone-aware"):
            _validate_dataframe(df, StrategyParams())

    def test_high_less_than_low_raises(self):
        df = self._good_df()
        df.loc[df.index[5], "high"] = 1990.0   # high < low
        with pytest.raises(ValueError, match="high < low"):
            _validate_dataframe(df, StrategyParams())


# ── Results summary ───────────────────────────────────────────────────────────

class TestResultsSummary:
    def _make_result(self, r: float, direction: int = 1) -> "TradeResult":
        """Build a minimal TradeResult for results testing."""
        from src.strategies.asian_range_breakout.execution import TradeResult, TradeSetup
        ar = _asian_range_fixture()
        setup = TradeSetup(
            direction=direction,
            entry_price=2010.0,
            sl_gross=2002.0,
            tp_gross=2022.0,
            effective_sl=2001.85,
            effective_tp=2021.85,
            stop_distance=8.15,
            position_size=12.27,
            risk_amount=100.0,
            entry_timestamp=pd.Timestamp("2023-01-15 08:15", tz="UTC"),
            signal_timestamp=pd.Timestamp("2023-01-15 08:00", tz="UTC"),
            asian_range=ar,
        )
        net_pnl = r * 100.0
        equity_after = 100_000.0 + net_pnl
        return TradeResult(
            setup=setup,
            exit_price=2010.0 + r * setup.stop_distance,
            exit_reason=EXIT_TP if r > 0 else EXIT_SL,
            exit_timestamp=pd.Timestamp("2023-01-15 10:00", tz="UTC"),
            net_pnl=net_pnl,
            realized_r=r,
            equity_after=equity_after,
        )

    def test_win_rate_computed_correctly(self):
        results = [self._make_result(1.5), self._make_result(-1.0), self._make_result(1.5)]
        summary = compute_results(results, StrategyParams())
        assert summary.win_rate == pytest.approx(2 / 3, abs=1e-4)

    def test_expectancy_positive_for_profitable_set(self):
        results = [self._make_result(1.5)] * 3 + [self._make_result(-1.0)] * 1
        summary = compute_results(results, StrategyParams())
        assert summary.expectancy_r > 0.0

    def test_empty_results_returns_summary_without_error(self):
        summary = compute_results([], StrategyParams())
        assert summary.total_trades == 0
        assert summary.passed_min_trade_count is False

    def test_passed_min_trade_count_requires_200(self):
        results = [self._make_result(0.5)] * 150
        summary = compute_results(results, StrategyParams())
        assert summary.passed_min_trade_count is False

        results_200 = [self._make_result(0.5)] * 200
        summary_200 = compute_results(results_200, StrategyParams())
        assert summary_200.passed_min_trade_count is True

    def test_direction_split_is_correct(self):
        longs = [self._make_result(1.5, direction=1)] * 3
        shorts = [self._make_result(-1.0, direction=-1)] * 2
        summary = compute_results(longs + shorts, StrategyParams())
        assert summary.long_summary.trade_count == 3
        assert summary.short_summary.trade_count == 2
        assert summary.long_summary.win_rate == pytest.approx(1.0)
        assert summary.short_summary.win_rate == pytest.approx(0.0)
