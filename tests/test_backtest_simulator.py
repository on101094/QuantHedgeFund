"""
Tests for the backtest portfolio simulator (qsresearch.backtest.run_backtest).
"""

import numpy as np
import pandas as pd
import pytest

from qsresearch.backtest.run_backtest import BacktestDataError, _simulate_portfolio
from qsresearch.strategies.factor.algorithms import use_factor_as_signal


def _prices_from_returns(daily_returns: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a long price frame from per-symbol constant or per-day returns."""
    frames = []
    for symbol, rets in daily_returns.items():
        rets = np.broadcast_to(np.asarray(rets, dtype=float), len(dates)).copy()
        rets[0] = 0.0
        close = 100.0 * np.cumprod(1.0 + rets)
        frames.append(pd.DataFrame({"date": dates, "symbol": symbol, "close": close}))
    return pd.concat(frames, ignore_index=True)


class TestSignalWeights:
    """The simulator must hold the signal weights, not the universe average."""

    def test_top_n_differs_from_universe_average(self):
        dates = pd.bdate_range("2024-01-01", periods=250)
        winners = {f"W{i}": 0.003 for i in range(3)}
        losers = {f"L{i}": -0.001 for i in range(7)}
        prices = _prices_from_returns({**winners, **losers}, dates)
        prices["factor"] = prices["symbol"].str.startswith("W").astype(float)

        top_n = use_factor_as_signal(prices, factor_column="factor", top_n=3)
        universe = prices[["date", "symbol"]].assign(weight=1.0 / 10)

        kwargs = dict(
            capital_base=1_000_000,
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            transaction_cost_bps=0.0,
        )
        top_perf = _simulate_portfolio(top_n, prices, **kwargs)
        universe_perf = _simulate_portfolio(universe, prices, **kwargs)

        assert set(top_n["symbol"]) == set(winners)
        # Held from day 0's close, so it earns every return from day 1 on
        expected_top = 1_000_000 * 1.003 ** (len(dates) - 1)
        assert top_perf["portfolio_value"].iloc[-1] == pytest.approx(expected_top, rel=1e-9)
        assert universe_perf["portfolio_value"].iloc[-1] < 1_000_000 * 1.001 ** len(dates)
        assert top_perf["portfolio_value"].iloc[-1] > 1.5 * universe_perf["portfolio_value"].iloc[-1]

    def test_cash_when_weights_sum_below_one(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = _prices_from_returns({"A": 0.01}, dates)
        signals = pd.DataFrame({"date": [dates[0]], "symbol": ["A"], "weight": [0.5]})

        perf = _simulate_portfolio(
            signals, prices, 100.0, str(dates[0].date()), str(dates[-1].date()),
            transaction_cost_bps=0.0,
        )

        # Half invested: day-1 return is exactly half the asset's return
        assert perf["returns"].iloc[1] == pytest.approx(0.005)


class TestNoLookAhead:
    """A weight decided on day t must earn day t+1's return, not day t's."""

    def test_weight_on_day_t_uses_day_t_plus_1_return(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        t = 2
        a_returns = np.zeros(len(dates))
        a_returns[t] = 0.10      # move on the signal day itself: must NOT be earned
        a_returns[t + 1] = 0.05  # move on the next day: must be earned
        prices = _prices_from_returns({"A": a_returns, "B": 0.0}, dates)
        signals = pd.DataFrame({"date": [dates[t]], "symbol": ["A"], "weight": [1.0]})

        perf = _simulate_portfolio(
            signals, prices, 100.0, str(dates[0].date()), str(dates[-1].date()),
            rebalance_frequency="D", transaction_cost_bps=0.0,
        )
        returns = perf.set_index("date")["returns"]

        assert returns[dates[t]] == pytest.approx(0.0)
        assert returns[dates[t + 1]] == pytest.approx(0.05)
        assert perf["portfolio_value"].iloc[-1] == pytest.approx(105.0)


class TestRebalancingAndCosts:
    def test_monthly_rebalance_ignores_mid_month_signals(self):
        dates = pd.bdate_range("2024-01-01", "2024-02-29")
        prices = _prices_from_returns({"A": 0.001, "B": -0.001}, dates)
        # Signal flips to B mid-January; monthly rebalancing must keep A until February
        signals = pd.DataFrame({
            "date": dates,
            "symbol": ["A" if d < pd.Timestamp("2024-01-15") else "B" for d in dates],
            "weight": 1.0,
        })

        perf = _simulate_portfolio(
            signals, prices, 100.0, "2024-01-01", "2024-02-29",
            rebalance_frequency="M", transaction_cost_bps=0.0,
        ).set_index("date")

        traded = perf.index[perf["turnover"] > 0]
        assert list(traded) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
        assert perf.loc["2024-01-31", "returns"] == pytest.approx(0.001)
        assert perf.loc["2024-02-02", "returns"] == pytest.approx(-0.001)

    def test_cost_charged_on_turnover(self):
        dates = pd.bdate_range("2024-01-01", "2024-02-29")
        prices = _prices_from_returns({"A": 0.0, "B": 0.0}, dates)
        signals = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
            "symbol": ["A", "B"],
            "weight": [1.0, 1.0],
        })

        perf = _simulate_portfolio(
            signals, prices, 100.0, "2024-01-01", "2024-02-29",
            rebalance_frequency="M", transaction_cost_bps=10.0,
        ).set_index("date")

        # Buy A (turnover 1), then switch A -> B (turnover 2)
        assert perf.loc["2024-01-01", "turnover"] == pytest.approx(1.0)
        assert perf.loc["2024-02-01", "turnover"] == pytest.approx(2.0)
        assert perf["portfolio_value"].iloc[-1] == pytest.approx(100.0 * (1 - 0.001) * (1 - 0.002))


class TestMissingData:
    """Missing data must raise, never produce a synthetic curve."""

    def _signals(self):
        return pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "symbol": ["A"], "weight": [1.0]})

    def test_empty_prices_raise(self):
        empty = pd.DataFrame(columns=["date", "symbol", "close"])
        with pytest.raises(BacktestDataError):
            _simulate_portfolio(self._signals(), empty, 1_000_000, "2024-01-01", "2024-12-31")

    def test_missing_price_columns_raise(self):
        prices = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "symbol": ["A"]})
        with pytest.raises(BacktestDataError, match="close"):
            _simulate_portfolio(self._signals(), prices, 1_000_000, "2024-01-01", "2024-12-31")

    def test_empty_signals_raise(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = _prices_from_returns({"A": 0.01}, dates)
        with pytest.raises(BacktestDataError):
            _simulate_portfolio(pd.DataFrame(), prices, 1_000_000, "2024-01-01", "2024-12-31")

    def test_no_prices_in_window_raise(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = _prices_from_returns({"A": 0.01}, dates)
        with pytest.raises(BacktestDataError):
            _simulate_portfolio(self._signals(), prices, 1_000_000, "2024-01-01", "2024-12-31")

    def test_random_fallback_is_gone(self):
        from qsresearch.backtest import run_backtest as module

        assert not hasattr(module, "_fallback_simulation")
