import numpy as np
import pandas as pd
import pytest

from algotrade.backtest.costs import ZERO_COSTS, CostModel
from algotrade.backtest.engine import run_backtest


def price_bars(closes, funding=None) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="1D", tz="UTC")
    frame = pd.DataFrame({"close": closes}, index=index, dtype="float64")
    frame["funding_rate"] = 0.0 if funding is None else funding
    return frame


def test_position_applies_from_next_bar() -> None:
    bars = price_bars([100, 110, 121])
    target = pd.Series([1.0, 0.0, 0.0], index=bars.index)
    ledger = run_backtest(bars, target, ZERO_COSTS).ledger
    # Decided at bar 0 close, so it earns bar 1's +10% and nothing else.
    assert ledger["position"].tolist() == [0.0, 1.0, 0.0]
    assert ledger["net"].tolist() == pytest.approx([0.0, 0.10, 0.0])


def test_buy_and_hold_without_costs_tracks_price(bars) -> None:
    result = run_backtest(bars, pd.Series(1.0, index=bars.index), ZERO_COSTS)
    expected = bars["close"].iloc[-1] / bars["close"].iloc[0]
    assert result.equity.iloc[-1] == pytest.approx(expected)


def test_turnover_is_charged_on_every_change() -> None:
    bars = price_bars([100] * 5)
    costs = CostModel(fee_bps=5, slippage_bps=5, include_funding=False)
    target = pd.Series([1.0, -1.0, -1.0, 0.0, 0.0], index=bars.index)
    ledger = run_backtest(bars, target, costs).ledger
    # enter long (1), reverse to short (2), flatten (1)
    assert ledger["trading_cost"].tolist() == pytest.approx([0, 0.001, 0.002, 0, 0.001])


def test_longs_pay_and_shorts_receive_positive_funding() -> None:
    bars = price_bars([100] * 3, funding=[0.0, 0.001, 0.001])
    long = run_backtest(bars, pd.Series(1.0, index=bars.index), CostModel(0, 0)).ledger
    short = run_backtest(bars, pd.Series(-1.0, index=bars.index), CostModel(0, 0)).ledger
    assert long["net"].sum() == pytest.approx(-0.002)
    assert short["net"].sum() == pytest.approx(0.002)


def test_target_is_clipped_to_max_leverage() -> None:
    bars = price_bars([100, 110])
    ledger = run_backtest(bars, pd.Series([3.0, 3.0], index=bars.index), ZERO_COSTS).ledger
    assert ledger["position"].max() == 1.0


def test_trade_returns_add_up_to_total_pnl(bars) -> None:
    rng = np.random.default_rng(1)
    target = pd.Series(rng.choice([-1.0, 0.0, 1.0], len(bars)), index=bars.index)
    target.iloc[-2:] = 0.0  # make sure every trade is closed
    bars = bars.assign(funding_rate=rng.normal(0, 1e-4, len(bars)))
    result = run_backtest(bars, target, CostModel(fee_bps=5, slippage_bps=3))
    assert not result.trades["open"].any()
    assert result.trades["return"].sum() == pytest.approx(result.ledger["net"].sum())


def test_trade_list_shape() -> None:
    bars = price_bars([100, 100, 110, 110, 99, 99])
    target = pd.Series([1.0, 1.0, -1.0, -1.0, 0.0, 0.0], index=bars.index)
    trades = run_backtest(bars, target, ZERO_COSTS).trades
    assert trades["direction"].tolist() == [1, -1]
    assert trades["bars"].tolist() == [2, 2]
    assert trades["return"].tolist() == pytest.approx([0.10, 0.10])
    assert trades["entry"].iloc[0] == bars.index[1]
    assert trades["exit"].iloc[0] == bars.index[3]


def test_account_is_liquidated_at_zero_equity() -> None:
    bars = price_bars([100, 100, 250, 100, 50])
    result = run_backtest(bars, pd.Series(-1.0, index=bars.index), ZERO_COSTS)
    # Short into a +150% bar loses everything; nothing after that can bring it back.
    assert result.equity.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert result.ledger["position"].iloc[3:].eq(0).all()
    assert result.meta["ruined_at"] == bars.index[2]
