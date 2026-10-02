"""The NautilusTrader adapter must reproduce our engine and bracket simulator exactly.

Run on gap-free synthetic bars (each bar opens at the previous close) at zero costs, so that
NautilusTrader's fill-at-close and our fill-at-next-open coincide, and with stop/target
distances wide enough that no bar touches both (NautilusTrader walks bars open-high-low-close,
our simulator assumes the stop first).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nautilus_trader")

from algotrade.live.adapter import FILL_COLUMNS
from algotrade.live.parity import run
from algotrade.research.incubation import FILL_COLUMNS as REPORT_COLUMNS
from algotrade.research.incubation import read_fills, slippage_bps
from algotrade.strategies import (
    EWMAC,
    STRATEGIES,
    BollingerReversion,
    BreakoutBracket,
    VolTarget,
    to_spec,
)


def continuous(seed: int, n: int = 1200, drift: float = 0.0, freq: str = "4h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    open_ = np.concatenate(([100.0], close[:-1]))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def test_hourly_bars_match_too() -> None:
    position = run(to_spec(EWMAC(fast=8, slow=32)), continuous(11, freq="1h"), "1h")
    assert position.equity_gap() < 1e-8
    strategy = BreakoutBracket(lookback=10, rsi_length=5, stop_atr=2.0, target_atr=3.0,
                               cooldown_win=3, cooldown_loss=1)  # fmt: skip
    bracket = run(to_spec(strategy), continuous(12, freq="1h"), "1h")
    ours, theirs = closed_trades(bracket), bracket.trades()
    assert len(ours) == len(theirs) > 20
    pd.testing.assert_series_equal(ours["exit"], theirs["exit"], check_names=False)
    assert bracket.equity_gap() < 1e-6


def closed_trades(result):
    trades = result.ours.trades
    return trades[~trades["open"].astype(bool)].reset_index(drop=True)


@pytest.mark.parametrize(
    "strategy",
    [
        EWMAC(fast=8, slow=32),
        VolTarget(EWMAC(fast=16, slow=64), annual_vol=0.4),
        BollingerReversion(),
    ],
    ids=str,
)
def test_position_strategies_match_our_engine(strategy) -> None:
    result = run(to_spec(strategy), continuous(1), "4h")
    assert result.equity_gap() < 1e-8
    assert len(result.equity) == len(result.ours.equity)


@pytest.mark.parametrize("seed", [2, 3])
def test_bracket_trades_match_the_simulator(seed) -> None:
    strategy = BreakoutBracket(lookback=10, rsi_length=5, stop_atr=2.0, target_atr=3.0,
                               cooldown_win=3, cooldown_loss=1)  # fmt: skip
    result = run(to_spec(strategy), continuous(seed), "4h")
    ours, theirs = closed_trades(result), result.trades()
    assert len(ours) == len(theirs) > 20
    pd.testing.assert_series_equal(ours["entry"], theirs["entry"], check_names=False)
    pd.testing.assert_series_equal(ours["exit"], theirs["exit"], check_names=False)
    np.testing.assert_array_equal(ours["direction"].to_numpy(), theirs["direction"].to_numpy())
    # NautilusTrader rounds stop and target levels to the tick (1e-6 here); we do not.
    np.testing.assert_allclose(ours["exit_price"], theirs["exit_price"], rtol=0, atol=1e-6)
    assert result.equity_gap() < 1e-6


@pytest.mark.parametrize("seed", [9, 10])
def test_kill_switch_matches(seed) -> None:
    strategy = BreakoutBracket(lookback=10, rsi_length=5, stop_atr=1.0, target_atr=4.0,
                               cooldown_win=0, cooldown_loss=0, kill_drawdown=0.1)  # fmt: skip
    result = run(to_spec(strategy), continuous(seed), "4h")
    killed = result.ours.meta["killed_at"]
    ours, theirs = closed_trades(result), result.trades()
    assert len(ours) == len(theirs) > 20
    # Our simulator closes a killed trade at the next bar's open, NautilusTrader at the kill
    # bar's close: the same price and moment on these gap-free bars, labelled one bar apart.
    exits = ours["exit"].where(ours["exit_reason"] != "kill", ours["exit"] - pd.Timedelta("4h"))
    pd.testing.assert_series_equal(exits, theirs["exit"], check_names=False)
    assert result.equity_gap() < 1e-6
    after = result.equity[result.equity.index > killed + pd.Timedelta("8h")]
    assert len(after) > 100 and after.nunique() == 1  # flat for good


@dataclass(frozen=True)
class TimedBracket(BreakoutBracket):
    max_bars: int = 4

    name = "test_timed_bracket"


def test_time_exits_match(monkeypatch) -> None:
    monkeypatch.setitem(STRATEGIES, TimedBracket.name, TimedBracket)
    strategy = TimedBracket(lookback=10, rsi_length=5, stop_atr=4.0, target_atr=6.0,
                            cooldown_win=1, cooldown_loss=1, max_bars=4)  # fmt: skip
    result = run(to_spec(strategy), continuous(5), "4h")
    ours, theirs = closed_trades(result), result.trades()
    assert (ours["exit_reason"] == "time").sum() > 5
    assert len(ours) == len(theirs)
    pd.testing.assert_series_equal(ours["exit"], theirs["exit"], check_names=False)
    assert result.equity_gap() < 1e-6


def test_fills_are_logged_for_the_incubation_report(tmp_path) -> None:
    assert FILL_COLUMNS == REPORT_COLUMNS
    path = tmp_path / "incubation" / "fills.csv"
    strategy = BreakoutBracket(lookback=10, rsi_length=5, stop_atr=2.0, target_atr=3.0,
                               cooldown_win=3, cooldown_loss=1)  # fmt: skip
    run(to_spec(strategy), continuous(2, n=400), "4h", fills_path=str(path))
    fills = read_fills(path.parent)
    assert list(fills.columns) == FILL_COLUMNS and len(fills) > 10
    assert set(fills["side"]) == {1, -1} and fills["symbol"].eq("BTCUSDT-PERP").all()
    # Market fills happen at the decision close, stops at their trigger, targets at their limit.
    np.testing.assert_allclose(slippage_bps(fills), 0.0, atol=1e-6)


def test_exposure_cap_matches_our_engines_leverage_cap() -> None:
    strategy = VolTarget(EWMAC(fast=8, slow=32), annual_vol=3.0, max_leverage=3.0)
    bars = continuous(6, n=600)
    assert strategy.target_position(bars).abs().max() > 1.5  # wants more than the cap
    result = run(to_spec(strategy), bars, "4h")
    assert result.equity_gap() < 1e-8  # both hold it at 1x


def test_rebalance_threshold_trades_less(tmp_path) -> None:
    strategy = VolTarget(EWMAC(fast=8, slow=32), annual_vol=0.5)
    bars = continuous(7, n=500)
    every, lazy = tmp_path / "every.csv", tmp_path / "lazy.csv"
    exact = run(to_spec(strategy), bars, "4h", fills_path=str(every))
    loose = run(to_spec(strategy), bars, "4h", fills_path=str(lazy), min_change=0.05)
    assert exact.equity_gap() < 1e-8
    assert len(pd.read_csv(lazy)) < len(pd.read_csv(every)) / 2
    assert 0 < loose.equity_gap() < 0.05  # close to the research result, not identical
