from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import bars_from, random_bars
from conftest import make_bars

from algotrade.backtest.bracket import BracketSignals, simulate_bracket
from algotrade.backtest.costs import ZERO_COSTS, CostModel
from algotrade.backtest.engine import run_backtest
from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
from algotrade.backtest.runner import backtest
from algotrade.indicators import atr
from algotrade.strategies import EWMAC, BreakoutBracket, Strategy
from algotrade.validation.diagnostics import diagnostics, regime_labels, trade_excursions
from algotrade.validation.entry_test import cell_profit, entry_signals, run_entry_test
from algotrade.validation.fast import bracket_returns, position_returns, sharpe
from algotrade.validation.monkey import monkey_test
from algotrade.validation.optimize import (
    choose,
    neighbourhood_scores,
    profitable_share,
    run_grid,
)

COSTS = CostModel(fee_bps=5.0, slippage_bps=3.0)
NO_FUNDING = CostModel(fee_bps=5.0, slippage_bps=3.0, include_funding=False)


def universe(n: int = 3, size: int = 600) -> dict[str, pd.DataFrame]:
    return {f"S{i}": random_bars(i, n=size) for i in range(n)}


@dataclass(frozen=True)
class Clairvoyant(Strategy):
    """Knows the next bar's direction: a planted edge no monkey can match."""

    name = "test_clairvoyant"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        return np.sign(bars["close"].pct_change().shift(-1)).fillna(0.0)


@dataclass(frozen=True)
class CoinFlip(Strategy):
    seed: int = 3

    name = "test_coin_flip"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        flips = np.random.default_rng(self.seed).choice([-1.0, 1.0], len(bars) // 20 + 1)
        return pd.Series(np.repeat(flips, 20)[: len(bars)], index=bars.index)


# --- fast paths equal the engine ----------------------------------------------------------


@pytest.mark.parametrize("costs", [COSTS, NO_FUNDING], ids=["funding", "no-funding"])
def test_position_returns_equal_the_engine(costs) -> None:
    bars = make_bars(800)
    target = EWMAC(fast=8, slow=32).target_position(bars)
    engine = run_backtest(bars, target, costs).returns.to_numpy()
    np.testing.assert_allclose(position_returns(bars, target.to_numpy(), costs), engine, atol=1e-15)


@pytest.mark.parametrize("max_bars", [0, 3])
def test_bracket_returns_equal_the_simulator(max_bars) -> None:
    bars = random_bars(5, n=500)
    strategy = BreakoutBracket(lookback=10, rsi_length=5, cooldown_win=2, cooldown_loss=1)
    signals = strategy.signals(bars)
    ledger = simulate_bracket(
        bars, signals, COSTS, cooldown_win=2, cooldown_loss=1, kill_drawdown=0.5,
        max_bars=max_bars,
    ).ledger  # fmt: skip
    fast = bracket_returns(
        bars, signals.long.fillna(False), signals.short.fillna(False), signals.stop_dist,
        signals.target_dist, COSTS, 2, 1, 0.5, max_bars=max_bars,
    )  # fmt: skip
    np.testing.assert_allclose(fast, ledger["net"].to_numpy(), atol=1e-12)


def test_bracket_returns_after_liquidation_stay_flat() -> None:
    bars = bars_from([(100, 100, 100, 100), (100, 100, 1, 1), (1, 1, 1, 1), (1, 2, 1, 2)])
    inf = np.full(4, np.inf)
    returns = bracket_returns(bars, [True, False, False, False], [False] * 4, inf, inf, COSTS,
                              leverage=5.0)  # fmt: skip
    assert returns[1] == -1.0 and (returns[2:] == 0).all()


def test_sharpe_matches_metrics() -> None:
    returns = np.array([0.01, -0.02, 0.015, 0.0, 0.007])
    assert sharpe(returns, 365.25) == pytest.approx(sharpe_ratio(pd.Series(returns), 365.25))
    assert sharpe(np.zeros(5), 365.25) == 0.0
    assert sharpe(np.array([0.01]), 365.25) == 0.0


# --- entry test -----------------------------------------------------------------------------


def test_position_entries_are_sign_changes() -> None:
    bars = make_bars(8)

    @dataclass(frozen=True)
    class Pattern(Strategy):
        name = "test_pattern"

        def target_position(self, bars):
            return pd.Series([0, 1, 1, -1, -1, 0, -0.5, 1.0], index=bars.index, dtype=float)

    long, short = entry_signals(Pattern(), bars)
    assert long.tolist() == [False, True, False, False, False, False, False, True]
    assert short.tolist() == [False, False, False, True, False, False, True, False]


def test_bracket_entries_are_its_signals() -> None:
    bars = random_bars(1)
    strategy = BreakoutBracket(lookback=10, rsi_length=5)
    long, short = entry_signals(strategy, bars)
    signals = strategy.signals(bars)
    assert long.equals(signals.long.fillna(False)) and short.equals(signals.short.fillna(False))


def test_entry_test_table_and_planted_edge() -> None:
    markets = universe()
    test = run_entry_test(Clairvoyant(), markets, NO_FUNDING, [1, 5], 2.0, 4.0, 14)
    assert len(test.table) == 3 * 3
    assert set(test.table["exit"]) == {"1 bars", "5 bars", "2/4 ATR bracket"}
    one_bar = test.table[test.table["exit"] == "1 bars"]
    # It knows each bar's close-to-close move; gaps at the open cost it a few trades.
    assert one_bar["profitable"].all() and (one_bar["win_rate"] > 0.75).all()
    assert test.by_exit().loc["1 bars", "profitable"] == 1.0
    assert 0 < test.profitable_share <= 1


def test_cell_profit_fixed_size_against_compounded() -> None:
    # +50% then -40%: one contract each time makes +10%, the whole account ends at 0.9.
    trades = pd.Series([0.5, -0.4])
    assert cell_profit(trades, 0.9, "fixed") == pytest.approx(0.1)
    assert cell_profit(trades, 0.9, "compounded") == pytest.approx(-0.1)
    assert cell_profit(pd.Series(dtype=float), 1.0, "fixed") == 0.0


@pytest.mark.parametrize("scoring", ["fixed", "compounded"])
def test_entry_test_scoring_decides_profitable(scoring) -> None:
    test = run_entry_test(Clairvoyant(), universe(), COSTS, [1, 5], 2.0, 4.0, 14, scoring)
    table = test.table
    column = "fixed_return" if scoring == "fixed" else "net_return"
    assert table["profitable"].equals((table[column] > 0) & (table["trades"] > 0))
    assert (table["fixed_return"] != table["net_return"]).any()


def test_fixed_scoring_counts_every_trade_after_a_wipe_out() -> None:
    # A 1x short from 160 to 400 loses 150%: the compounded account is gone after the first
    # trade, while the fixed-size count records that loss and the three flat trades after it.
    close = np.concatenate([np.linspace(100, 400, 6), np.full(30, 400.0)])
    index = pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC")
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=index)

    @dataclass(frozen=True)
    class ShortEvery8(Strategy):
        name = "test_short_every_8"

        def target_position(self, bars):
            return pd.Series(np.where(np.arange(len(bars)) % 8 == 0, -1.0, 0.0), index=bars.index)

    test = run_entry_test(ShortEvery8(), {"X": bars}, ZERO_COSTS, [5], 2.0, 4.0, 14, "fixed")
    row = test.table.set_index("exit").loc["5 bars"]
    assert row["trades"] == 4 and row["win_rate"] == 0.0
    assert row["fixed_return"] == pytest.approx(-1.5)
    assert row["net_return"] <= -1.0 and not row["profitable"]


def test_entry_test_rejects_unknown_scoring() -> None:
    with pytest.raises(ValueError, match="scoring must be one of"):
        run_entry_test(Clairvoyant(), universe(1), COSTS, [5], 2.0, 4.0, 14, "average")


def test_entry_test_without_entries_is_not_profitable() -> None:
    @dataclass(frozen=True)
    class Never(Strategy):
        name = "test_never"

        def target_position(self, bars):
            return pd.Series(0.0, index=bars.index)

    test = run_entry_test(Never(), universe(1), COSTS, [5], 2.0, 4.0, 14)
    assert test.profitable_share == 0.0 and (test.table["trades"] == 0).all()
    assert test.table["win_rate"].eq(0).all() and test.table["avg_trade"].eq(0).all()


# --- monkey test ----------------------------------------------------------------------------


def test_clairvoyant_beats_every_monkey() -> None:
    result = monkey_test(Clairvoyant(), universe(), NO_FUNDING, runs=40)
    assert result.percentile == 1.0 and result.real > 5


def test_random_signals_fail_the_monkey_test() -> None:
    result = monkey_test(CoinFlip(), universe(), COSTS, runs=200, seed=1)
    assert result.percentile < 0.9


def test_monkeys_are_reproducible_and_brackets_work() -> None:
    markets = universe(2, 400)
    strategy = BreakoutBracket(lookback=10, rsi_length=5, cooldown_win=0, cooldown_loss=0)
    first = monkey_test(strategy, markets, COSTS, runs=25, seed=7)
    second = monkey_test(strategy, markets, COSTS, runs=25, seed=7)
    np.testing.assert_array_equal(first.monkeys, second.monkeys)
    real = np.median(
        [sharpe_ratio(backtest(strategy, bars, COSTS).returns, periods_per_year(bars.index))
         for bars in markets.values()]
    )  # fmt: skip
    assert first.real == pytest.approx(real)
    assert 0.0 <= first.percentile <= 1.0


def test_monkey_without_runs() -> None:
    assert monkey_test(EWMAC(), universe(1), COSTS, runs=0).percentile == 0.0


# --- limited optimisation -------------------------------------------------------------------


def test_run_grid_rows() -> None:
    board = run_grid({"type": "ewmac"}, {"fast": [4, 8], "slow": [32, 64]}, universe(2), COSTS)
    assert len(board) == 4 and list(board[["fast", "slow"]].iloc[1]) == [4, 64]
    assert {"median_sharpe", "median_trades", "killed", "spec_hash", "spec"} <= set(board)
    assert board["spec_hash"].nunique() == 4
    single = run_grid({"type": "ewmac"}, {}, universe(1), COSTS)
    assert len(single) == 1 and '"type": "ewmac"' in single["spec"].iloc[0]
    assert 0.0 <= profitable_share(board) <= 1.0
    assert profitable_share(board.iloc[:0]) == 0.0


def test_plateau_beats_a_lonely_spike() -> None:
    cells = [(a, b) for a in (1, 2, 3) for b in (10, 20, 30)]
    value = {(1, 10): 3.0, (2, 20): 1.5, (2, 30): 1.5, (3, 20): 1.5, (3, 30): 1.5}
    board = pd.DataFrame(
        [{"a": a, "b": b, "median_sharpe": value.get((a, b), 0.0)} for a, b in cells]
    )
    scores = neighbourhood_scores(board, ["a", "b"])
    assert scores[0] == pytest.approx((3.0 + 0 + 0 + 1.5) / 4)  # the spike's neighbourhood
    assert scores[8] == pytest.approx(1.5)  # (3, 30): all four neighbours on the plateau
    assert scores[4] == pytest.approx((3.0 + 4 * 1.5) / 9)  # centre sees everything
    best = choose(board, ["a", "b"])
    assert (best["a"], best["b"]) == (3, 30)
    flat = pd.DataFrame({"median_sharpe": [0.1, 0.4, 0.2]})
    assert choose(flat, [])["median_sharpe"] == 0.4
    assert neighbourhood_scores(flat, []).tolist() == [0.1, 0.4, 0.2]


# --- diagnostics ----------------------------------------------------------------------------


def test_excursions_of_a_bracket_trade() -> None:
    bars = bars_from(
        [(100, 102, 98, 100), (100, 104, 97, 103), (103, 108, 101, 105), (105, 105, 105, 105)]
    )
    long = pd.Series([True, False, False, False], index=bars.index)
    inf = pd.Series(np.inf, index=bars.index)
    signals = BracketSignals(long, ~long & False, inf, inf)
    result = simulate_bracket(bars, signals, CostModel(0, 0, False), max_bars=2)
    trade = trade_excursions(result, bars, atr_length=1).iloc[0]
    # Bought at 100 on bar 1, sold at bar 2's close: high 108, low 97 while held.
    assert trade["mfe"] == pytest.approx(0.08) and trade["mae"] == pytest.approx(0.03)
    signal_bar_atr = atr(bars["high"], bars["low"], bars["close"], 1).iloc[0] / 100
    assert trade["mfe_atr"] == pytest.approx(0.08 / signal_bar_atr)
    assert trade["mae_atr"] == pytest.approx(0.03 / signal_bar_atr)


def test_excursions_of_an_engine_trade_short() -> None:
    bars = bars_from([(100, 100, 100, 100), (100, 101, 95, 96), (96, 99, 90, 92), (92, 92, 92, 92)])
    target = pd.Series([-1.0, -1.0, 0.0, 0.0], index=bars.index)
    result = run_backtest(bars, target, CostModel(0, 0, False))
    trade = trade_excursions(result, bars).iloc[0]
    # Short entered at the first close (100), held through bars 1-2: best low 90, worst high 101.
    assert trade["mfe"] == pytest.approx(0.10) and trade["mae"] == pytest.approx(0.01)


def test_excursions_without_trades() -> None:
    bars = make_bars(50)
    result = run_backtest(bars, pd.Series(0.0, index=bars.index))
    assert trade_excursions(result, bars).empty


def test_regime_labels_cover_every_bar() -> None:
    bars = make_bars(400)
    labels = regime_labels(bars)
    assert set(labels["trend"]) <= {"trending (ADX>=25)", "ranging (ADX<=20)", "neither", "warm-up"}
    assert labels["trend"].iloc[0] == "warm-up" and labels["volatility"].iloc[0] == "warm-up"
    assert {"low vol", "mid vol", "high vol"} <= set(labels["volatility"])


def test_diagnostics_tables() -> None:
    markets = universe(2, 500)
    results = {s: backtest(EWMAC(fast=4, slow=16), b, COSTS) for s, b in markets.items()}
    found = diagnostics(results, markets)
    assert set(found) == {"symbols", "sides", "years", "regimes", "holding", "excursions", "trades"}
    assert list(found["sides"].index) == ["all", "long", "short"]
    sides = found["sides"]
    assert sides.loc["all", "trades"] == sides.loc["long", "trades"] + sides.loc["short", "trades"]
    shares = found["regimes"].groupby("kind")["share_of_time"].sum()
    np.testing.assert_allclose(shares.to_numpy(), 1.0)
    assert "median" in found["years"].columns
    assert set(found["excursions"].index) <= {"winners", "losers"}


def test_diagnostics_without_any_trades() -> None:
    markets = universe(1, 300)
    flat = {s: run_backtest(b, pd.Series(0.0, index=b.index)) for s, b in markets.items()}
    found = diagnostics(flat, markets)
    assert found["sides"]["trades"].eq(0).all()
    assert found["holding"].empty and found["excursions"].empty
