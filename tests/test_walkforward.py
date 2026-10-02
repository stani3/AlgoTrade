from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import random_bars

from algotrade.backtest.costs import CostModel
from algotrade.backtest.engine import run_backtest
from algotrade.strategies import EWMAC, BreakoutBracket, to_spec
from algotrade.validation.walkforward import (
    WalkForward,
    make_windows,
    out_of_sample,
    stitch,
    walk_forward,
)

COSTS = CostModel(fee_bps=5.0, slippage_bps=3.0)
UTC = "UTC"


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=UTC)


# --- windows --------------------------------------------------------------------------------


def test_windows_start_on_a_month_and_roll() -> None:
    windows = make_windows(ts("2019-09-08 16:00"), ts("2025-05-01"), 2, 6)
    assert windows[0].in_sample_start == ts("2019-10-01")
    assert windows[0].out_of_sample_start == ts("2021-10-01")
    assert windows[0].out_of_sample_end == ts("2022-04-01")
    assert all(a.out_of_sample_end == b.out_of_sample_start for a, b in pairwise(windows))
    assert windows[-1].out_of_sample_start == ts("2025-04-01")
    assert windows[-1].out_of_sample_end == ts("2025-05-01")
    assert len(windows) == 8


def test_short_final_window_is_dropped_and_month_start_kept() -> None:
    windows = make_windows(ts("2023-01-01"), ts("2025-01-20"), 1, 6)
    assert windows[0].out_of_sample_start == ts("2024-01-01")
    assert [w.out_of_sample_end for w in windows] == [ts("2024-07-01"), ts("2025-01-01")]
    assert make_windows(ts("2024-01-01"), ts("2024-06-01"), 1, 6) == []


# --- out-of-sample pieces and stitching -------------------------------------------------------


def test_out_of_sample_starts_flat_with_warm_indicators() -> None:
    bars = random_bars(3, n=900)
    start = bars.index[500]
    spec = to_spec(EWMAC(fast=8, slow=32))
    piece = out_of_sample(spec, bars, start, COSTS)
    assert piece.ledger.index[0] == start and piece.ledger["position"].iloc[0] == 0.0
    warm = EWMAC(fast=8, slow=32).target_position(bars)
    expected = run_backtest(bars.loc[start:], warm.loc[start:], COSTS)
    pd.testing.assert_frame_equal(piece.ledger, expected.ledger)


def test_bracket_out_of_sample_uses_full_history_signals_and_a_fresh_account() -> None:
    bars = random_bars(4, n=900)
    strategy = BreakoutBracket(lookback=10, rsi_length=5, kill_drawdown=0.3)
    start = bars.index[450]
    piece = out_of_sample(to_spec(strategy), bars, start, COSTS)
    assert piece.ledger.index[0] == start and piece.ledger["equity"].iloc[0] == pytest.approx(1.0)
    assert (piece.trades["entry"] > start).all()


def test_stitched_positions_equal_one_engine_run_that_goes_flat_at_each_boundary() -> None:
    bars = random_bars(8, n=1200)
    cuts = [bars.index[400], bars.index[700], bars.index[1000]]
    ends = [bars.index[700], bars.index[1000], bars.index[-1] + pd.Timedelta("4h")]
    specs = [EWMAC(fast=8, slow=32), EWMAC(fast=16, slow=64), EWMAC(fast=4, slow=16)]
    pieces, targets = [], []
    for start, end, strategy in zip(cuts, ends, specs, strict=True):
        history = bars[bars.index < end]
        pieces.append(out_of_sample(to_spec(strategy), history, start, COSTS))
        target = strategy.target_position(history).loc[start:].copy()
        target.iloc[-1] = 0.0  # the piece ends flat at its last close
        targets.append(target)
    stitched = stitch(pieces, COSTS)
    sequence = pd.concat(targets)
    sequence.iloc[-1] = targets[-1].iloc[-1]  # the last piece's final target is never used
    reference = run_backtest(bars.loc[cuts[0] :], sequence, COSTS)
    np.testing.assert_allclose(
        stitched.returns.to_numpy(), reference.returns.to_numpy(), atol=1e-15
    )
    np.testing.assert_allclose(stitched.equity.to_numpy(), reference.equity.to_numpy(), rtol=1e-12)
    pd.testing.assert_frame_equal(stitched.trades, reference.trades)


def test_stitched_bracket_trades_add_up_to_the_equity() -> None:
    bars = random_bars(11, n=1300)
    strategy = BreakoutBracket(lookback=8, rsi_length=4, stop_atr=3, target_atr=6,
                               cooldown_win=0, cooldown_loss=0)  # fmt: skip
    cuts = [bars.index[300], bars.index[700], bars.index[1000]]
    ends = [bars.index[700], bars.index[1000], bars.index[-1] + pd.Timedelta("4h")]
    pieces = [
        out_of_sample(to_spec(strategy), bars[bars.index < end], start, COSTS)
        for start, end in zip(cuts, ends, strict=True)
    ]
    assert any(p.trades["open"].any() for p in pieces[:-1]), "test needs a trade cut by a window"
    stitched = stitch(pieces, COSTS)
    trades = stitched.trades
    assert (trades.loc[trades["exit_reason"] == "window end", "open"] == False).all()
    assert stitched.equity.iloc[-1] - 1.0 == pytest.approx(trades["pnl"].sum(), rel=1e-9)
    first_bars = [p.ledger.index[0] for p in pieces[1:]]
    held = [abs(p.ledger["position"].iloc[-1]) for p in pieces[:-1]]
    for when, size in zip(first_bars, held, strict=True):
        assert stitched.ledger.at[when, "turnover"] == pytest.approx(size)


# --- the analysis ------------------------------------------------------------------------------


def trending(seed: int, n: int = 6000, edge: float = 0.002) -> pd.DataFrame:
    bars = random_bars(seed, n=n)
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1.0, 1.0], n // 120 + 1), 120)[:n]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n) + edge * regime))
    bars["close"] = close
    bars["open"] = np.concatenate(([close[0]], close[:-1]))
    bars["high"] = np.maximum(bars["open"], close) * 1.004
    bars["low"] = np.minimum(bars["open"], close) * 0.996
    return bars


def test_walk_forward_on_a_planted_edge() -> None:
    universe = {f"S{i}": trending(i) for i in range(3)}
    end = universe["S0"].index[-1]
    wf = walk_forward({"type": "ewmac"}, {"fast": [8, 16], "slow": [64, 128]}, universe, COSTS,
                      end, 1, 3)  # fmt: skip
    assert len(wf.windows) >= 4
    assert set(wf.windows["fast"]) <= {8, 16} and (wf.windows["is_symbols"] == 3).all()
    assert wf.oos_median_sharpe > 1 and wf.efficiency > 0.5 and wf.profitable_windows > 0.7
    stitched = wf.results["S0"]
    assert stitched.ledger.index[0] == pd.Timestamp(wf.windows["oos_start"].iloc[0])
    assert stitched.ledger.index.is_unique


def test_symbols_without_enough_history_sit_out_the_in_sample() -> None:
    long = trending(1, n=5000)
    short = trending(2, n=5000).iloc[2500:]
    wf = walk_forward({"type": "ewmac"}, {}, {"A": long, "B": short}, COSTS,
                      long.index[-1], 1, 3)  # fmt: skip
    first = wf.windows.iloc[0]
    # B is too new to be optimised on, but trades out-of-sample with the parameters A chose.
    assert first["is_symbols"] == 1 and first["oos_symbols"] == 2
    assert wf.windows["is_symbols"].iloc[-1] == 2 and set(wf.results) == {"A", "B"}


def test_empty_walk_forward() -> None:
    bars = trending(1, n=800)
    wf = walk_forward({"type": "ewmac"}, {}, {"A": bars}, COSTS, bars.index[-1], 5, 3)
    assert wf.windows.empty and wf.results == {}
    assert wf.efficiency == 0.0 and wf.profitable_windows == 0.0 and wf.oos_median_sharpe == 0.0
    losing = WalkForward(
        pd.DataFrame(
            {"is_median_cagr": [-0.1], "oos_annualised": [0.2], "oos_median_return": [0.1]}
        ),
        {},
    )
    assert losing.efficiency == 0.0  # in-sample never made money


def test_windows_where_no_symbol_has_in_sample_data_are_skipped() -> None:
    late = trending(3, n=4000).iloc[3000:]
    wf = walk_forward({"type": "ewmac"}, {}, {"A": late}, COSTS, late.index[-1], 0.05, 1)
    assert len(wf.windows) >= 1 and (wf.windows["is_symbols"] == 1).all()


def test_annualising_impossible_returns() -> None:
    from algotrade.validation.walkforward import _annualised

    assert _annualised(-1.0, 90) == -1.0 and _annualised(0.1, 0) == -1.0
    assert _annualised(0.21, 365) == pytest.approx(0.21, rel=1e-3)


def test_windows_with_no_usable_symbol_or_no_data_are_skipped() -> None:
    early = trending(5, n=4000).iloc[:20]  # a few bars, then the market disappears
    late = trending(6, n=8000).iloc[4000:]
    wf = walk_forward({"type": "ewmac"}, {}, {"E": early, "L": late}, COSTS,
                      late.index[-1], 1, 3)  # fmt: skip
    assert set(wf.results) == {"L"}
    assert (wf.windows["is_symbols"] == 1).all() and (wf.windows["oos_symbols"] == 1).all()
    assert wf.windows["oos_start"].iloc[0] >= late.index[0] + pd.Timedelta(days=180)
