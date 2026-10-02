import numpy as np
import pandas as pd
import pytest
from bracket_helpers import ZERO, bar_numbers, bars_from, random_bars
from conftest import make_bars

from algotrade.backtest.bracket import simulate_bracket
from algotrade.live.decider import Decider, Decision, Entry
from algotrade.strategies import EWMAC, BreakoutBracket, to_spec


def feed(decider: Decider, bars: pd.DataFrame, i: int) -> None:
    row = bars.iloc[i]
    decider.add_bar(bars.index[i], row["open"], row["high"], row["low"], row["close"], 0.0,
                    row.get("funding_rate", 0.0))  # fmt: skip


# --- position strategies -------------------------------------------------------------------


def test_position_decisions_equal_the_strategy_bar_by_bar() -> None:
    bars = make_bars(400)
    strategy = EWMAC(fast=8, slow=32)
    decider = Decider(to_spec(strategy), stake=0.5)
    expected = strategy.target_position(bars) * 0.5
    decisions = []
    for i in range(len(bars)):
        feed(decider, bars, i)
        decisions.append(decider.decide().target)
    np.testing.assert_allclose(decisions, expected.to_numpy(), atol=1e-12)
    assert not decider.bracket


def test_warm_history_and_bar_order() -> None:
    bars = make_bars(300)
    strategy = EWMAC(fast=8, slow=32)
    decider = Decider(to_spec(strategy), history=bars.iloc[:250][["open", "high", "low", "close"]])
    feed(decider, bars, 250)
    assert decider.decide().target == pytest.approx(
        strategy.target_position(bars.iloc[:251]).iloc[-1]
    )
    with pytest.raises(ValueError, match="is not after the last bar"):
        feed(decider, bars, 100)
    short = Decider(to_spec(strategy), max_history=50)
    for i in range(60):
        feed(short, bars, i)
    assert len(short.history) == 50


def test_nan_target_means_flat() -> None:
    decider = Decider({"type": "vol_target", "strategy": {"type": "buy_and_hold"}})
    feed(decider, make_bars(5), 0)
    assert decider.decide() == Decision(target=0.0)


# --- bracket strategies: a fake executor must reproduce the simulator ----------------------------


def paper_loop(bars: pd.DataFrame, strategy: BreakoutBracket) -> list[tuple]:
    """Execute the decider's decisions with the simulator's fill rules, at zero cost."""

    decider = Decider(to_spec(strategy))
    equity, trade, pending, kill_next = 1.0, None, None, False
    trades = []
    for i in range(len(bars)):
        o, h, lo, c = (bars[col].iloc[i] for col in ("open", "high", "low", "close"))
        if kill_next and trade:
            trades.append((trade["bar"], i, trade["d"], "kill"))
            equity = trade["equity"] + trade["d"] * trade["qty"] * (o - trade["px"])
            trade = None
        kill_next = False
        if pending is not None:
            px = o
            trade = {"d": pending.direction, "px": px, "qty": equity / px, "bar": i, "equity": equity,
                     "stop": px - pending.direction * pending.stop_distance,
                     "target": px + pending.direction * pending.target_distance}  # fmt: skip
            decider.entered()
            pending = None
        if trade:
            d, stop, target = trade["d"], trade["stop"], trade["target"]
            hit = None
            if (d > 0 and o <= stop) or (d < 0 and o >= stop):
                hit = (o, "stop")
            elif (d > 0 and o >= target) or (d < 0 and o <= target):
                hit = (o, "target")
            elif (d > 0 and lo <= stop) or (d < 0 and h >= stop):
                hit = (stop, "stop")
            elif (d > 0 and h >= target) or (d < 0 and lo <= target):
                hit = (target, "target")
            if hit:
                pnl = d * trade["qty"] * (hit[0] - trade["px"])
                equity = trade["equity"] + pnl
                trades.append((trade["bar"], i, d, hit[1]))
                decider.exited(pnl)
                trade = None
        mark = (
            equity if not trade else trade["equity"] + trade["d"] * trade["qty"] * (c - trade["px"])
        )
        feed(decider, bars, i)
        decision = decider.decide(equity=mark)
        if decision.close == "kill":
            kill_next = True
        if decision.entry:
            pending = decision.entry
    if trade:
        trades.append((trade["bar"], len(bars) - 1, trade["d"], "open"))
    return trades


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    ("cooldown_win", "cooldown_loss", "kill"), [(0, 0, 1.0), (4, 2, 1.0), (2, 1, 0.25)]
)
def test_decider_reproduces_the_bracket_simulator(seed, cooldown_win, cooldown_loss, kill) -> None:
    bars = random_bars(seed, n=500)
    strategy = BreakoutBracket(lookback=10, rsi_length=5, stop_atr=1.5, target_atr=3.0,
                               cooldown_win=cooldown_win, cooldown_loss=cooldown_loss,
                               kill_drawdown=kill)  # fmt: skip
    ours = paper_loop(bars, strategy)
    result = simulate_bracket(bars, strategy.signals(bars), ZERO, cooldown_win=cooldown_win,
                              cooldown_loss=cooldown_loss, kill_drawdown=kill)  # fmt: skip
    trades = result.trades
    expected = list(zip(bar_numbers(bars, trades["entry"]), bar_numbers(bars, trades["exit"]),
                        trades["direction"], trades["exit_reason"], strict=True))  # fmt: skip
    assert [(int(a), int(b), int(d), r) for a, b, d, r in expected] == ours
    assert len(ours) > 3


def test_time_exit_and_no_signal_cases() -> None:
    rows = [(100, 101, 99, 100)] * 30 + [(100, 111, 99, 110)] + [(110, 111, 109, 110)] * 6
    bars = bars_from(rows)
    strategy = BreakoutBracket(lookback=5, rsi_length=3, stop_atr=50, target_atr=50,
                               cooldown_win=0, cooldown_loss=0)  # fmt: skip
    spec = {**to_spec(strategy)}
    decider = Decider(spec)
    object.__setattr__(decider.strategy, "max_bars", 2)  # a bracket idea with a time exit
    for i in range(31):
        feed(decider, bars, i)
    entry = decider.decide(equity=1.0).entry
    assert entry == Entry(1, entry.stop_distance, entry.target_distance) and entry.stop_distance > 0
    decider.entered()
    feed(decider, bars, 31)
    assert decider.decide(equity=1.0) == Decision()  # held one bar
    feed(decider, bars, 32)
    assert decider.decide(equity=1.0) == Decision(close="time")
    decider.exited(-0.01, at_close=True)
    assert decider.state.last_exit_bar == 32 and decider.state.cooldown == 0


def test_kill_switch_while_flat_stops_everything() -> None:
    decider = Decider(to_spec(BreakoutBracket(lookback=5, rsi_length=3, kill_drawdown=0.5)))
    bars = random_bars(1, n=40)
    for i in range(40):
        feed(decider, bars, i)
    assert decider.decide(equity=1.0) is not None
    assert decider.decide(equity=0.4) == Decision() and decider.state.killed
    assert decider.decide(equity=2.0) == Decision()


def test_invalid_distances_mean_no_entry() -> None:
    decider = Decider(to_spec(BreakoutBracket(lookback=5, rsi_length=3, atr_length=200)))
    bars = random_bars(2, n=30)
    for i in range(30):
        feed(decider, bars, i)
    assert decider.decide(equity=1.0) == Decision()  # ATR still warming up
    assert decider.decide() == Decision()  # no equity known: no kill-switch bookkeeping
    assert decider.state.peak == 1.0
