"""Costs, funding and P&L bookkeeping, plus a cross-check against a slow reference simulator."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import (
    BASE,
    ZERO,
    bar_numbers,
    directional,
    entry,
    make_signals,
    random_bars,
    run,
)

from algotrade.backtest.bracket import simulate_bracket
from algotrade.backtest.costs import CostModel
from algotrade.backtest.engine import run_backtest
from algotrade.backtest.metrics import summarize
from algotrade.strategies import BreakoutBracket

DIRECTIONS = pytest.mark.parametrize("d", [1, -1], ids=["long", "short"])
SEEDS = range(20)


# --- costs and funding --------------------------------------------------------------------


@DIRECTIONS
def test_zero_cost_pnl_is_quantity_times_price_move(d) -> None:
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 11, 0, 0), (0, 0, 0, 0), (0, 0, 6, 0)] + [
        (0, 0, 0, 0)
    ] * 2
    bars = directional(d, rows)
    result = run(bars, cooldown_win=0, cooldown_loss=0, **entry(d, 0, 2))
    trades = result.trades
    assert trades["exit_reason"].tolist() == ["target", "stop"]
    expected = d * trades["qty"] * (trades["exit_price"] - trades["entry_price"])
    np.testing.assert_allclose(trades["pnl"], expected)
    assert result.ledger["equity"].iloc[-1] == pytest.approx(1 + trades["pnl"].sum())


@DIRECTIONS
def test_fees_are_charged_on_both_fills(d) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 6, 0), (0, 0, 0, 0)])
    fee = 0.001
    result = run(
        bars, costs=CostModel(fee_bps=10.0, slippage_bps=0.0, include_funding=False), **entry(d, 0)
    )
    trade = result.trades.iloc[0]
    qty, entry_px, exit_px = trade["qty"], trade["entry_price"], trade["exit_price"]
    assert qty == pytest.approx(1 / BASE)
    fees = qty * entry_px * fee + qty * exit_px * fee
    assert trade["pnl"] == pytest.approx(d * qty * (exit_px - entry_px) - fees)
    assert result.ledger["trading_cost"].sum() == pytest.approx(fees, rel=1e-3)
    assert result.ledger["equity"].iloc[-1] == pytest.approx(1 + trade["pnl"])


@DIRECTIONS
@pytest.mark.parametrize(
    ("exit_row", "slipped_fills"), [((0, 0, 6, 0), 2), ((0, 11, 0, 0), 1)], ids=["stop", "target"]
)
def test_slippage_hits_market_and_stop_fills_but_not_targets(d, exit_row, slipped_fills) -> None:
    bars = directional(d, [(0, 0, 0, 0), (0, 0, 0, 0), exit_row, (0, 0, 0, 0)])
    slip = CostModel(fee_bps=0.0, slippage_bps=10.0, include_funding=False)
    result = run(bars, costs=slip, **entry(d, 0))
    # Each slipped fill costs ~0.1% of a ~1x notional.
    assert result.ledger["trading_cost"].sum() == pytest.approx(slipped_fills * 0.001, rel=0.06)


@DIRECTIONS
def test_funding_is_paid_only_on_positions_held_at_the_close(d) -> None:
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 6, 0), (0, 0, 0, 0)]
    rate = 0.001
    bars = directional(d, rows, funding=rate)
    costs = CostModel(fee_bps=0.0, slippage_bps=0.0, include_funding=True)
    result = run(bars, costs=costs, **entry(d, 0))
    funding = result.ledger["funding_cost"]
    # Held at the close of bars 1 and 2; flat at bar 0 and stopped out inside bar 3.
    assert funding.iloc[[0, 3, 4]].eq(0).all()
    # Ledger columns are fractions of the previous bar's equity, which the first payment moved.
    expected = [d * rate, d * rate / (1 - d * rate)]
    np.testing.assert_allclose(funding.iloc[[1, 2]], expected, rtol=1e-9)
    trade = result.trades.iloc[0]
    move = d * trade["qty"] * (trade["exit_price"] - trade["entry_price"])
    assert trade["pnl"] == pytest.approx(move - 2 * d * rate)  # longs pay, shorts receive


def test_funding_is_ignored_when_switched_off() -> None:
    bars = directional(1, [(0, 0, 0, 0)] * 4, funding=0.01)
    result = run(bars, long=[0], costs=ZERO)
    assert result.ledger["funding_cost"].eq(0).all()


# --- invariants on random markets ----------------------------------------------------------


def busy_strategy(**overrides) -> BreakoutBracket:
    params = {
        "lookback": 5, "rsi_length": 5, "atr_length": 5, "stop_atr": 1.5, "target_atr": 2.0,
        "cooldown_win": 7, "cooldown_loss": 3, "kill_drawdown": 0.5,
    }  # fmt: skip
    params.update(overrides)
    return BreakoutBracket(**params)


@pytest.mark.parametrize("seed", SEEDS)
def test_invariants_on_random_markets(seed) -> None:
    bars = random_bars(seed)
    strategy = busy_strategy(kill_drawdown=0.9 if seed % 2 else 0.5)
    costs = CostModel(fee_bps=5.0, slippage_bps=3.0)
    slip = costs.slippage_bps / 10_000
    result = strategy.simulate(bars, costs)
    trades, ledger = result.trades, result.ledger
    sig = strategy.signals(bars)
    assert len(trades) > 3, "random market should produce trades"

    assert (ledger["equity"] >= 0).all()
    assert list(ledger.columns) == list(
        run_backtest(bars, pd.Series(0.0, index=bars.index)).ledger.columns
    )
    stats = summarize(result)
    assert all(math.isfinite(v) for k, v in stats.items() if k != "profit_factor")

    entries = bar_numbers(bars, trades["entry"])
    exits = bar_numbers(bars, trades["exit"])
    held = np.zeros(len(bars), dtype=bool)
    for k, trade in trades.reset_index(drop=True).iterrows():
        e, x, d = entries[k], exits[k], trade["direction"]
        # a valid signal on the bar before the entry
        assert (sig.long if d > 0 else sig.short).iloc[e - 1]
        # no overlap, and the right cooldown since the previous exit
        if k:
            prev = trades.iloc[k - 1]
            wait = strategy.cooldown_win if prev["pnl"] > 0 else strategy.cooldown_loss
            assert (e - 1) - exits[k - 1] >= wait
        # exit prices are a level or an open, inside the exit bar's range
        bar = bars.iloc[x]
        stop = trade["entry_price"] - d * sig.stop_dist.iloc[e - 1]
        target = trade["entry_price"] + d * sig.target_dist.iloc[e - 1]
        reason = trade["exit_reason"]
        if reason == "stop":
            raw = trade["exit_price"] / (1 - d * slip)
            assert raw == pytest.approx(stop) or raw == pytest.approx(bar["open"])
        elif reason == "target":
            raw = trade["exit_price"]
            assert raw == pytest.approx(target) or raw == pytest.approx(bar["open"])
        elif reason == "kill":
            raw = trade["exit_price"] / (1 - d * slip)
            assert raw == pytest.approx(bar["open"])
        else:
            raw = trade["exit_price"]
        assert bar["low"] - 1e-9 <= raw <= bar["high"] + 1e-9
        held[e : x + (1 if reason == "open" else 0)] = True

    # flat at every close outside a trade (cooldowns and after a kill included)
    assert ledger["position"][~held].eq(0).all()
    assert ledger["position"][held].ne(0).all()
    # trade P&L adds up to the change in equity
    assert trades["pnl"].sum() == pytest.approx(ledger["equity"].iloc[-1] - 1.0, abs=1e-9)
    killed = result.meta.get("killed_at")
    if killed is not None:
        assert (trades["entry"] <= killed).all()


# --- cross-check against an independent reference implementation ----------------------------


def reference_simulation(
    bars, signals, costs, leverage, cooldown_win, cooldown_loss, kill_drawdown, max_bars=0
):
    """A deliberately plain re-implementation of the documented rules, using cash accounting.

    It shares no code with the numba simulator: equity = cash + open P&L at the close, fills
    move cash directly. If both agree on every bar and trade, the rules are implemented as
    documented (or both are wrong in the same way, which is much less likely).
    """
    fee = costs.fee_bps / 10_000
    slip = costs.slippage_bps / 10_000
    use_funding = costs.include_funding
    o, h, lo, c = (bars[col].tolist() for col in ("open", "high", "low", "close"))
    rate = bars["funding_rate"].tolist()
    go_long, go_short = signals.long.tolist(), signals.short.tolist()
    stop_d, target_d = signals.stop_dist.tolist(), signals.target_dist.tolist()

    cash, peak = 1.0, 1.0
    trade = None  # open trade as a dict
    pending = None
    exit_at_open = False
    stopped_trading = False
    killed_bar = None
    earliest_signal = 0
    equity_path, trades = [], []

    def close_trade(i, price, reason):
        nonlocal cash, earliest_signal
        exit_fee = trade["qty"] * price * fee
        gain = trade["dir"] * trade["qty"] * (price - trade["entry_px"])
        cash += gain - exit_fee
        pnl = gain - trade["entry_fee"] - exit_fee - trade["funding"]
        trades.append(
            (
                trade["entry_bar"],
                i,
                trade["dir"],
                trade["entry_px"],
                price,
                reason,
                pnl,
                trade["equity"],
            )
        )
        earliest_signal = i + (cooldown_win if pnl > 0 else cooldown_loss)

    for i in range(len(c)):
        closed_this_bar = False
        if exit_at_open and trade is not None:
            close_trade(i, o[i] * (1 - trade["dir"] * slip), "kill")
            trade, closed_this_bar = None, True
        exit_at_open = False

        if pending is not None:
            direction, sd, td = pending
            px = o[i] * (1 + direction * slip)
            qty = leverage * cash / px  # flat, so cash == equity
            trade = {
                "dir": direction, "qty": qty, "entry_px": px, "entry_bar": i,
                "stop": px - direction * sd, "target": px + direction * td,
                "entry_fee": qty * px * fee, "funding": 0.0, "equity": cash,
            }  # fmt: skip
            cash -= trade["entry_fee"]
            pending = None

        if trade is not None:
            t = trade
            if t["dir"] > 0:
                if o[i] <= t["stop"]:
                    hit = (o[i] * (1 - slip), "stop")
                elif o[i] >= t["target"]:
                    hit = (o[i], "target")
                elif lo[i] <= t["stop"]:
                    hit = (t["stop"] * (1 - slip), "stop")
                elif h[i] >= t["target"]:
                    hit = (t["target"], "target")
                else:
                    hit = None
            else:
                if o[i] >= t["stop"]:
                    hit = (o[i] * (1 + slip), "stop")
                elif o[i] <= t["target"]:
                    hit = (o[i], "target")
                elif h[i] >= t["stop"]:
                    hit = (t["stop"] * (1 + slip), "stop")
                elif lo[i] <= t["target"]:
                    hit = (t["target"], "target")
                else:
                    hit = None
            if hit:
                close_trade(i, *hit)
                trade, closed_this_bar = None, True

        if trade is not None and use_funding:
            paid = trade["dir"] * trade["qty"] * c[i] * rate[i]
            cash -= paid
            trade["funding"] += paid

        if trade is not None and max_bars and i - trade["entry_bar"] + 1 >= max_bars:
            close_trade(i, c[i] * (1 - trade["dir"] * slip), "time")
            trade, closed_this_bar = None, True

        open_pnl = (
            0.0 if trade is None else trade["dir"] * trade["qty"] * (c[i] - trade["entry_px"])
        )
        equity = cash + open_pnl
        if equity <= 0:
            if trade is not None:
                trades.append(
                    (
                        trade["entry_bar"],
                        i,
                        trade["dir"],
                        trade["entry_px"],
                        c[i],
                        "liquidated",
                        -trade["equity"],
                        trade["equity"],
                    )
                )
                trade = None
            elif closed_this_bar:
                last = trades[-1]
                trades[-1] = (*last[:5], "liquidated", -last[7], last[7])
            cash = equity = 0.0
            if not stopped_trading:
                stopped_trading, killed_bar = True, i
        equity_path.append(equity)

        peak = max(peak, equity)
        if not stopped_trading and equity <= (1 - kill_drawdown) * peak:
            stopped_trading, killed_bar = True, i
            exit_at_open = trade is not None

        if trade is None and not stopped_trading and i >= earliest_signal:
            valid = stop_d[i] > 0 and target_d[i] > 0
            if go_long[i] and valid:
                pending = (1, stop_d[i], target_d[i])
            elif go_short[i] and valid:
                pending = (-1, stop_d[i], target_d[i])

    if trade is not None:
        last = len(c) - 1
        pnl = (
            trade["dir"] * trade["qty"] * (c[last] - trade["entry_px"])
            - trade["entry_fee"]
            - trade["funding"]
        )
        trades.append(
            (
                trade["entry_bar"],
                last,
                trade["dir"],
                trade["entry_px"],
                c[last],
                "open",
                pnl,
                trade["equity"],
            )
        )
    return np.array(equity_path), trades, killed_bar


def random_signals(bars, seed):
    rng = np.random.default_rng(seed + 1000)
    n = len(bars)
    long = rng.random(n) < 0.08
    short = rng.random(n) < 0.08
    stop = bars["close"].to_numpy() * rng.uniform(0.005, 0.05, n)
    target = bars["close"].to_numpy() * rng.uniform(0.005, 0.08, n)
    stop[rng.random(n) < 0.05] = np.nan  # warm-up style gaps in the distances
    signals = make_signals(bars, np.flatnonzero(long), np.flatnonzero(short), stop, target)
    return signals


SCENARIOS = [
    pytest.param(CostModel(0.0, 0.0, include_funding=False), 1.0, 0, 0, 1.0, 0, id="frictionless"),
    pytest.param(CostModel(5.0, 3.0), 1.0, 5, 20, 0.5, 0, id="book-rules"),
    pytest.param(CostModel(5.5, 10.0), 2.0, 3, 7, 0.3, 0, id="levered-tight-kill"),
    pytest.param(CostModel(5.0, 3.0), 25.0, 1, 1, 1.0, 0, id="liquidation-prone"),
    pytest.param(CostModel(5.0, 3.0), 1.0, 2, 4, 0.6, 6, id="time-exit"),
    pytest.param(CostModel(5.0, 3.0), 1.0, 0, 0, 1.0, 1, id="one-bar-trades"),
    pytest.param(CostModel(5.0, 3.0), 30.0, 0, 0, 1.0, 3, id="time-exit-liquidation"),
]
SCENARIO_ARGS = ("costs", "leverage", "cd_loss", "cd_win", "kill", "max_bars")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(SCENARIO_ARGS, SCENARIOS)
def test_matches_reference_simulation(seed, costs, leverage, cd_loss, cd_win, kill, max_bars):
    bars = random_bars(seed, n=300)
    signals = random_signals(bars, seed)
    result = simulate_bracket(
        bars, signals, costs, leverage=leverage,
        cooldown_win=cd_win, cooldown_loss=cd_loss, kill_drawdown=kill, max_bars=max_bars,
    )  # fmt: skip
    equity, trades, killed_bar = reference_simulation(
        bars, signals, costs, leverage, cd_win, cd_loss, kill, max_bars
    )

    np.testing.assert_allclose(result.ledger["equity"].to_numpy(), equity, rtol=1e-9, atol=1e-12)
    ours = result.trades
    assert len(ours) == len(trades)
    if trades:
        entry_bar, exit_bar, direction, entry_px, exit_px, reason, pnl, _ = map(
            list, zip(*trades, strict=True)
        )
        assert list(bar_numbers(bars, ours["entry"])) == entry_bar
        assert list(bar_numbers(bars, ours["exit"])) == exit_bar
        assert ours["direction"].tolist() == direction
        assert ours["exit_reason"].tolist() == reason
        np.testing.assert_allclose(ours["entry_price"], entry_px, rtol=1e-12)
        np.testing.assert_allclose(ours["exit_price"], exit_px, rtol=1e-12)
        np.testing.assert_allclose(ours["pnl"], pnl, rtol=1e-9, atol=1e-12)
    expected_kill = None if killed_bar is None else bars.index[killed_bar]
    assert result.meta.get("killed_at") == expected_kill


def test_reference_scenarios_cover_every_exit_reason() -> None:
    """Guard against the cross-check passing only because it never reaches some branches."""
    seen = set()
    for seed in SEEDS:
        bars = random_bars(seed, n=300)
        signals = random_signals(bars, seed)
        for scenario in SCENARIOS:
            costs, leverage, cd_loss, cd_win, kill, max_bars = scenario.values
            result = simulate_bracket(
                bars, signals, costs, leverage=leverage,
                cooldown_win=cd_win, cooldown_loss=cd_loss, kill_drawdown=kill,
                max_bars=max_bars,
            )  # fmt: skip
            seen |= set(result.trades["exit_reason"])
    assert seen == {"stop", "target", "kill", "liquidated", "open", "time"}
