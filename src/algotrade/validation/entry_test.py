"""Davey's entry test: is the entry any good on its own, whatever the exit?

The strategy's entries are traded with exits that know nothing about the strategy: a fixed
number of bars (market-on-close), and a fixed ATR stop and target. A good entry makes money on
most (exit, market) combinations; Davey looks for roughly 70% of them profitable.

A cell's profit is scored one of two ways:

* ``fixed`` (Davey): every trade has the same size and the profit is the sum of the trade
  returns, like trading one contract each time. No loss, however large, ends the test early.
* ``compounded``: the whole equity goes into each trade in turn. On volatile markets a run of
  full-size trades with a positive average can still compound to a loss, so this is harsher.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.backtest.bracket import BracketSignals, simulate_bracket
from algotrade.backtest.costs import CostModel, costs_for
from algotrade.indicators import atr
from algotrade.strategies import BracketStrategy, Strategy

FIXED_SIZE = 1e-3  # position size, as a share of equity, for the fixed-size run

SCORINGS = {
    "fixed": "fixed size per trade, summed (Davey)",
    "compounded": "whole equity per trade, compounded",
}


def cell_profit(trade_returns: pd.Series, final_equity: float, scoring: str) -> float:
    """One (exit, market) cell's profit under ``scoring`` (see the module notes)."""

    return float(trade_returns.sum()) if scoring == "fixed" else final_equity - 1.0


def entry_signals(strategy: Strategy, bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Long and short entry bars: a bracket strategy's signals, or the bars where a position
    strategy's exposure turns from flat or the opposite side into a direction."""

    if isinstance(strategy, BracketStrategy):
        signals = strategy.signals(bars)
        return signals.long.fillna(False).astype(bool), signals.short.fillna(False).astype(bool)
    side = np.sign(strategy.target_position(bars).fillna(0.0))
    before = side.shift(1, fill_value=0.0)
    return (side > 0) & (before <= 0), (side < 0) & (before >= 0)


@dataclass
class EntryTest:
    table: pd.DataFrame  # one row per (exit, symbol)

    @property
    def profitable_share(self) -> float:
        return float(self.table["profitable"].mean()) if len(self.table) else 0.0

    def by_exit(self) -> pd.DataFrame:
        return self.table.groupby("exit", sort=False).agg(
            profitable=("profitable", "mean"),
            trades=("trades", "sum"),
            win_rate=("win_rate", "median"),
            avg_trade=("avg_trade", "median"),
        )


def entry_rows(
    strategy: Strategy,
    symbol: str,
    bars: pd.DataFrame,
    costs: CostModel,
    exit_bars: list[int],
    stop_atr: float,
    target_atr: float,
    atr_length: int,
    scoring: str = "compounded",
    sides: tuple[pd.Series, pd.Series] | None = None,
) -> list[dict]:
    """One symbol's (exit, symbol) cells; ``sides`` are its entries if already computed."""

    long, short = sides if sides is not None else entry_signals(strategy, bars)
    volatility = atr(bars["high"], bars["low"], bars["close"], atr_length)
    never = pd.Series(np.inf, index=bars.index)
    exits = {f"{n} bars": (never, never, n) for n in exit_bars}
    exits[f"{stop_atr:g}/{target_atr:g} ATR bracket"] = (
        stop_atr * volatility,
        target_atr * volatility,
        0,
    )
    rows = []
    for name, (stop, target, max_bars) in exits.items():
        signals = BracketSignals(long, short, stop, target)
        whole = simulate_bracket(bars, signals, costs, max_bars=max_bars)
        # A tiny size never wipes out the account, so every trade happens; dividing by the
        # size gives each trade's return per unit traded (fees and funding scale with it).
        small = simulate_bracket(bars, signals, costs, leverage=FIXED_SIZE, max_bars=max_bars)
        closed = small.trades[~small.trades["open"].astype(bool)]
        per_unit = closed["return"] / FIXED_SIZE
        fixed = cell_profit(per_unit, float(small.equity.iloc[-1]), "fixed")
        compounded = cell_profit(per_unit, float(whole.equity.iloc[-1]), "compounded")
        profit = fixed if scoring == "fixed" else compounded
        rows.append(
            {
                "exit": name,
                "symbol": symbol,
                "trades": len(closed),
                "win_rate": float((per_unit > 0).mean()) if len(closed) else 0.0,
                "avg_trade": float(per_unit.mean()) if len(closed) else 0.0,
                "net_return": compounded,
                "fixed_return": fixed,
                "profitable": bool(profit > 0 and len(closed) > 0),
            }
        )
    return rows


def check_scoring(scoring: str) -> None:
    if scoring not in SCORINGS:
        raise ValueError(f"entry test scoring must be one of {sorted(SCORINGS)}, not {scoring!r}")


def run_entry_test(
    strategy: Strategy,
    universe: dict[str, pd.DataFrame],
    costs: CostModel | Mapping[str, CostModel],
    exit_bars: list[int],
    stop_atr: float,
    target_atr: float,
    atr_length: int,
    scoring: str = "compounded",
) -> EntryTest:
    check_scoring(scoring)
    rows = []
    for symbol, bars in universe.items():
        rows += entry_rows(
            strategy, symbol, bars, costs_for(costs, symbol), exit_bars, stop_atr, target_atr,
            atr_length, scoring,
        )  # fmt: skip
    return EntryTest(pd.DataFrame(rows))
