"""Davey's entry test: is the entry any good on its own, whatever the exit?

The strategy's entries are traded with exits that know nothing about the strategy: a fixed
number of bars (market-on-close), and a fixed ATR stop and target. A good entry makes money on
most (exit, market) combinations; Davey looks for roughly 70% of them profitable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.backtest.bracket import BracketSignals, simulate_bracket
from algotrade.backtest.costs import CostModel
from algotrade.indicators import atr
from algotrade.strategies import BracketStrategy, Strategy


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


def run_entry_test(
    strategy: Strategy,
    universe: dict[str, pd.DataFrame],
    costs: CostModel,
    exit_bars: list[int],
    stop_atr: float,
    target_atr: float,
    atr_length: int,
) -> EntryTest:
    rows = []
    for symbol, bars in universe.items():
        long, short = entry_signals(strategy, bars)
        volatility = atr(bars["high"], bars["low"], bars["close"], atr_length)
        never = pd.Series(np.inf, index=bars.index)
        exits = {f"{n} bars": (never, never, n) for n in exit_bars}
        exits[f"{stop_atr:g}/{target_atr:g} ATR bracket"] = (
            stop_atr * volatility,
            target_atr * volatility,
            0,
        )
        for name, (stop, target, max_bars) in exits.items():
            signals = BracketSignals(long, short, stop, target)
            result = simulate_bracket(bars, signals, costs, max_bars=max_bars)
            closed = result.trades[~result.trades["open"].astype(bool)]
            net = float(result.equity.iloc[-1] - 1.0)
            rows.append(
                {
                    "exit": name,
                    "symbol": symbol,
                    "trades": len(closed),
                    "win_rate": float((closed["return"] > 0).mean()) if len(closed) else 0.0,
                    "avg_trade": float(closed["return"].mean()) if len(closed) else 0.0,
                    "net_return": net,
                    "profitable": bool(net > 0 and len(closed) > 0),
                }
            )
    return EntryTest(pd.DataFrame(rows))
