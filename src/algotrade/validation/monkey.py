"""Monkey test: does the strategy beat random trading with the same habits?

* Bracket strategies: random entry signals with the strategy's own signal frequency and
  long/short mix, traded with its own stops, targets, cooldowns, time exit and kill switch.
* Position strategies: the strategy's own exposure series, circularly shifted against the
  market by a random offset. Turnover, exposure and holding times stay; the timing is lost.

The score is the share of monkeys whose median Sharpe ratio across symbols is below the
strategy's. Davey wants a strategy to beat about 90% of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.backtest.costs import CostModel
from algotrade.backtest.metrics import periods_per_year
from algotrade.strategies import BracketStrategy, Strategy

from .fast import bracket_returns, position_returns, sharpe


@dataclass
class MonkeyTest:
    real: float  # median Sharpe across symbols
    monkeys: np.ndarray  # median Sharpe across symbols of each monkey run

    @property
    def percentile(self) -> float:
        return float((self.monkeys < self.real).mean()) if len(self.monkeys) else 0.0


def _bracket_rules(strategy: BracketStrategy) -> dict:
    return {
        "cooldown_win": strategy.cooldown_win,
        "cooldown_loss": strategy.cooldown_loss,
        "kill_drawdown": strategy.kill_drawdown,
        "max_bars": getattr(strategy, "max_bars", 0),
    }


def monkey_test(
    strategy: Strategy,
    universe: dict[str, pd.DataFrame],
    costs: CostModel,
    runs: int,
    seed: int = 0,
) -> MonkeyTest:
    rng = np.random.default_rng(seed)
    real = []
    monkeys = np.zeros((runs, len(universe)))
    for column, bars in enumerate(universe.values()):
        ppy = periods_per_year(bars.index)
        n = len(bars)
        if isinstance(strategy, BracketStrategy):
            signals = strategy.signals(bars)
            long = signals.long.fillna(False).to_numpy(dtype=bool)
            short = signals.short.fillna(False).to_numpy(dtype=bool)
            stop, target = signals.stop_dist.to_numpy(), signals.target_dist.to_numpy()
            rules = _bracket_rules(strategy)
            real.append(
                sharpe(bracket_returns(bars, long, short, stop, target, costs, **rules), ppy)
            )
            p_long, p_short = long.mean(), short.mean()
            for run in range(runs):
                draw = rng.random(n)
                fake_long = draw < p_long
                fake_short = (draw >= p_long) & (draw < p_long + p_short)
                returns = bracket_returns(bars, fake_long, fake_short, stop, target, costs, **rules)
                monkeys[run, column] = sharpe(returns, ppy)
        else:
            exposure = strategy.target_position(bars).to_numpy(dtype="float64")
            real.append(sharpe(position_returns(bars, exposure, costs), ppy))
            low, high = max(n // 10, 1), max(n - n // 10, 2)
            for run in range(runs):
                shifted = np.roll(exposure, int(rng.integers(low, high)))
                monkeys[run, column] = sharpe(position_returns(bars, shifted, costs), ppy)
    return MonkeyTest(real=float(np.median(real)), monkeys=np.median(monkeys, axis=1))
