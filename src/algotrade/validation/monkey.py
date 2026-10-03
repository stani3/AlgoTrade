"""Monkey test: does the strategy beat random trading with the same habits?

* Bracket strategies: random entry signals with the strategy's own signal frequency and
  long/short mix, traded with its own stops, targets, cooldowns, time exit and kill switch.
* Position strategies: the strategy's own exposure series, circularly shifted against the
  market by a random offset. Turnover, exposure and holding times stay; the timing is lost.

The score is the share of monkeys whose median Sharpe ratio across symbols is below the
strategy's. Davey wants a strategy to beat about 90% of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.backtest.costs import CostModel, costs_for
from algotrade.backtest.metrics import periods_per_year
from algotrade.parallel import Pool, run
from algotrade.strategies import BracketStrategy, Strategy

from .fast import bracket_returns, position_returns, sharpe


@dataclass
class MonkeyTest:
    real: float  # median Sharpe across symbols
    monkeys: np.ndarray  # median Sharpe across symbols of each monkey run

    @property
    def percentile(self) -> float:
        return float((self.monkeys < self.real).mean()) if len(self.monkeys) else 0.0


@dataclass(frozen=True)
class MonkeyDraws:
    """One symbol's share of the random numbers.

    The test draws from one generator, symbol after symbol. To run symbols in any order (or in
    parallel) and still get the same monkeys, each symbol gets the generator's state where its
    draws begin (bracket strategies: ``runs`` x bars uniform draws) or its shift offsets drawn
    up front (position strategies).
    """

    state: dict | None = None
    offsets: np.ndarray | None = None


def plan_draws(bracket: bool, lengths: list[int], runs: int, seed: int) -> list[MonkeyDraws]:
    """Each symbol's draws (symbols with ``lengths`` bars, in universe order)."""

    rng = np.random.default_rng(seed)
    plans = []
    for n in lengths:
        if bracket:
            plans.append(MonkeyDraws(state=rng.bit_generator.state))
            rng.bit_generator.advance(runs * n)  # rng.random(n) takes one 64-bit draw per value
        else:
            low, high = max(n // 10, 1), max(n - n // 10, 2)
            plans.append(
                MonkeyDraws(offsets=np.array([int(rng.integers(low, high)) for _ in range(runs)]))
            )
    return plans


def _bracket_rules(strategy: BracketStrategy) -> dict:
    return {
        "cooldown_win": strategy.cooldown_win,
        "cooldown_loss": strategy.cooldown_loss,
        "kill_drawdown": strategy.kill_drawdown,
        "max_bars": getattr(strategy, "max_bars", 0),
    }


def monkey_symbol(
    strategy: Strategy,
    bars: pd.DataFrame,
    costs: CostModel,
    runs: int,
    draws: MonkeyDraws,
    signals=None,
) -> tuple[float, np.ndarray]:
    """The strategy's Sharpe ratio on one symbol and that of each of its ``runs`` monkeys.

    ``signals`` are the bracket strategy's signals on ``bars`` if already computed.
    """

    ppy = periods_per_year(bars.index)
    n = len(bars)
    monkeys = np.zeros(runs)
    if isinstance(strategy, BracketStrategy):
        signals = signals if signals is not None else strategy.signals(bars)
        long = signals.long.fillna(False).to_numpy(dtype=bool)
        short = signals.short.fillna(False).to_numpy(dtype=bool)
        stop, target = signals.stop_dist.to_numpy(), signals.target_dist.to_numpy()
        rules = _bracket_rules(strategy)
        real = sharpe(bracket_returns(bars, long, short, stop, target, costs, **rules), ppy)
        p_long, p_short = long.mean(), short.mean()
        rng = np.random.default_rng()
        rng.bit_generator.state = draws.state
        for run_ in range(runs):
            draw = rng.random(n)
            fake_long = draw < p_long
            fake_short = (draw >= p_long) & (draw < p_long + p_short)
            returns = bracket_returns(bars, fake_long, fake_short, stop, target, costs, **rules)
            monkeys[run_] = sharpe(returns, ppy)
    else:
        exposure = strategy.target_position(bars).to_numpy(dtype="float64")
        real = sharpe(position_returns(bars, exposure, costs), ppy)
        for run_, offset in enumerate(draws.offsets):
            shifted = np.roll(exposure, int(offset))
            monkeys[run_] = sharpe(position_returns(bars, shifted, costs), ppy)
    return real, monkeys


def _monkey_job(args: tuple) -> tuple[float, np.ndarray]:
    return monkey_symbol(*args)


def combine(outcomes: list[tuple[float, np.ndarray]], runs: int) -> MonkeyTest:
    """Medians across symbols of the per-symbol results (in universe order)."""

    real = [outcome[0] for outcome in outcomes]
    monkeys = np.column_stack([o[1] for o in outcomes]) if outcomes else np.zeros((runs, 0))
    return MonkeyTest(real=float(np.median(real)), monkeys=np.median(monkeys, axis=1))


def monkey_test(
    strategy: Strategy,
    universe: dict[str, pd.DataFrame],
    costs: CostModel | Mapping[str, CostModel],
    runs: int,
    seed: int = 0,
    pool: Pool | None = None,
) -> MonkeyTest:
    bracket = isinstance(strategy, BracketStrategy)
    draws = plan_draws(bracket, [len(bars) for bars in universe.values()], runs, seed)
    jobs = [
        (strategy, bars, costs_for(costs, symbol), runs, plan)
        for (symbol, bars), plan in zip(universe.items(), draws, strict=True)
    ]
    return combine(run(_monkey_job, jobs, pool), runs)
