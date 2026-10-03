"""The stake: how big to trade, from Davey's Monte Carlo on out-of-sample trades.

Closed trades from the walk-forward's out-of-sample record (all symbols pooled) are resampled
into many one-year sequences at several position-size multipliers. The stake is the largest
multiplier that still meets all three of Davey's goals (risk of ruin, median drawdown,
return / drawdown); if none does, the strategy is not tradeable at any size.

It is never larger than the multiplier with the highest median one-year return: past that size
losses compound faster than gains, so more size means more drawdown for less return, however
much drawdown the limits allow.

A multiplier scales the strategy's exposure: 0.5 means half the spec's position (for a
``vol_target`` spec, half its annual volatility target).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.backtest.engine import BacktestResult
from algotrade.backtest.metrics import YEAR
from algotrade.backtest.report import MonteCarlo, monte_carlo


def closed_trade_returns(results: dict[str, BacktestResult]) -> np.ndarray:
    parts = [r.trades.loc[~r.trades["open"].astype(bool), "return"] for r in results.values()]
    return pd.concat(parts).to_numpy(dtype=float) if parts else np.array([])


def trades_per_year(results: dict[str, BacktestResult]) -> int:
    """Closed trades per year of a typical (median) symbol."""

    rates = []
    for result in results.values():
        index = result.ledger.index
        years = max((index[-1] - index[0]) / YEAR, 1 / 365.25)
        rates.append((~result.trades["open"].astype(bool)).sum() / years)
    return max(round(float(np.median(rates))), 1) if rates else 1


def limits(criteria) -> dict:
    return {
        "risk_of_ruin": criteria.get("validation.max_risk_of_ruin"),
        "median_max_dd": criteria.get("validation.max_median_drawdown"),
        "return_dd": criteria.get("validation.min_return_drawdown"),
    }


@dataclass
class Sizing:
    monte_carlo: MonteCarlo
    stake: float | None
    limits: dict

    @property
    def table(self) -> pd.DataFrame:
        return self.monte_carlo.table

    def passing(self, size: float) -> bool:
        return all(passed for *_, passed in self.monte_carlo.checks(self.limits, size))


def choose_stake(
    returns: np.ndarray,
    per_year: int,
    multipliers: list[float],
    runs: int,
    ruin: float,
    davey_limits: dict,
    seed: int = 0,
    max_stake: float | None = None,
) -> Sizing:
    """``max_stake`` is the most the instruments allow (leverage caps); None means no cap."""

    mc = monte_carlo(returns, per_year, runs, ruin, tuple(multipliers), seed)
    sizing = Sizing(mc, None, davey_limits)
    peak = mc.table["median_return"].idxmax()
    passing = [
        size
        for size in mc.table.index
        if size <= peak and sizing.passing(size) and (max_stake is None or size <= max_stake)
    ]
    sizing.stake = float(max(passing)) if passing else None
    return sizing
