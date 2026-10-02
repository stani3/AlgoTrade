"""Numpy-only net returns for tests that need thousands of backtests (monkeys).

They reproduce the engine's and the bracket simulator's per-bar net returns without building
ledgers or trade lists. The position version leaves out the liquidation rule, which only matters
for exposures far above 1x.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algotrade.backtest.bracket import _simulate
from algotrade.backtest.costs import CostModel


def _floats(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="float64")


def _distances(values: pd.Series | np.ndarray) -> np.ndarray:
    """Missing distances mean "no signal" (as in the simulator); infinite ones mean no level."""

    return np.nan_to_num(_floats(values), nan=0.0, posinf=np.inf)


def funding_rates(bars: pd.DataFrame, costs: CostModel) -> np.ndarray:
    if costs.include_funding and "funding_rate" in bars:
        return _floats(bars["funding_rate"])
    return np.zeros(len(bars))


def position_returns(
    bars: pd.DataFrame, target: np.ndarray, costs: CostModel, max_leverage: float = 1.0
) -> np.ndarray:
    """Net returns of holding ``target`` (decided at each close) through the next bar."""

    target = np.clip(np.nan_to_num(_floats(target)), -max_leverage, max_leverage)
    position = np.concatenate(([0.0], target[:-1]))
    close = _floats(bars["close"])
    move = np.zeros_like(close)
    move[1:] = close[1:] / close[:-1] - 1.0
    turnover = np.abs(np.diff(position, prepend=0.0))
    return position * move - turnover * costs.rate - position * funding_rates(bars, costs)


def bracket_returns(
    bars: pd.DataFrame,
    long: np.ndarray,
    short: np.ndarray,
    stop_dist: np.ndarray,
    target_dist: np.ndarray,
    costs: CostModel,
    cooldown_win: int = 0,
    cooldown_loss: int = 0,
    kill_drawdown: float = 1.0,
    leverage: float = 1.0,
    max_bars: int = 0,
) -> np.ndarray:
    """Net returns of the bracket simulator for raw signal arrays."""

    equity = _simulate(
        _floats(bars["open"]),
        _floats(bars["high"]),
        _floats(bars["low"]),
        _floats(bars["close"]),
        funding_rates(bars, costs),
        np.asarray(long, dtype=bool),
        np.asarray(short, dtype=bool),
        _distances(stop_dist),
        _distances(target_dist),
        int(cooldown_win),
        int(cooldown_loss),
        float(kill_drawdown),
        costs.fee_bps / 10_000,
        costs.slippage_bps / 10_000,
        float(leverage),
        int(max_bars),
    )[4]
    previous = np.concatenate(([1.0], equity[:-1]))
    return np.where(previous > 0, (equity - previous) / np.where(previous > 0, previous, 1.0), 0.0)


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Annualised Sharpe ratio with the engine's convention (sample std, zero if flat)."""

    std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    return 0.0 if not std > 0 else float(np.mean(returns) / std * np.sqrt(periods_per_year))
