"""Glue shared by the command-line scripts: load bars, build costs, run a strategy."""

from __future__ import annotations

import pandas as pd

from algotrade.config import load_settings
from algotrade.data.market import load_market
from algotrade.strategies import BracketStrategy, Strategy

from .costs import EXCHANGE_COSTS, CostModel
from .engine import BacktestResult, run_backtest


def cost_model(
    exchange: str,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
    include_funding: bool = True,
) -> CostModel:
    base = EXCHANGE_COSTS[exchange]
    return CostModel(
        fee_bps=base.fee_bps if fee_bps is None else fee_bps,
        slippage_bps=base.slippage_bps if slippage_bps is None else slippage_bps,
        include_funding=include_funding,
    )


def load_bars(
    exchange: str, base: str, timeframe: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    bars = load_market(load_settings().data_paths.raw, exchange, base.strip(), timeframe)
    if start:
        bars = bars[bars.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        bars = bars[bars.index < pd.Timestamp(end, tz="UTC")]
    return bars


def backtest(
    strategy: Strategy, bars: pd.DataFrame, costs: CostModel, max_leverage: float = 1.0
) -> BacktestResult:
    """Run any strategy; bracket strategies go through the intrabar simulator."""
    if isinstance(strategy, BracketStrategy):
        return strategy.simulate(bars, costs, leverage=max_leverage)
    return run_backtest(bars, strategy.target_position(bars), costs, max_leverage=max_leverage)
