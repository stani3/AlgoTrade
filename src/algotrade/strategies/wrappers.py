"""Building blocks that modify or combine other strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.indicators import ewm_vol, sma

from .base import Strategy, bars_per_day, bars_per_year


@dataclass(frozen=True)
class VolTarget(Strategy):
    """Scale exposure so the position runs at ``annual_vol`` annualized volatility.

    Exposure = inner target * annual_vol / recent realized vol, capped at ``max_leverage``.
    Smaller in wild markets, bigger in calm ones: Carver's core sizing rule. The engine's own
    ``max_leverage`` must be at least this one or it will clip the result.
    """

    strategy: Strategy
    annual_vol: float = 0.25
    vol_days: float = 25.0
    max_leverage: float = 1.0

    name = "vol_target"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        span = self.vol_days * bars_per_day(bars.index)
        realized = ewm_vol(bars["close"].pct_change(), span) * np.sqrt(bars_per_year(bars.index))
        scale = (self.annual_vol / realized).fillna(0.0)
        target = self.strategy.target_position(bars) * scale
        return target.clip(-self.max_leverage, self.max_leverage)


@dataclass(frozen=True)
class TrendFilter(Strategy):
    """Only keep longs above the ``length``-bar SMA and shorts below it."""

    strategy: Strategy
    length: int = 200

    name = "trend_filter"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        target = self.strategy.target_position(bars)
        average = sma(bars["close"], self.length)
        allowed = ((target > 0) & (bars["close"] > average)) | (
            (target < 0) & (bars["close"] < average)
        )
        return target.where(allowed, 0.0)


@dataclass(frozen=True)
class Combine(Strategy):
    """Weighted blend of several strategies' exposures (Carver's forecast combination).

    Weights are normalized to sum to 1. ``multiplier`` is the forecast diversification
    multiplier: blended signals partly cancel, so scaling up restores the average size.
    """

    strategies: tuple[Strategy, ...]
    weights: tuple[float, ...] = ()
    multiplier: float = 1.0

    name = "combine"

    def __post_init__(self) -> None:
        if not self.strategies:
            raise ValueError("combine needs at least one strategy")
        if self.weights and len(self.weights) != len(self.strategies):
            raise ValueError("one weight per strategy")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        weights = np.asarray(self.weights or [1.0] * len(self.strategies), dtype="float64")
        weights = weights / weights.sum()
        blended = sum(
            w * s.target_position(bars) for w, s in zip(weights, self.strategies, strict=True)
        )
        return (blended * self.multiplier).clip(-1.0, 1.0)
