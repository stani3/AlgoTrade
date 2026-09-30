"""Continuous forecast rules from Carver's *Systematic Trading*.

Each rule produces a raw forecast, scales it so its average absolute value is 10 (capped at
+-20), and maps it to exposure in [-1, 1]. Strength of conviction sets position size, instead of
all-in/all-out switching. Wrap them in ``VolTarget`` to size by risk the way the book does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.indicators import ema

from .base import Strategy, bars_per_day, bars_per_year, scale_forecast


def _long_only(target: pd.Series, allow_short: bool) -> pd.Series:
    return target if allow_short else target.clip(lower=0.0)


@dataclass(frozen=True)
class EWMAC(Strategy):
    """Exponentially weighted moving-average crossover, normalized by price volatility.

    Carver's standard speeds are 2/8, 4/16, 8/32, 16/64, 32/128 and 64/256 (in daily bars).
    """

    fast: int = 16
    slow: int = 64
    vol_days: float = 25.0
    allow_short: bool = True

    name = "ewmac"

    def __post_init__(self) -> None:
        if not 0 < self.fast < self.slow:
            raise ValueError("need 0 < fast < slow")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        span = self.vol_days * bars_per_day(bars.index)
        price_vol = close.diff().ewm(span=span, adjust=False, min_periods=int(span)).std()
        raw = (ema(close, self.fast) - ema(close, self.slow)) / price_vol
        return _long_only(scale_forecast(raw, min_periods=self.slow), self.allow_short)


@dataclass(frozen=True)
class CarverBreakout(Strategy):
    """Where the close sits inside its ``lookback``-bar range, smoothed over lookback / 4."""

    lookback: int = 80
    allow_short: bool = True

    name = "carver_breakout"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        high = close.rolling(self.lookback, min_periods=self.lookback).max()
        low = close.rolling(self.lookback, min_periods=self.lookback).min()
        raw = 40 * (close - (high + low) / 2) / (high - low)
        smooth = raw.ewm(span=max(self.lookback // 4, 1), adjust=False).mean()
        return _long_only(scale_forecast(smooth, min_periods=self.lookback), self.allow_short)


@dataclass(frozen=True)
class FundingCarry(Strategy):
    """Perpetual-futures carry: be on the side that receives funding, sized by its strength.

    Positive funding means longs pay shorts, so the forecast is minus the smoothed, annualized
    funding rate divided by annualized volatility (a carry "Sharpe").
    """

    span_days: float = 30.0
    vol_days: float = 25.0
    allow_short: bool = True

    name = "funding_carry"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        per_year = bars_per_year(bars.index)
        per_day = bars_per_day(bars.index)
        span = self.span_days * per_day
        annual_funding = (
            bars["funding_rate"].ewm(span=span, adjust=False, min_periods=int(span)).mean()
            * per_year
        )
        vol_span = self.vol_days * per_day
        returns = bars["close"].pct_change()
        annual_vol = returns.ewm(span=vol_span, adjust=False, min_periods=int(vol_span)).std()
        raw = -annual_funding / (annual_vol * np.sqrt(per_year))
        return _long_only(scale_forecast(raw, min_periods=int(span)), self.allow_short)
