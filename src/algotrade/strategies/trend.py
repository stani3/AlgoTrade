"""Trend-following rules built on classic indicators."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algotrade.indicators import adx, macd, sma, supertrend

from .base import Strategy, long_short


@dataclass(frozen=True)
class MACDStrategy(Strategy):
    """Long while the MACD line is above its signal line (or above zero with ``mode='zero'``)."""

    fast: int = 12
    slow: int = 26
    signal: int = 9
    mode: str = "signal"
    allow_short: bool = True

    name = "macd"

    def __post_init__(self) -> None:
        if not 0 < self.fast < self.slow:
            raise ValueError("need 0 < fast < slow")
        if self.mode not in ("signal", "zero"):
            raise ValueError("mode must be 'signal' or 'zero'")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        m = macd(bars["close"], self.fast, self.slow, self.signal)
        ref = m["signal"] if self.mode == "signal" else 0.0
        valid = m["signal"].notna()
        return long_short((m["macd"] > ref) & valid, (m["macd"] < ref) & valid, self.allow_short)


@dataclass(frozen=True)
class SuperTrendStrategy(Strategy):
    """Follow the SuperTrend direction (ATR trailing band that flips on a close through it)."""

    length: int = 10
    mult: float = 3.0
    allow_short: bool = True

    name = "supertrend"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        st = supertrend(bars["high"], bars["low"], bars["close"], self.length, self.mult)
        return long_short(st["direction"] > 0, st["direction"] < 0, self.allow_short)


@dataclass(frozen=True)
class ADXTrendStrategy(Strategy):
    """Trade the DI direction only when ADX says a trend exists; flat in chop.

    Optionally require price on the same side of a moving average (``ma_length`` > 0).
    """

    length: int = 14
    threshold: float = 25.0
    ma_length: int = 0
    allow_short: bool = True

    name = "adx_trend"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        d = adx(bars["high"], bars["low"], bars["close"], self.length)
        trending = d["adx"] > self.threshold
        up = trending & (d["plus_di"] > d["minus_di"])
        down = trending & (d["minus_di"] > d["plus_di"])
        if self.ma_length:
            average = sma(bars["close"], self.ma_length)
            up &= bars["close"] > average
            down &= bars["close"] < average
        return long_short(up, down, self.allow_short)
