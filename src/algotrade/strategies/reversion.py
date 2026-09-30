"""Mean-reversion rules: fade stretched moves, exit on the snap back."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algotrade.indicators import bollinger, rsi, sma

from .base import Strategy, entry_exit


def _trend_gate(bars: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series]:
    """(longs allowed, shorts allowed). ``length`` 0 disables the gate."""
    if not length:
        everywhere = pd.Series(True, index=bars.index)
        return everywhere, everywhere
    average = sma(bars["close"], length)
    return bars["close"] > average, bars["close"] < average


@dataclass(frozen=True)
class RSIReversion(Strategy):
    """Buy oversold / sell overbought RSI, exit when RSI crosses ``exit_level``.

    ``trend_filter`` > 0 only buys dips above that SMA and sells rips below it (the classic
    Connors setup is length=2, lower=10, upper=90, trend_filter=200).
    """

    length: int = 14
    lower: float = 30.0
    upper: float = 70.0
    exit_level: float = 50.0
    trend_filter: int = 0
    allow_short: bool = True

    name = "rsi_reversion"

    def __post_init__(self) -> None:
        if not self.lower < self.exit_level < self.upper:
            raise ValueError("need lower < exit_level < upper")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        value = rsi(bars["close"], self.length)
        longs_ok, shorts_ok = _trend_gate(bars, self.trend_filter)
        return entry_exit(
            enter_long=(value < self.lower) & longs_ok,
            exit_long=value > self.exit_level,
            enter_short=(value > self.upper) & shorts_ok,
            exit_short=value < self.exit_level,
            allow_short=self.allow_short,
        )


@dataclass(frozen=True)
class BollingerReversion(Strategy):
    """Fade closes outside the band, exit at the middle band."""

    length: int = 20
    mult: float = 2.0
    trend_filter: int = 0
    allow_short: bool = True

    name = "bollinger_reversion"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        bands = bollinger(close, self.length, self.mult)
        longs_ok, shorts_ok = _trend_gate(bars, self.trend_filter)
        return entry_exit(
            enter_long=(close < bands["lower"]) & longs_ok,
            exit_long=close > bands["mid"],
            enter_short=(close > bands["upper"]) & shorts_ok,
            exit_short=close < bands["mid"],
            allow_short=self.allow_short,
        )
