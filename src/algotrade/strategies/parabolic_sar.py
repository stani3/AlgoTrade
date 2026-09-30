"""Parabolic SAR trend following: long while close is above the SAR, short while below."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algotrade.indicators import parabolic_sar

from .base import Strategy, long_short


@dataclass(frozen=True)
class ParabolicSARStrategy(Strategy):
    step: float = 0.02
    max_step: float = 0.2
    allow_short: bool = True

    name = "parabolic_sar"

    def __post_init__(self) -> None:
        if self.step <= 0 or self.max_step <= 0:
            raise ValueError("step and max_step must be positive")
        if self.step > self.max_step:
            raise ValueError("step must be <= max_step")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        psar = parabolic_sar(bars["high"], bars["low"], self.step, self.max_step)
        return long_short(bars["close"] >= psar, bars["close"] < psar, self.allow_short)
