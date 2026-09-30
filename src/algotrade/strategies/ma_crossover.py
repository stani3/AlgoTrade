"""Moving-average crossover: long while the fast average is above the slow one."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algotrade.indicators import ema, sma

from .base import Strategy, long_short


@dataclass(frozen=True)
class MovingAverageCrossover(Strategy):
    fast: int = 20
    slow: int = 100
    kind: str = "sma"
    allow_short: bool = True

    name = "ma_crossover"

    def __post_init__(self) -> None:
        if not 0 < self.fast < self.slow:
            raise ValueError("need 0 < fast < slow")
        if self.kind not in ("sma", "ema"):
            raise ValueError("kind must be 'sma' or 'ema'")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        average = sma if self.kind == "sma" else ema
        fast = average(bars["close"], self.fast)
        slow = average(bars["close"], self.slow)
        return long_short(fast > slow, fast < slow, self.allow_short)
