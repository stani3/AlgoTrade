"""Time-series momentum: long if price is above its level ``lookback`` bars ago."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy, long_short


@dataclass(frozen=True)
class MomentumStrategy(Strategy):
    lookback: int = 20
    allow_short: bool = True

    name = "momentum"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        change = bars["close"].pct_change(periods=self.lookback)
        return long_short(change > 0, change < 0, self.allow_short)
