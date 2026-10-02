"""Reference strategies that are not trading ideas in their own right."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy


@dataclass(frozen=True)
class BuyAndHold(Strategy):
    """Always fully long: the benchmark every idea is compared with, and a duplicate check."""

    name = "buy_and_hold"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)
