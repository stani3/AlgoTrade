"""i002 liquidation shock fade: fade a bar's move of at least ``z_entry`` prior volatilities.

A close-to-close return is scored against the volatility known at the previous close. A move of
``z_entry`` or more is taken to be forced flow (liquidations, stop cascades) that partly comes
back: the trade fades it at the shock bar's close and holds it for exactly ``hold_bars`` bars,
with no stop, no target and no volatility scaling. A new shock during a trade restarts the clock
in its own direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from algotrade.indicators import ewm_vol
from algotrade.strategies.base import Strategy, bars_per_day


@dataclass(frozen=True)
class ShockFade(Strategy):
    """Fade volatility shocks for a fixed number of bars.

    * Return ``r_t = close_t / close_{t-1} - 1``; volatility ``sigma_t = ewm_vol(r, span)`` with
      ``span = vol_days x bars per day`` (``ewm_vol``'s own warm-up).
    * Bar t is a shock when ``|r_t / sigma_{t-1}| >= z_entry``; never while ``sigma_{t-1}`` is
      undefined or zero. Its trade direction is ``-sign(r_t)``.
    * Target at bar t: the direction of the most recent shock s with ``t - hold_bars < s <= t``,
      else 0. With ``allow_short`` false, short targets become 0.
    """

    z_entry: float = 3.0
    hold_bars: int = 6
    vol_days: float = 25.0
    allow_short: bool = True

    name = "i002_shock_fade"

    def __post_init__(self) -> None:
        if not 0 < self.z_entry < math.inf:
            raise ValueError("z_entry must be positive and finite")
        hold = self.hold_bars
        if isinstance(hold, bool) or not isinstance(hold, Integral) or hold < 1:
            raise ValueError("hold_bars must be an integer >= 1")
        if not 0 < self.vol_days < math.inf:
            raise ValueError("vol_days must be positive and finite")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        returns = close / close.shift() - 1
        span = self.vol_days * bars_per_day(bars.index)
        known_vol = ewm_vol(returns, span).shift()  # the volatility known at the previous close
        shock = (known_vol > 0) & ((returns / known_vol).abs() >= self.z_entry)
        # The most recent shock at or before each bar: its trade direction and its age in bars.
        direction = (-np.sign(returns)).where(shock).ffill()
        bar = pd.Series(np.arange(len(bars), dtype="float64"), index=bars.index)
        age = bar - bar.where(shock).ffill()
        target = direction.where(age < self.hold_bars, 0.0).rename(None)
        return target if self.allow_short else target.clip(lower=0.0)
