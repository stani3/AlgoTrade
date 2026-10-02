"""i005 funding-settlement rebound: trade the hours right after an 8-hourly funding settlement.

Binance perpetuals settle funding at 00:00, 08:00 and 16:00 UTC between the positions open at
that instant. When the previous settlement was far from zero, traders dodging the payment (and
collectors earning it) sell the paying side into the next settlement and buy it back after it.
The rule takes the rebound leg: at the close of the 1h bar that ends at a settlement it goes long
after a previous rate of at least ``threshold`` (short after one of at most ``-threshold``) and
holds for ``hold_bars`` hours, with no stop and no target. It is flat the rest of the time.

Settlements are found from bar timestamps alone (many recorded rates are exactly zero), and the
previous rate is read from the two 1h bars that can hold it whichever side of the hour its
timestamp put it on (see ``algotrade.data.market.align_funding``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from algotrade.strategies.base import Strategy

HOUR = pd.Timedelta(hours=1)
EVERY = 8  # hours between scheduled settlements: 00:00, 08:00 and 16:00 UTC
MAX_HOLD = EVERY - 1  # a trade ends before the next decision bar closes, so trades never overlap
# The previous settlement (at H - 8h) is booked on the bar opening H - 9h or H - 8h, which are
# 8 and 7 hours before the decision bar opening at H - 1h.
FUNDING_LAGS = (8, 7)


def utc_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """The UTC hour each bar opens at (a timezone-naive index is read as UTC).

    Refuses bars that do not all open on whole hours (30-minute bars, say): the rule looks up
    1h bars by their opening hour, and those would be the wrong bars.
    """

    utc = index.tz_convert("UTC") if index.tz is not None else index
    if not (utc == utc.floor("h")).all():
        raise ValueError("i005_settlement_rebound needs 1h bars that open on whole UTC hours")
    return np.asarray(utc.hour)


@dataclass(frozen=True)
class SettlementRebound(Strategy):
    """Take the side of the previous funding settlement for ``hold_bars`` hours after the next.

    * Settlement hours H: 00:00, 08:00 and 16:00 UTC. The decision bar of H is the 1h bar
      opening at H - 1h (23:00, 07:00 or 15:00 UTC); it closes at H.
    * Funding proxy, looked up by timestamp:
      ``f_H = funding_rate(bar opening H - 9h) + funding_rate(bar opening H - 8h)``, undefined
      (no trade) when either bar is missing.
    * At the decision bar's close: +1 if ``f_H >= threshold``, -1 if ``f_H <= -threshold``,
      else 0.
    * The target keeps that side on the bars opening H - 1h, H, ..., H + (hold_bars - 2)h and is
      0 from the bar opening H + (hold_bars - 1)h, counted by the clock (a missing bar uses up
      its hour). With no decision bar there is no trade.

    On bars two or more hours apart (4h, daily) the two funding bars never both exist, so the
    rule never trades there.
    """

    threshold: float = 0.0005
    hold_bars: int = 2

    name = "i005_settlement_rebound"

    def __post_init__(self) -> None:
        if not 0 < self.threshold < math.inf:
            raise ValueError("threshold must be positive and finite")
        hold = self.hold_bars
        if isinstance(hold, bool) or not isinstance(hold, Integral) or not 1 <= hold <= MAX_HOLD:
            raise ValueError(f"hold_bars must be an integer from 1 to {MAX_HOLD}")

    def settlements(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Per bar: ``decision`` (it opens at 23:00, 07:00 or 15:00 UTC), ``funding`` (``f_H`` on
        decision bars where it is defined, NaN elsewhere) and ``side`` (the side a decision bar
        opens, 0 on every other bar)."""

        index = bars.index
        decision = utc_hours(index) % EVERY == EVERY - 1
        funding = bars["funding_rate"]
        early, late = (
            funding.reindex(index - lag * HOUR).to_numpy("float64") for lag in FUNDING_LAGS
        )
        proxy = np.where(decision, early + late, np.nan)
        side = np.select([proxy >= self.threshold, proxy <= -self.threshold], [1.0, -1.0], 0.0)
        return pd.DataFrame({"decision": decision, "funding": proxy, "side": side}, index=index)

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        side = self.settlements(bars)["side"]
        # Hours since the latest decision bar opened: 0 on it, 1 on the settlement bar, ... 7.
        since = (utc_hours(bars.index) + 1) % EVERY
        decided = bars.index - pd.to_timedelta(since, unit="h")
        opened = side.reindex(decided, fill_value=0.0).to_numpy()
        return pd.Series(np.where(since < self.hold_bars, opened, 0.0), index=bars.index)
