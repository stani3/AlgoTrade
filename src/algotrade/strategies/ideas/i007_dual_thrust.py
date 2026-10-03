"""i007 Dual Thrust intraday breakout: trade a break of k ranges from the UTC day's open.

Michael Chalek's Dual Thrust on 1h bars. The range of the previous ``range_days`` complete UTC
days, ``R = max(HH - LC, HC - LL)``, scaled by ``k`` and laid either side of today's 00:00 UTC
open gives a buy line and a sell line. An hourly close above the buy line goes long, one below
the sell line goes short; a close back through the open is a stop, a close beyond the opposite
line reverses once. There is at most one long and one short entry per UTC day, and the rule is
always flat at the close of the 23:00 UTC bar.

Days are UTC calendar days of the bar open times, and every value uses only bars up to the
close it is decided on: the range only finished days, the lines only today's first bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from numba import njit

from algotrade.strategies.base import Strategy

HOURS = 24  # hourly bars in a complete UTC day
LAST_HOUR = HOURS - 1  # the 23:00 UTC bar: flat at its close, no entries on it
EPOCH = pd.Timestamp("1970-01-01")
DAY = pd.Timedelta(days=1)
PRICES = ("open", "high", "low", "close")


def utc_clock(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """The UTC day (days since 1970-01-01) and hour each bar opens at.

    A timezone-naive index is read as UTC. Refuses bars that do not all open on whole hours
    (30-minute bars, say): days are counted in whole hourly bars.
    """

    utc = index.tz_convert(None) if index.tz is not None else index
    if not (utc == utc.floor("h")).all():
        raise ValueError("i007_dual_thrust needs bars that open on whole UTC hours")
    day = np.asarray((utc - EPOCH) // DAY, dtype="int64")
    return day, np.asarray(utc.hour, dtype="int64")


@njit(cache=True)
def _dual_thrust(
    day: np.ndarray,
    active: np.ndarray,
    close: np.ndarray,
    today_open: np.ndarray,
    buy: np.ndarray,
    sell: np.ndarray,
) -> np.ndarray:
    """The card's state machine on each bar's close (``active``: a trading day, not 23:00)."""
    n = close.shape[0]
    target = np.zeros(n)
    position = 0.0
    long_used = False
    short_used = False
    for t in range(n):
        if t == 0 or day[t] != day[t - 1]:  # every UTC day starts flat with both sides unused
            position = 0.0
            long_used = False
            short_used = False
        if not active[t]:
            position = 0.0
        elif position == 0.0:
            if close[t] > buy[t] and not long_used:
                position = 1.0
                long_used = True
            elif close[t] < sell[t] and not short_used:
                position = -1.0
                short_used = True
        elif position > 0.0:
            if close[t] < sell[t] and not short_used:  # stop and reverse
                position = -1.0
                short_used = True
            elif close[t] < today_open[t]:  # the move from the open has failed
                position = 0.0
        else:
            if close[t] > buy[t] and not long_used:
                position = 1.0
                long_used = True
            elif close[t] > today_open[t]:
                position = 0.0
        target[t] = position
    return target


def _positive(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and 0 < value < math.inf


@dataclass(frozen=True)
class DualThrust(Strategy):
    """Dual Thrust on 1h bars, anchored to the UTC day.

    * Day d: the bars opening 00:00 .. 23:00 UTC on date d. ``O_d`` is the open of its 00:00
      bar; without that bar, no trading on d.
    * Range from days d - N .. d - 1 (N = ``range_days``), each with all 24 hourly bars,
      otherwise no trading on d: HH and LL the highest high and lowest low of their bars, HC
      and LC the highest and lowest of their 23:00 closes, ``R_d = max(HH - LC, HC - LL)``. No
      trading on d if ``R_d <= 0``.
    * Lines ``Buy_d = O_d + k R_d`` and ``Sell_d = O_d - k R_d``.
    * On each close of a trading day, starting flat with both sides unused: the 23:00 bar is
      flat. Flat: long on ``close > Buy_d`` if no long yet today, else short on
      ``close < Sell_d`` if no short yet. Long: reverse to short on ``close < Sell_d`` if no
      short yet today, else flat on ``close < O_d``. Short: the mirror image.
    """

    k: float = 0.5
    range_days: int = 1

    name = "i007_dual_thrust"

    def __post_init__(self) -> None:
        if not _positive(self.k):
            raise ValueError("k must be positive and finite")
        days = self.range_days
        if isinstance(days, bool) or not isinstance(days, Integral) or days < 1:
            raise ValueError("range_days must be an integer >= 1")

    def levels(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Per bar, for its UTC day: ``open`` (``O_d``, NaN without the 00:00 bar), ``range``
        (``R_d``, NaN while the previous days are not all complete), the ``buy`` and ``sell``
        lines and whether the day is ``trading`` (lines NaN when it is not)."""

        day, hour = utc_clock(bars.index)
        frame = pd.DataFrame(
            {"day": day, "hour": hour} | {name: bars[name].to_numpy("float64") for name in PRICES}
        )
        by_day = frame.groupby("day")
        daily = pd.DataFrame(
            {
                "bars": by_day.size(),
                "high": by_day["high"].max(),
                "low": by_day["low"].min(),
                "close": frame[frame["hour"] == LAST_HOUR].set_index("day")["close"],
            }
        )
        # Every calendar day from the first to the last, so a day without bars counts as one.
        calendar = np.arange(day.min(), day.max() + 1) if day.size else day
        before = daily.reindex(calendar).shift(1)  # the row of day d holds day d - 1
        window = before.rolling(self.range_days, min_periods=self.range_days)
        lowest, highest = window.min(), window.max()
        complete = lowest["bars"] == HOURS  # each of days d - N .. d - 1 has all 24 bars
        span = np.maximum(highest["high"] - lowest["close"], highest["close"] - lowest["low"])
        day_range = span.where(complete).reindex(day).to_numpy()
        opening = frame[frame["hour"] == 0].set_index("day")["open"]
        today_open = opening.reindex(day).to_numpy()
        trading = (day_range > 0) & ~np.isnan(today_open)  # a NaN range never trades
        return pd.DataFrame(
            {
                "open": today_open,
                "range": day_range,
                "buy": np.where(trading, today_open + self.k * day_range, np.nan),
                "sell": np.where(trading, today_open - self.k * day_range, np.nan),
                "trading": trading,
            },
            index=bars.index,
        )

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        day, hour = utc_clock(bars.index)
        lines = self.levels(bars)
        target = _dual_thrust(
            day,
            lines["trading"].to_numpy(dtype=bool) & (hour != LAST_HOUR),
            bars["close"].to_numpy("float64"),
            lines["open"].to_numpy("float64"),
            lines["buy"].to_numpy("float64"),
            lines["sell"].to_numpy("float64"),
        )
        return pd.Series(target, index=bars.index)
