"""i008 US-open session momentum: follow a large 08:00-10:00 New York move for the session.

The two hours around the US open hold the 08:30 macro data releases and the 09:30 NYSE open,
the busiest moment of the US day, and the large orders that move price then are worked through
the session. On each New York weekday the rule measures the move from 08:00 to 10:00 New York
time against the scale of a two-hour return; at 10:00 it goes long after a rise of at least
``k`` scales (short after a fall of at least ``k`` scales) and holds for ``hold_bars`` hours,
with no stop and no target. It is flat the rest of the time.

Hours are read on the New York clock: bar open times are converted from UTC with the time-zone
database, so the decision falls at 14:00 UTC in summer and 15:00 UTC in winter. Daylight saving
changes at 02:00 on a Sunday, so the hours 07:00 to 21:00 of a New York weekday are always
consecutive whole UTC hours. Every value uses only bars up to the close it is decided on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from algotrade.strategies.base import Strategy

NEW_YORK = "America/New_York"
HOUR = pd.Timedelta(hours=1)
DECISION_HOUR = 9  # the bar opening 09:00 New York time closes at 10:00
MOVE_BARS = 2  # the move runs from the close of the 07:00 bar (08:00) to that of the 09:00 bar
FRIDAY = 4  # trading days are Monday (0) to Friday (4) on the New York calendar
MAX_HOLD = 12  # the longest trade ends at 21:00 New York time, long before the next decision


def new_york_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Bar open times on the New York clock (a timezone-naive index is read as UTC).

    Refuses bars that do not all open on whole UTC hours (30-minute bars, say): the rule looks
    up 1h bars by their opening hour, and those would be the wrong bars.
    """

    utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    if not (utc == utc.floor("h")).all():
        raise ValueError("i008_us_open_momentum needs 1h bars that open on whole UTC hours")
    return utc.tz_convert(NEW_YORK)


def _positive(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and 0 < value < math.inf


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


@dataclass(frozen=True)
class USOpenMomentum(Strategy):
    """Trade in the direction of a large move around the US open, from 10:00 New York time.

    * Trading day d: a New York weekday (Monday to Friday; US holidays are not excluded). Its
      decision bar is the 1h bar opening at 09:00 New York time, which closes at 10:00.
    * Opening move ``r_d = close(09:00 bar) / close(07:00 bar) - 1``, undefined (no trade) when
      the bar opening at 07:00, 08:00 or 09:00 New York time is missing.
    * Scale ``sigma_open = sigma_t * sqrt(2)``, where ``sigma_t`` is the standard deviation
      (ddof 1) of the close-to-close returns of the last ``vol_bars`` bars up to and including
      the decision bar (returns between consecutive bars of the data), undefined (no trade)
      until ``vol_bars`` returns exist.
    * At the decision bar's close: +1 if ``r_d >= k * sigma_open``, else -1 if
      ``r_d <= -k * sigma_open``, else 0. The tests are checked in that order, so the one case
      that meets both (no move at all after ``vol_bars`` identical returns) goes long.
    * The target keeps that side on the bars opening 09:00 .. (09 + hold_bars - 1):00 New York
      time and is 0 from the bar opening (09 + hold_bars):00, counted by the clock (a missing
      bar uses up its hour). With no decision bar there is no trade that day.
    """

    k: float = 1.0
    hold_bars: int = 6
    vol_bars: int = 720

    name = "i008_us_open_momentum"

    def __post_init__(self) -> None:
        if not _positive(self.k):
            raise ValueError("k must be positive and finite")
        if not _integer(self.hold_bars) or not 1 <= self.hold_bars <= MAX_HOLD:
            raise ValueError(f"hold_bars must be an integer from 1 to {MAX_HOLD}")
        if not _integer(self.vol_bars) or self.vol_bars < 2:
            raise ValueError("vol_bars must be an integer >= 2")

    def opening(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Per bar: ``decision`` (it opens at 09:00 New York time on a weekday), ``move``
        (``r_d`` on decision bars where it is defined, NaN elsewhere), ``scale``
        (``sigma_open`` on every bar, NaN until ``vol_bars`` returns exist) and ``side`` (the
        side a decision bar opens, 0 on every other bar)."""

        index = bars.index
        local = new_york_clock(index)
        decision = np.asarray((local.hour == DECISION_HOUR) & (local.dayofweek <= FRIDAY))
        close = bars["close"].astype("float64")
        # The bars opening 07:00 and 08:00 New York time, looked up by timestamp: on a weekday
        # they are always two and one UTC hours before the 09:00 bar.
        before = close.reindex(index - MOVE_BARS * HOUR).to_numpy()
        complete = decision & np.asarray((index - HOUR).isin(index))
        move = np.where(complete, close.to_numpy() / before - 1.0, np.nan)
        returns = close / close.shift(1) - 1.0
        sigma = returns.rolling(self.vol_bars, min_periods=self.vol_bars).std()
        scale = sigma.to_numpy() * math.sqrt(MOVE_BARS)
        threshold = self.k * scale
        side = np.select([move >= threshold, move <= -threshold], [1.0, -1.0], 0.0)
        return pd.DataFrame(
            {"decision": decision, "move": move, "scale": scale, "side": side}, index=index
        )

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        index = bars.index
        side = self.opening(bars)["side"]
        # Hours since the day's decision bar opened: 0 on it, 1 on the 10:00 bar, ...
        since = np.asarray(new_york_clock(index).hour) - DECISION_HOUR
        held = (since >= 0) & (since < self.hold_bars)
        decided = index - pd.to_timedelta(np.where(held, since, 0), unit="h")
        opened = side.reindex(decided, fill_value=0.0).to_numpy()
        return pd.Series(np.where(held, opened, 0.0), index=index)
