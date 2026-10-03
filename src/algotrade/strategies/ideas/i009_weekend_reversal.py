"""i009 weekend move reversal: fade a large Friday-to-Sunday move as the traditional markets reopen.

From Friday 22:00 to Sunday 22:00 UTC crypto keeps trading while CME crypto futures, US equities
and the banks are closed, so a weekend move meets fewer liquidity providers and part of it is a
price concession that is competed away when they return. Once a week, at the close of the 1h bar
opening Sunday 21:00 UTC, the rule measures the weekend move against the scale of a 48-hour
return; it goes short after a rise of at least ``min_move`` scales (long after a fall of at
least ``min_move`` scales) and holds for ``hold_bars`` hours, with no stop and no target. It is
flat the rest of the time.

All times are UTC, which has no daylight saving, so the weekend is always 48 whole hours. Every
value uses only bars up to the close it is decided on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from algotrade.strategies.base import Strategy

HOUR = pd.Timedelta(hours=1)
WEEK = 7 * 24  # hours in a week
SUNDAY = 6  # pandas day of the week: Monday 0 .. Sunday 6
DECISION_HOUR = 21  # the bar opening Sunday 21:00 closes at 22:00
DECISION = SUNDAY * 24 + DECISION_HOUR  # its hour of the week, counted from Monday 00:00
WEEKEND_BARS = 48  # the move runs from the close of the bar opening Friday 21:00 (22:00)
MAX_HOLD = 120  # the longest trade is flat from the bar opening Friday 21:00, before the next


def utc_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Bar open times in UTC (a timezone-naive index is read as UTC).

    Refuses bars that do not all open on whole UTC hours (30-minute bars, say): the rule looks
    up 1h bars by their opening hour, and those would be the wrong bars.
    """

    utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    if not (utc == utc.floor("h")).all():
        raise ValueError("i009_weekend_reversal needs 1h bars that open on whole UTC hours")
    return utc


def _positive(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and 0 < value < math.inf


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


@dataclass(frozen=True)
class WeekendReversal(Strategy):
    """Take the other side of a large weekend move from Sunday 22:00 UTC.

    * Decision bar: the 1h bar opening Sunday 21:00 UTC, which closes at 22:00.
    * Weekend move ``W = close(Sunday 21:00 bar) / close(Friday 21:00 bar) - 1``, the bar 48
      hours earlier looked up by timestamp; undefined (no trade that week) when either bar is
      missing.
    * Scale ``sigma_48 = sigma_t * sqrt(48)``, where ``sigma_t`` is the standard deviation
      (ddof 1) of the close-to-close returns of the last ``vol_bars`` bars up to and including
      the decision bar (returns between consecutive bars of the data), undefined (no trade)
      until ``vol_bars`` returns exist.
    * At the decision bar's close: -1 if ``W >= min_move * sigma_48``, else +1 if
      ``W <= -min_move * sigma_48``, else 0. The tests are checked in that order, so the one
      case that meets both (no move at all after ``vol_bars`` identical returns) goes short.
    * The target keeps that side on the bars opening Sunday 21:00 .. Sunday 21:00 +
      (hold_bars - 1) hours and is 0 from the bar opening Sunday 21:00 + hold_bars hours,
      counted by the clock (a missing bar uses up its hour). With no decision bar there is no
      trade that week.
    """

    min_move: float = 0.5
    hold_bars: int = 24
    vol_bars: int = 720

    name = "i009_weekend_reversal"

    def __post_init__(self) -> None:
        if not _positive(self.min_move):
            raise ValueError("min_move must be positive and finite")
        if not _integer(self.hold_bars) or not 1 <= self.hold_bars <= MAX_HOLD:
            raise ValueError(f"hold_bars must be an integer from 1 to {MAX_HOLD}")
        if not _integer(self.vol_bars) or self.vol_bars < 2:
            raise ValueError("vol_bars must be an integer >= 2")

    def weekend(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Per bar: ``decision`` (it opens Sunday 21:00 UTC), ``move`` (``W`` on decision bars
        where it is defined, NaN elsewhere), ``scale`` (``sigma_48`` on every bar, NaN until
        ``vol_bars`` returns exist) and ``side`` (the side a decision bar opens, 0 on every
        other bar)."""

        index = bars.index
        clock = utc_clock(index)
        decision = np.asarray((clock.dayofweek == SUNDAY) & (clock.hour == DECISION_HOUR))
        close = bars["close"].astype("float64")
        friday = close.reindex(index - WEEKEND_BARS * HOUR).to_numpy()
        move = np.where(decision, close.to_numpy() / friday - 1.0, np.nan)
        returns = close / close.shift(1) - 1.0
        sigma = returns.rolling(self.vol_bars, min_periods=self.vol_bars).std()
        scale = sigma.to_numpy() * math.sqrt(WEEKEND_BARS)
        threshold = self.min_move * scale
        side = np.select([move >= threshold, move <= -threshold], [-1.0, 1.0], 0.0)
        return pd.DataFrame(
            {"decision": decision, "move": move, "scale": scale, "side": side}, index=index
        )

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        index = bars.index
        side = self.weekend(bars)["side"]
        clock = utc_clock(index)
        # Hours since the latest Sunday 21:00 UTC: 0 on the decision bar, 1 on the 22:00 bar, ...
        week_hour = np.asarray(clock.dayofweek) * 24 + np.asarray(clock.hour)
        since = (week_hour - DECISION) % WEEK
        decided = index - pd.to_timedelta(since, unit="h")
        opened = side.reindex(decided, fill_value=0.0).to_numpy()
        return pd.Series(np.where(since < self.hold_bars, opened, 0.0), index=index)
