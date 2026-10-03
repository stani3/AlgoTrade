"""Sanity checks on downloaded bars, for a person to read before research uses them.

Nothing here changes data. A flagged jump may be real (USO in April 2020) or a corporate action
the source did not adjust (a spin-off, a reverse split); a high share of filled bars means a
thinly traded instrument whose bars are often flat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.calendars import infer_calendar

from .sessions import session_buckets, us_equity_sessions

MAX_MOVE = {"24/7": 0.25, "us_equity": 0.15, "fx": 0.05}  # one bar, close to close
MAX_FILLED = 0.02
PRICES = ["open", "high", "low", "close"]
SHOWN = 5


@dataclass(frozen=True)
class Issue:
    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


def _times(index: pd.Index) -> str:
    shown = ", ".join(f"{t:%Y-%m-%d %H:%M}" for t in index[:SHOWN])
    return shown + (f" and {len(index) - SHOWN} more" if len(index) > SHOWN else "")


def check_bars(bars: pd.DataFrame, timeframe: str, calendar: str | None = None) -> list[Issue]:
    """Problems found in ``bars`` (indexed by UTC open time); an empty list means none."""

    issues: list[Issue] = []
    if bars.empty:
        return [Issue("empty", "no bars")]
    calendar = calendar or infer_calendar(bars.index)
    index = bars.index
    if index.has_duplicates:
        issues.append(Issue("duplicates", _times(index[index.duplicated()])))
    if not index.is_monotonic_increasing:
        issues.append(Issue("order", "bars are not sorted by time"))

    # Row by row (numpy), so duplicated timestamps cannot confuse the comparisons.
    o, h, low, c, volume = (bars[k].to_numpy(dtype=float) for k in [*PRICES, "volume"])
    bad = (h < np.maximum(o, c)) | (low > np.minimum(o, c)) | (low <= 0)
    if bad.any():
        issues.append(Issue("ohlc", f"inconsistent or non-positive prices at {_times(index[bad])}"))

    move = np.abs(c[1:] / c[:-1] - 1.0)
    jumps = np.concatenate(([False], move > MAX_MOVE[calendar]))
    if jumps.any():
        issues.append(
            Issue("jump", f"close-to-close moves above {MAX_MOVE[calendar]:.0%} at "
                          f"{_times(index[jumps])}")
        )  # fmt: skip

    flat = (volume <= 0) & (o == h) & (h == low) & (low == c)
    if calendar != "24/7" and flat.mean() > MAX_FILLED:
        issues.append(Issue("filled", f"{flat.mean():.1%} of bars have no trades (flat)"))

    if calendar == "us_equity":
        issues += _session_issues(bars, timeframe)
    return issues


def _session_issues(bars: pd.DataFrame, timeframe: str) -> list[Issue]:
    first, last = bars.index.min(), bars.index.max()
    sessions = us_equity_sessions(first, last)
    expected = pd.DatetimeIndex(session_buckets(sessions, timeframe)["start"])
    expected = expected[(expected >= first) & (expected <= last)]
    issues = []
    missing = expected.difference(bars.index)
    if len(missing):
        issues.append(Issue("missing", f"{len(missing)} bars missing: {_times(missing)}"))
    extra = bars.index.difference(expected)
    if len(extra):
        issues.append(Issue("outside", f"{len(extra)} bars outside sessions: {_times(extra)}"))
    return issues
