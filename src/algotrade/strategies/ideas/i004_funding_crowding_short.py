"""i004 funding-crowding short: sell the onset of crowded long positioning seen in funding.

Funding is the price of leverage on a perpetual. A window of summed funding that is both high
for the coin's own trailing year (a percentile rank) and above Binance's neutral rate (0.03% a
day) marks longs crowding in and paying a premium. The rule sells short on the first bar of such
crowding and covers after ``hold_bars`` bars, when the funding rank falls back below
``exit_quantile``, or on a close ``stop_atr`` ATRs above the entry close. It never goes long.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from algotrade.indicators import atr
from algotrade.strategies.base import Strategy, bars_per_day

ATR_LENGTH = 14  # Wilder's ATR, fixed by the card (not a parameter)
TOLERANCE = 1e-9  # keeps floating-point sums of exactly-neutral settlements from counting


@dataclass(frozen=True)
class FundingCrowdingShort(Strategy):
    """Short the onset of funding crowding for at most ``hold_bars`` bars; flat otherwise.

    * Windows in bars: ``nW = round(funding_days x bars per day)`` and
      ``nY = round(history_days x bars per day)``.
    * ``F_t``: ``funding_rate`` summed over bars ``t-nW+1 .. t``, undefined for the first
      ``nW - 1`` bars.
    * ``rank_t``: the share of the defined F values in bars ``t-nY+1 .. t`` that are
      ``<= F_t``; undefined until ``nY // 4`` of them exist.
    * Crowded when ``rank_t >= quantile`` and
      ``F_t > neutral_daily_funding x funding_days + 1e-9``; an onset is a crowded bar whose
      previous bar was not crowded (the bar before the first counts as not crowded).
    * Flat at the previous close: go short (-1) at an onset once ATR(14) is defined, with a stop
      at the entry bar's ``close + stop_atr x ATR``. Short since bar s: cover (0) at bar t when
      ``t - s >= hold_bars``, ``rank_t < exit_quantile`` or ``close_t >= stop``. Onsets while
      short are ignored, and a bar that covers never opens a new short.
    """

    funding_days: float = 7
    quantile: float = 0.8
    hold_bars: int = 7
    history_days: float = 365
    exit_quantile: float = 0.5
    neutral_daily_funding: float = 0.0003
    stop_atr: float = 3.0

    name = "i004_funding_crowding_short"

    def __post_init__(self) -> None:
        if not 0 < self.funding_days < math.inf:
            raise ValueError("funding_days must be positive and finite")
        if not self.funding_days < self.history_days < math.inf:
            raise ValueError("history_days must be finite and greater than funding_days")
        if not 0 < self.quantile <= 1:
            raise ValueError("quantile must be in (0, 1]")
        if not 0 < self.exit_quantile < self.quantile:
            raise ValueError("exit_quantile must be in (0, quantile)")
        hold = self.hold_bars
        if isinstance(hold, bool) or not isinstance(hold, Integral) or hold < 1:
            raise ValueError("hold_bars must be an integer >= 1")
        if not 0 < self.stop_atr < math.inf:
            raise ValueError("stop_atr must be positive and finite")
        if not 0 <= self.neutral_daily_funding < math.inf:
            raise ValueError("neutral_daily_funding must be non-negative and finite")

    def crowding(self, bars: pd.DataFrame) -> pd.DataFrame:
        """``funding_sum`` (F), its ``rank``, and the ``crowded`` and ``onset`` flags per bar."""

        per_day = bars_per_day(bars.index)
        n_window = round(self.funding_days * per_day)
        if n_window < 1:
            raise ValueError("funding_days must cover at least one bar")
        n_year = round(self.history_days * per_day)
        funding = bars["funding_rate"].rolling(n_window, min_periods=n_window).sum()
        rank = funding.rolling(n_year, min_periods=n_year // 4).rank(method="max", pct=True)
        floor = self.neutral_daily_funding * self.funding_days + TOLERANCE
        crowded = (rank >= self.quantile) & (funding > floor)
        onset = crowded & ~crowded.shift(fill_value=False)
        return pd.DataFrame(
            {"funding_sum": funding, "rank": rank, "crowded": crowded, "onset": onset}
        )

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        signals = self.crowding(bars)
        onset = signals["onset"].to_numpy(dtype=bool)
        normalised = (signals["rank"] < self.exit_quantile).to_numpy(dtype=bool)
        close = bars["close"].to_numpy(dtype="float64")
        known_atr = atr(bars["high"], bars["low"], bars["close"], ATR_LENGTH).to_numpy()
        target = np.zeros(len(bars))
        short, entry, stop = False, 0, math.nan
        for t in range(len(bars)):
            if short:
                expired = t - entry >= self.hold_bars
                short = not (expired or normalised[t] or close[t] >= stop)
            elif onset[t] and not math.isnan(known_atr[t]):
                short, entry, stop = True, t, close[t] + self.stop_atr * known_atr[t]
            target[t] = -1.0 if short else 0.0
        return pd.Series(target, index=bars.index)
