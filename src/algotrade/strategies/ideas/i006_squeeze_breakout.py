"""i006 Bollinger squeeze breakout: trade the first closes outside the bands after a squeeze.

Bollinger's Method I. When the width of the 20-bar, 2-standard-deviation bands falls to its
lowest of the last ``squeeze_bars`` bars, volatility has contracted and tends to expand next. For
``armed_bars`` bars after such a squeeze, a close above the upper band is a long signal and a
close below the lower band a short one. The bracket simulator enters at the next open with a stop
``stop_atr`` and a target ``target_atr`` ATRs from the fill (ATR read on the signal bar) and
closes a trade still open at the close of its ``max_bars``-th bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import pandas as pd

from algotrade.backtest.bracket import BracketSignals
from algotrade.indicators import atr, bollinger
from algotrade.strategies.bracket import BracketStrategy


def _integer(value: object, low: int) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= low


def _positive(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and 0 < value < math.inf


@dataclass(frozen=True)
class SqueezeBreakout(BracketStrategy):
    """Bollinger squeeze breakout with an ATR bracket and a time exit.

    * Bands ``bollinger(close, length, mult)``; bandwidth ``BW_t = (upper_t - lower_t) / mid_t``.
    * Squeeze bar: ``BW_t <= min(BW_{t - squeeze_bars + 1} .. BW_t)`` (ties count), never while
      that minimum is undefined (the first ``length + squeeze_bars - 2`` bars).
    * Armed: a squeeze bar among bars ``t - armed_bars + 1 .. t``.
    * Long when armed and ``close_t > upper_t``; short when armed and ``close_t < lower_t``
      (never with ``allow_short`` false); no signal while ATR(``atr_length``) is undefined.
    * Stop ``stop_atr`` x ATR_t and target ``target_atr`` x ATR_t from the fill at the next open;
      a trade still open at the close of its ``max_bars``-th bar exits there. Signals during a
      trade are ignored; ``cooldown_win``, ``cooldown_loss`` and ``kill_drawdown`` go to the
      simulator unchanged.
    """

    length: int = 20
    mult: float = 2.0
    squeeze_bars: int = 125
    armed_bars: int = 20
    atr_length: int = 14
    stop_atr: float = 2.0
    target_atr: float = 4.0
    max_bars: int = 48
    cooldown_win: int = 0
    cooldown_loss: int = 0
    kill_drawdown: float = 1.0
    allow_short: bool = True

    name = "i006_squeeze_breakout"

    def __post_init__(self) -> None:
        for field_name, low in (
            ("length", 2),
            ("squeeze_bars", 2),
            ("armed_bars", 1),
            ("atr_length", 1),
            ("max_bars", 1),
            ("cooldown_win", 0),
            ("cooldown_loss", 0),
        ):
            if not _integer(getattr(self, field_name), low):
                raise ValueError(f"{field_name} must be an integer >= {low}")
        for field_name in ("mult", "stop_atr", "target_atr"):
            if not _positive(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be positive and finite")
        drawdown = self.kill_drawdown
        if isinstance(drawdown, bool) or not isinstance(drawdown, Real) or not 0 < drawdown <= 1:
            raise ValueError("kill_drawdown must be in (0, 1]; 1 disables it")
        if not isinstance(self.allow_short, bool):
            raise TypeError("allow_short must be true or false")

    def setup(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Per bar: the bands, ``bandwidth``, its rolling ``lowest``, whether the bar is a
        ``squeeze`` bar, whether the setup is ``armed`` and the signal bar's ``atr``."""

        close = bars["close"]
        bands = bollinger(close, self.length, self.mult)
        bandwidth = (bands["upper"] - bands["lower"]) / bands["mid"]
        lowest = bandwidth.rolling(self.squeeze_bars, min_periods=self.squeeze_bars).min()
        squeeze = bandwidth <= lowest
        recent = squeeze.astype("float64").rolling(self.armed_bars, min_periods=1).max()
        return pd.DataFrame(
            {
                "upper": bands["upper"],
                "lower": bands["lower"],
                "bandwidth": bandwidth,
                "lowest": lowest,
                "squeeze": squeeze,
                "armed": recent > 0,
                "atr": atr(bars["high"], bars["low"], close, self.atr_length),
            },
            index=bars.index,
        )

    def signals(self, bars: pd.DataFrame) -> BracketSignals:
        state = self.setup(bars)
        close = bars["close"]
        ready = state["armed"] & state["atr"].notna()
        long = ready & (close > state["upper"])
        short = ready & (close < state["lower"]) & self.allow_short
        return BracketSignals(
            long=long.rename(None),
            short=short.rename(None),
            stop_dist=self.stop_atr * state["atr"],
            target_dist=self.target_atr * state["atr"],
        )
