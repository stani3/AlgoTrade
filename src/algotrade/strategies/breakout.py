"""Channel breakout systems with separate entry and exit rules (Turtle / Davey style)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algotrade.indicators import bollinger, donchian, keltner

from .base import Strategy, entry_exit


@dataclass(frozen=True)
class DonchianBreakout(Strategy):
    """Enter on a close beyond the prior ``entry``-bar high/low, exit on the ``exit``-bar one.

    Defaults are the Turtle System 1 channels (20 in, 10 out). Channels use prior bars only,
    so the breakout bar itself cannot move the level it has to break.
    """

    entry: int = 20
    exit: int = 10
    allow_short: bool = True

    name = "donchian_breakout"

    def __post_init__(self) -> None:
        if not 0 < self.exit <= self.entry:
            raise ValueError("need 0 < exit <= entry")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        outer = donchian(bars["high"], bars["low"], self.entry).shift()
        inner = donchian(bars["high"], bars["low"], self.exit).shift()
        return entry_exit(
            enter_long=close > outer["upper"],
            exit_long=close < inner["lower"],
            enter_short=close < outer["lower"],
            exit_short=close > inner["upper"],
            allow_short=self.allow_short,
        )


@dataclass(frozen=True)
class BollingerBreakout(Strategy):
    """Enter on a close outside the band, exit when price closes back through the middle."""

    length: int = 20
    mult: float = 2.0
    allow_short: bool = True

    name = "bollinger_breakout"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        bands = bollinger(close, self.length, self.mult)
        return entry_exit(
            enter_long=close > bands["upper"],
            exit_long=close < bands["mid"],
            enter_short=close < bands["lower"],
            exit_short=close > bands["mid"],
            allow_short=self.allow_short,
        )


@dataclass(frozen=True)
class KeltnerBreakout(Strategy):
    """Like the Bollinger breakout but with ATR-based bands around an EMA."""

    length: int = 20
    mult: float = 2.0
    allow_short: bool = True

    name = "keltner_breakout"

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        bands = keltner(bars["high"], bars["low"], close, self.length, self.mult)
        return entry_exit(
            enter_long=close > bands["upper"],
            exit_long=close < bands["mid"],
            enter_short=close < bands["lower"],
            exit_short=close > bands["mid"],
            allow_short=self.allow_short,
        )
