"""Strategies that trade with a fixed stop and profit target (bracket orders).

They need intrabar fills and depend on previous trade outcomes, so they run on the bar-by-bar
simulator in ``algotrade.backtest.bracket`` instead of returning a target-exposure series.
Wrappers such as ``VolTarget`` and ``Combine`` therefore cannot wrap them.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import pandas as pd

from algotrade.backtest.bracket import BracketSignals, simulate_bracket
from algotrade.backtest.costs import CostModel
from algotrade.backtest.engine import BacktestResult
from algotrade.indicators import atr, rsi

from .base import Strategy


@dataclass(frozen=True)
class BracketStrategy(Strategy):
    """Base class. Subclasses define ``signals`` plus the trade-management fields below."""

    name = "bracket"

    @abstractmethod
    def signals(self, bars: pd.DataFrame) -> BracketSignals: ...

    def simulate(
        self, bars: pd.DataFrame, costs: CostModel | None = None, leverage: float = 1.0
    ) -> BacktestResult:
        return simulate_bracket(
            bars,
            self.signals(bars),
            costs=costs,
            leverage=leverage,
            cooldown_win=self.cooldown_win,
            cooldown_loss=self.cooldown_loss,
            kill_drawdown=self.kill_drawdown,
            max_bars=getattr(self, "max_bars", 0),
        )

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        raise TypeError(
            f"{self.name} uses intrabar stops and targets; run it with "
            "algotrade.backtest.runner.backtest (or .simulate), and don't wrap it in "
            "vol_target / combine / trend_filter"
        )


@dataclass(frozen=True)
class BreakoutBracket(BracketStrategy):
    """Close breakout with an RSI filter, ATR stop and target, cooldowns and a kill switch.

    * Long when the close is the highest close of the last ``lookback`` bars and
      RSI(``rsi_length``) > ``rsi_level``; short on the lowest close with RSI below the level.
    * Entry at the next bar's open. Stop ``stop_atr`` x ATR and target ``target_atr`` x ATR from
      the fill, with ATR(``atr_length``) read on the signal bar.
    * After a winning trade wait ``cooldown_win`` bars, after a loser ``cooldown_loss`` bars.
    * Stop trading for good once equity is ``kill_drawdown`` below its peak.
    """

    lookback: int = 48
    rsi_length: int = 30
    rsi_level: float = 50.0
    atr_length: int = 14
    stop_atr: float = 2.0
    target_atr: float = 4.0
    cooldown_win: int = 20
    cooldown_loss: int = 5
    kill_drawdown: float = 0.5
    allow_short: bool = True

    name = "breakout_bracket"

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.rsi_length < 1 or self.atr_length < 1:
            raise ValueError("rsi_length and atr_length must be positive")
        if not 0 < self.rsi_level < 100:
            raise ValueError("rsi_level must be between 0 and 100")
        if self.stop_atr <= 0 or self.target_atr <= 0:
            raise ValueError("stop_atr and target_atr must be positive")
        if self.cooldown_win < 0 or self.cooldown_loss < 0:
            raise ValueError("cooldowns cannot be negative")
        if not 0 < self.kill_drawdown <= 1:
            raise ValueError("kill_drawdown must be in (0, 1]; 1 disables it")

    def signals(self, bars: pd.DataFrame) -> BracketSignals:
        close = bars["close"]
        strength = rsi(close, self.rsi_length)
        highest = close.rolling(self.lookback, min_periods=self.lookback).max()
        lowest = close.rolling(self.lookback, min_periods=self.lookback).min()
        long = (close >= highest) & (strength > self.rsi_level)
        short = (close <= lowest) & (strength < self.rsi_level)
        if not self.allow_short:
            short = pd.Series(False, index=bars.index)
        volatility = atr(bars["high"], bars["low"], close, self.atr_length)
        return BracketSignals(
            long=long,
            short=short,
            stop_dist=self.stop_atr * volatility,
            target_dist=self.target_atr * volatility,
        )
