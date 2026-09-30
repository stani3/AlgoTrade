"""Strategy interface: turn one instrument's bars into a target-exposure series."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from algotrade.indicators import entry_exit_state

YEAR = pd.Timedelta(days=365.25)


@dataclass(frozen=True)
class Strategy(ABC):
    """Base class for single-instrument strategies.

    ``target_position`` returns the signed exposure (fraction of equity) wanted at each bar's
    close, using only data available at that close. The engine handles execution timing.
    Discrete rules return -1/0/+1; forecast rules return values in [-1, 1].
    """

    name = "base"

    @abstractmethod
    def target_position(self, bars: pd.DataFrame) -> pd.Series: ...

    def params(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __str__(self) -> str:
        def show(value: object) -> str:
            if isinstance(value, tuple):
                return "[" + ", ".join(show(v) for v in value) + "]"
            return str(value)

        args = ", ".join(f"{key}={show(value)}" for key, value in self.params().items())
        return f"{self.name}({args})"


def long_short(
    condition_long: pd.Series, condition_short: pd.Series, allow_short: bool
) -> pd.Series:
    """+1 where long, -1 (or 0 if shorts are off) where short, 0 when undefined."""

    target = pd.Series(0.0, index=condition_long.index)
    target[condition_long.fillna(False).astype(bool)] = 1.0
    if allow_short:
        target[condition_short.fillna(False).astype(bool)] = -1.0
    return target


def entry_exit(
    enter_long: pd.Series,
    exit_long: pd.Series,
    enter_short: pd.Series,
    exit_short: pd.Series,
    allow_short: bool,
) -> pd.Series:
    """Stateful position from separate entry and exit rules (see ``entry_exit_state``).

    With shorts disabled, a short entry still closes an open long.
    """

    def arr(condition: pd.Series) -> np.ndarray:
        return condition.fillna(False).to_numpy(dtype=bool)

    el, xl, es, xs = arr(enter_long), arr(exit_long), arr(enter_short), arr(exit_short)
    if not allow_short:
        xl = xl | es
        es = np.zeros_like(es)
    return pd.Series(entry_exit_state(el, xl, es, xs), index=enter_long.index)


def bars_per_year(index: pd.DatetimeIndex) -> float:
    return YEAR / (index[1] - index[0]) if len(index) > 1 else 365.25


def bars_per_day(index: pd.DatetimeIndex) -> float:
    return bars_per_year(index) / 365.25


def scale_forecast(raw: pd.Series, min_periods: int, target_abs: float = 10.0) -> pd.Series:
    """Carver-style forecast scaling, done causally.

    The scalar makes the average absolute forecast ``target_abs`` using an expanding window, so
    it never uses future data. The result is capped at +-20 and mapped to exposure in [-1, 1].
    """

    avg_abs = raw.abs().expanding(min_periods=min_periods).mean()
    forecast = (raw * target_abs / avg_abs).clip(-20, 20)
    return (forecast / 20).fillna(0.0)
