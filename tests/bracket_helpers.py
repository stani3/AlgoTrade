"""Shared builders for the bracket-simulator tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from algotrade.backtest.bracket import BracketSignals, simulate_bracket
from algotrade.backtest.costs import CostModel

ZERO = CostModel(fee_bps=0.0, slippage_bps=0.0, include_funding=False)
BASE = 128.0  # a power of two keeps the kill-switch arithmetic exact


def bars_from(
    rows: Sequence[tuple[float, float, float, float]],
    funding: float | Sequence[float] | None = None,
    freq: str = "4h",
) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(rows), freq=freq, tz="UTC")
    frame = pd.DataFrame(
        list(rows), columns=["open", "high", "low", "close"], index=index, dtype="float64"
    )
    frame["funding_rate"] = 0.0 if funding is None else funding
    return frame


def directional(
    d: int,
    rows: Sequence[tuple[float, float, float, float]],
    base: float = BASE,
    **kwargs,
) -> pd.DataFrame:
    """Bars described relative to a trade in direction ``d`` (+1 long, -1 short).

    Each row is (open, favourable, adverse, close) as offsets from ``base``: ``favourable`` is
    how far price moved in the trade's favour within the bar, ``adverse`` how far against it.
    The same rows therefore describe mirror-image markets for a long and a short.
    """
    out = []
    for o, fav, adv, c in rows:
        prices = [base + d * o, base + d * fav, base - d * adv, base + d * c]
        out.append((prices[0], max(prices), min(prices), prices[3]))
    return bars_from(out, **kwargs)


def flat(n: int) -> list[tuple[float, float, float, float]]:
    return [(0.0, 0.0, 0.0, 0.0)] * n


def make_signals(
    bars: pd.DataFrame,
    long: Iterable[int] = (),
    short: Iterable[int] = (),
    stop: float | Sequence[float] = 5.0,
    target: float | Sequence[float] = 10.0,
) -> BracketSignals:
    def flags(positions: Iterable[int]) -> pd.Series:
        series = pd.Series(False, index=bars.index)
        series.iloc[list(positions)] = True
        return series

    def dist(value: float | Sequence[float]) -> pd.Series:
        values = np.full(len(bars), value) if np.ndim(value) == 0 else value
        return pd.Series(values, index=bars.index, dtype="float64")

    return BracketSignals(flags(long), flags(short), dist(stop), dist(target))


def run(
    bars: pd.DataFrame,
    long: Iterable[int] = (),
    short: Iterable[int] = (),
    stop: float | Sequence[float] = 5.0,
    target: float | Sequence[float] = 10.0,
    costs: CostModel = ZERO,
    **kwargs,
):
    return simulate_bracket(bars, make_signals(bars, long, short, stop, target), costs, **kwargs)


def entry(d: int, *bars_at: int) -> dict:
    """Signal keyword for a trade in direction ``d`` at the given bar positions."""
    return {"long": bars_at} if d > 0 else {"short": bars_at}


def bar_numbers(bars: pd.DataFrame, stamps: pd.Series) -> np.ndarray:
    return bars.index.get_indexer(pd.DatetimeIndex(stamps))


def random_bars(seed: int, n: int = 400, freq: str = "4h") -> pd.DataFrame:
    """Random walk with real gaps between bars, wide ranges and some funding."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    gap = np.exp(rng.normal(0, 0.01, n))
    open_ = np.concatenate(([close[0]], close[:-1])) * gap
    wick_up = np.abs(rng.normal(0, 0.012, n))
    wick_down = np.abs(rng.normal(0, 0.012, n))
    high = np.maximum(open_, close) * (1 + wick_up)
    low = np.minimum(open_, close) * (1 - wick_down)
    funding = rng.normal(5e-5, 2e-4, n)
    return bars_from(list(zip(open_, high, low, close, strict=True)), funding=funding, freq=freq)
