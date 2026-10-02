"""What a period of a given length should look like, from the out-of-sample bar returns.

Davey's Monte Carlo resamples closed trades, which is right for sizing but blind to drawdowns
inside open trades. To judge a live-looking equity curve (the holdout, the incubation forward
test) bar by bar, compare it with paths of the same length built by a block bootstrap of the
walk-forward's out-of-sample bar returns: consecutive blocks keep volatility clustering and
the autocorrelation trends create.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bands:
    final_returns: np.ndarray
    max_drawdowns: np.ndarray

    def drawdown(self, quantile: float) -> float:
        return float(np.quantile(self.max_drawdowns, quantile))

    def total_return(self, quantile: float) -> float:
        return float(np.quantile(self.final_returns, quantile))


def block_bootstrap(
    series: list[np.ndarray], length: int, runs: int, block: int, seed: int = 0
) -> Bands:
    """``runs`` paths of ``length`` bars: each picks one series at random and strings together
    random blocks of ``block`` consecutive bars from it."""

    rng = np.random.default_rng(seed)
    usable = [np.asarray(s, dtype=float) for s in series if len(s) >= block]
    if not usable or length < 1 or runs < 1:
        raise ValueError("need at least one series longer than the block, and length/runs >= 1")
    finals, drawdowns = np.zeros(runs), np.zeros(runs)
    for run in range(runs):
        source = usable[rng.integers(len(usable))]
        starts = rng.integers(0, len(source) - block + 1, size=-(-length // block))
        path = np.concatenate([source[s : s + block] for s in starts])[:length]
        equity = np.cumprod(1.0 + np.maximum(path, -1.0))
        peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
        finals[run] = equity[-1] - 1.0
        drawdowns[run] = float(np.max(1.0 - equity / peak))
    return Bands(finals, drawdowns)
