"""Probabilistic and deflated Sharpe ratios (Bailey & Lopez de Prado).

The probabilistic Sharpe ratio is the probability that the true Sharpe ratio exceeds a
benchmark, given the track record's length, skew and fat tails. The deflated Sharpe ratio uses
as benchmark the Sharpe ratio the best of ``N`` worthless strategies would show by luck, so the
more configurations were tried, the higher the bar.

Everything here is per period (not annualised): convert annualised figures by dividing by
sqrt(periods per year), and variances by periods per year.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
NORMAL = NormalDist()


def expected_max_sharpe(trials: int, variance: float) -> float:
    """Expected maximum Sharpe ratio among ``trials`` independent zero-skill strategies whose
    Sharpe ratios vary with ``variance``."""

    if trials < 2 or variance <= 0:
        return 0.0
    z = NORMAL.inv_cdf
    return math.sqrt(variance) * (
        (1 - EULER_GAMMA) * z(1 - 1 / trials) + EULER_GAMMA * z(1 - 1 / (trials * math.e))
    )


def probabilistic_sharpe(
    sharpe: float, benchmark: float, observations: int, skew: float, kurtosis: float
) -> float:
    """P(true Sharpe > benchmark). ``kurtosis`` is the full kurtosis (3 for a normal)."""

    if observations < 2:
        return 0.0
    variance = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if variance <= 0:
        return 0.0
    return NORMAL.cdf((sharpe - benchmark) * math.sqrt(observations - 1) / math.sqrt(variance))


def deflated_sharpe(
    sharpe: float, observations: int, skew: float, kurtosis: float, trials: int, variance: float
) -> float:
    return probabilistic_sharpe(
        sharpe, expected_max_sharpe(trials, variance), observations, skew, kurtosis
    )


def moments(returns: np.ndarray) -> tuple[float, int, float, float]:
    """Per-period Sharpe ratio, number of observations, skew and full kurtosis."""

    returns = np.asarray(returns, dtype=float)
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    # Rounding noise in a constant series is not volatility.
    if not std > 1e-12 * max(1.0, abs(float(returns.mean())) if len(returns) else 1.0):
        return 0.0, len(returns), 0.0, 3.0
    centred = returns - returns.mean()
    m2 = np.mean(centred**2)
    skew = float(np.mean(centred**3) / m2**1.5)
    kurtosis = float(np.mean(centred**4) / m2**2)
    return float(returns.mean() / std), len(returns), skew, kurtosis
