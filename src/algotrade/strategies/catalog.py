"""Strategy registry and a JSON spec format for describing strategies without code.

A spec is a dict with a ``type`` key naming a strategy and its parameters. Wrappers nest:

    {"type": "vol_target", "annual_vol": 0.25,
     "strategy": {"type": "combine",
                  "strategies": [{"type": "ewmac", "fast": 16, "slow": 64},
                                 {"type": "carver_breakout", "lookback": 80}]}}
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import Strategy
from .bracket import BreakoutBracket
from .breakout import BollingerBreakout, DonchianBreakout, KeltnerBreakout
from .carver import EWMAC, CarverBreakout, FundingCarry
from .ma_crossover import MovingAverageCrossover
from .momentum import MomentumStrategy
from .parabolic_sar import ParabolicSARStrategy
from .reversion import BollingerReversion, RSIReversion
from .trend import ADXTrendStrategy, MACDStrategy, SuperTrendStrategy
from .wrappers import Combine, TrendFilter, VolTarget

RULES: tuple[type[Strategy], ...] = (
    MomentumStrategy,
    MovingAverageCrossover,
    ParabolicSARStrategy,
    MACDStrategy,
    SuperTrendStrategy,
    ADXTrendStrategy,
    DonchianBreakout,
    BollingerBreakout,
    KeltnerBreakout,
    RSIReversion,
    BollingerReversion,
    EWMAC,
    CarverBreakout,
    FundingCarry,
)
WRAPPERS: tuple[type[Strategy], ...] = (VolTarget, TrendFilter, Combine)
# Bracket strategies run on the intrabar simulator and cannot be wrapped.
BRACKETS: tuple[type[Strategy], ...] = (BreakoutBracket,)

STRATEGIES: dict[str, type[Strategy]] = {cls.name: cls for cls in RULES + WRAPPERS + BRACKETS}


def from_spec(spec: dict | str | Path) -> Strategy:
    """Build a strategy from a spec dict, a JSON string, or a path to a JSON file."""

    if isinstance(spec, Path) or (isinstance(spec, str) and not spec.lstrip().startswith("{")):
        spec = json.loads(Path(spec).read_text())
    elif isinstance(spec, str):
        spec = json.loads(spec)
    params = dict(spec)
    kind = params.pop("type")
    if kind not in STRATEGIES:
        raise KeyError(f"Unknown strategy type '{kind}'. Known: {', '.join(sorted(STRATEGIES))}")
    if "strategy" in params:
        params["strategy"] = from_spec(params["strategy"])
    if "strategies" in params:
        params["strategies"] = tuple(from_spec(child) for child in params["strategies"])
    if "weights" in params:
        params["weights"] = tuple(params["weights"])
    return STRATEGIES[kind](**params)


def to_spec(strategy: Strategy) -> dict:
    """Inverse of ``from_spec``: a JSON-serializable description of ``strategy``."""

    def encode(value: object) -> object:
        if isinstance(value, Strategy):
            return to_spec(value)
        if isinstance(value, tuple):
            return [encode(item) for item in value]
        return value

    return {"type": strategy.name, **{k: encode(v) for k, v in strategy.params().items()}}
