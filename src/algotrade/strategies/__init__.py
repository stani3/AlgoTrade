"""Strategy catalogue. See ``catalog.py`` for the registry and JSON spec format."""

from .base import Strategy
from .bracket import BracketStrategy, BreakoutBracket
from .breakout import BollingerBreakout, DonchianBreakout, KeltnerBreakout
from .carver import EWMAC, CarverBreakout, FundingCarry
from .catalog import BRACKETS, RULES, STRATEGIES, WRAPPERS, from_spec, to_spec
from .ma_crossover import MovingAverageCrossover
from .momentum import MomentumStrategy
from .parabolic_sar import ParabolicSARStrategy
from .reversion import BollingerReversion, RSIReversion
from .trend import ADXTrendStrategy, MACDStrategy, SuperTrendStrategy
from .wrappers import Combine, TrendFilter, VolTarget

__all__ = [
    "BRACKETS",
    "EWMAC",
    "RULES",
    "STRATEGIES",
    "WRAPPERS",
    "ADXTrendStrategy",
    "BollingerBreakout",
    "BollingerReversion",
    "BracketStrategy",
    "BreakoutBracket",
    "CarverBreakout",
    "Combine",
    "DonchianBreakout",
    "FundingCarry",
    "KeltnerBreakout",
    "MACDStrategy",
    "MomentumStrategy",
    "MovingAverageCrossover",
    "ParabolicSARStrategy",
    "RSIReversion",
    "Strategy",
    "SuperTrendStrategy",
    "TrendFilter",
    "VolTarget",
    "from_spec",
    "to_spec",
]
