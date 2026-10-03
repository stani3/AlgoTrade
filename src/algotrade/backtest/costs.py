"""Transaction-cost assumptions: per exchange for crypto perpetuals, per instrument elsewhere."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Costs charged on every unit of turnover (fraction of equity traded).

    Defaults assume taker execution at the base VIP tier plus a slippage allowance that is
    reasonable for BTC/ETH-sized books on 4h+ bars; raise ``slippage_bps`` for thinner alts.
    """

    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    include_funding: bool = True

    @property
    def rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000


EXCHANGE_COSTS = {
    "binanceusdm": CostModel(fee_bps=5.0, slippage_bps=3.0),
    "bybit": CostModel(fee_bps=5.5, slippage_bps=3.0),
}

ZERO_COSTS = CostModel(fee_bps=0.0, slippage_bps=0.0, include_funding=False)


def costs_for(costs: CostModel | Mapping[str, CostModel], symbol: str) -> CostModel:
    """``symbol``'s cost model: either one model for every symbol or a model per symbol."""

    return costs if isinstance(costs, CostModel) else costs[symbol]


def cap_for(caps: float | Mapping[str, float], symbol: str) -> float:
    """``symbol``'s leverage cap: one cap for every symbol or a cap per symbol (default 1)."""

    return float(caps) if isinstance(caps, int | float) else float(caps.get(symbol, 1.0))
