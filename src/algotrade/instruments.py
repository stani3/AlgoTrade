"""The instruments research can trade, by asset class, as pre-registered in criteria.yaml.

Crypto is the universe it always was: ``data.exchange`` and ``data.symbols``, with the exchange's
costs (:data:`algotrade.backtest.costs.EXCHANGE_COSTS`) and perpetual funding. Other asset
classes are listed under ``data.universes``, each with its source, its first date, its costs per
side and, optionally, per-symbol overrides::

    data:
      exchange: binanceusdm
      symbols: [BTC, ETH, ...]
      universes:
        bonds:
          source: alpaca
          start: "2016-01-01"
          symbols: [TLT, IEF, ...]
          costs: {fee_bps: 0.5, slippage_bps: 2.0}
          cost_overrides: {MUB: {slippage_bps: 5.0}}
          financing: fed_funds
          max_leverage: 10.0
          leverage_overrides: {TIP: 2.0}

An idea card names the classes it is about (``universe: [fx, commodities]`` or ``all``); a card
without one means ``[crypto]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS, CostModel

ASSET_CLASSES = ("crypto", "stocks", "bonds", "commodities", "fx")
SOURCE_CALENDAR = {
    "binanceusdm": "24/7",
    "bybit": "24/7",
    "alpaca": "us_equity",
    "dukascopy": "fx",
}
CRYPTO = "crypto"


class UniverseError(ValueError):
    """A universe or instrument that criteria.yaml does not define."""


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: str
    source: str
    calendar: str
    costs: CostModel
    start: pd.Timestamp | None = None
    quote: str = "USDT"
    financing: str | None = None  # crypto pays funding; others: fed_funds, rate_difference
    max_leverage: float = 1.0

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == CRYPTO


def universes(criteria) -> dict[str, dict]:
    """Every asset class in criteria, crypto first, each with ``source`` and ``symbols``."""

    found = {CRYPTO: {"source": criteria.get("data.exchange"),
                      "symbols": list(criteria.get("data.symbols"))}}  # fmt: skip
    for name, spec in (criteria.get("data.universes", None) or {}).items():
        if name not in ASSET_CLASSES or name == CRYPTO:
            raise UniverseError(f"unknown asset class '{name}' in data.universes")
        found[name] = spec
    return found


def _instrument(asset_class: str, symbol: str, spec: dict) -> Instrument:
    source = spec["source"]
    if source not in SOURCE_CALENDAR:
        raise UniverseError(f"unknown data source '{source}' for {asset_class}")
    if asset_class == CRYPTO:
        return Instrument(symbol, CRYPTO, source, "24/7", EXCHANGE_COSTS[source])
    base = dict(spec.get("costs") or {})
    base.update((spec.get("cost_overrides") or {}).get(symbol, {}))
    leverage = (spec.get("leverage_overrides") or {}).get(symbol, spec.get("max_leverage", 1.0))
    start = spec.get("start")
    return Instrument(
        symbol=symbol,
        asset_class=asset_class,
        source=source,
        calendar=SOURCE_CALENDAR[source],
        costs=CostModel(fee_bps=float(base.get("fee_bps", 0.0)),
                        slippage_bps=float(base.get("slippage_bps", 0.0))),
        start=pd.Timestamp(start, tz="UTC") if start else None,
        quote="",
        financing=spec.get("financing"),
        max_leverage=float(leverage),
    )  # fmt: skip


def registry(criteria) -> dict[str, Instrument]:
    """Every instrument by symbol; a symbol listed in two classes is an error."""

    out: dict[str, Instrument] = {}
    for asset_class, spec in universes(criteria).items():
        for symbol in spec["symbols"]:
            if symbol in out:
                raise UniverseError(
                    f"{symbol} is listed in both {out[symbol].asset_class} and {asset_class}"
                )
            out[symbol] = _instrument(asset_class, symbol, spec)
    return out


def resolve_universe(value, criteria) -> tuple[str, ...]:
    """A card's ``universe`` as asset classes in canonical order: missing means crypto only,
    ``all`` means every class criteria defines."""

    defined = universes(criteria)
    if value is None:
        return (CRYPTO,)
    if value == "all":
        return tuple(name for name in ASSET_CLASSES if name in defined)
    if isinstance(value, str) or not value:
        raise UniverseError("universe must be 'all' or a non-empty list of asset classes")
    unknown = [name for name in value if name not in defined]
    if unknown:
        raise UniverseError(
            f"unknown asset class {', '.join(map(str, unknown))}; criteria define "
            f"{', '.join(defined)}"
        )
    return tuple(name for name in ASSET_CLASSES if name in set(value))


def universe_key(classes) -> str:
    """One string per universe, for trial ledgers and fingerprints: ``crypto``, ``commodities+fx``."""

    return "+".join(classes)


def symbols_for(criteria, classes) -> list[str]:
    """The symbols of ``classes``, class by class in canonical order (crypto order as listed)."""

    defined = universes(criteria)
    return [s for name in ASSET_CLASSES if name in classes for s in defined[name]["symbols"]]


def costs_for_symbols(criteria, symbols) -> dict[str, CostModel]:
    found = registry(criteria)
    return {symbol: found[symbol].costs for symbol in symbols}


def describe(criteria, symbols, timeframe: str) -> str:
    """Where the bars of ``symbols`` come from, for report headers: ``binanceusdm 4h`` or
    ``bonds (alpaca), fx (dukascopy) 1d``."""

    found = registry(criteria)
    classes = []
    for symbol in symbols:
        instrument = found[symbol]
        label = (
            instrument.source
            if instrument.is_crypto and len(symbols) == sum(found[s].is_crypto for s in symbols)
            else f"{instrument.asset_class} ({instrument.source})"
        )
        if label not in classes:
            classes.append(label)
    return f"{', '.join(classes)} {timeframe}"


def costs_text(costs: Mapping[str, CostModel]) -> str:
    """Report text for per-symbol costs: one line if every symbol pays the same."""

    groups: dict[CostModel, list[str]] = {}
    for symbol, model in costs.items():
        groups.setdefault(model, []).append(symbol)
    if len(groups) == 1:
        model = next(iter(groups))
        return f"fee {model.fee_bps} bps + slippage {model.slippage_bps} bps, funding on"
    return (
        "; ".join(
            f"{model.fee_bps:g}+{model.slippage_bps:g} bps per side: {', '.join(symbols)}"
            for model, symbols in groups.items()
        )
        + " (funding or financing on)"
    )
