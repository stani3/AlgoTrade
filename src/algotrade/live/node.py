"""Paper-trading node for a frozen strategy: exchange TESTNETS only.

There is deliberately no way to configure a live (mainnet) environment here: every client is
built with the testnet environment, and :func:`check_testnet` refuses any config that is not.
API keys are never passed in code; the adapters read them from the environment (``.env``), and
only the variable *names* are ever reported.
"""

from __future__ import annotations

import json
import os

from nautilus_trader.adapters.binance import (
    BINANCE,
    BinanceAccountType,
    BinanceDataClientConfig,
    BinanceExecClientConfig,
    BinanceInstrumentProviderConfig,
    BinanceLiveDataClientFactory,
    BinanceLiveExecClientFactory,
)
from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
from nautilus_trader.adapters.bybit import (
    BYBIT,
    BybitDataClientConfig,
    BybitEnvironment,
    BybitExecClientConfig,
    BybitLiveDataClientFactory,
    BybitLiveExecClientFactory,
    BybitProductType,
)
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveExecEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId, TraderId

from .adapter import SpecStrategy, SpecStrategyConfig

VENUES = ("binance", "bybit")
TESTNET_KEYS = {
    "binance": ("BINANCE_FUTURES_TESTNET_API_KEY", "BINANCE_FUTURES_TESTNET_API_SECRET"),
    "bybit": ("BYBIT_TESTNET_API_KEY", "BYBIT_TESTNET_API_SECRET"),
}
BAR_STEPS = {"1h": "1-HOUR", "4h": "4-HOUR", "1d": "1-DAY"}
WARMUP_DAYS = {"1h": 120, "4h": 120, "1d": 500}


class NotTestnet(RuntimeError):
    """A paper-trading config pointed somewhere other than an exchange testnet."""


def instrument_id(venue: str, base: str, quote: str = "USDT") -> InstrumentId:
    suffix = {"binance": "PERP.BINANCE", "bybit": "LINEAR.BYBIT"}[venue]
    return InstrumentId.from_str(f"{base}{quote}-{suffix}")


def bar_type(instrument: InstrumentId, timeframe: str) -> BarType:
    return BarType.from_str(f"{instrument}-{BAR_STEPS[timeframe]}-LAST-EXTERNAL")


def missing_keys(venue: str) -> list[str]:
    """Names of the testnet key variables not set in the environment (never their values)."""

    return [name for name in TESTNET_KEYS[venue] if not os.environ.get(name)]


def node_config(
    venue: str, instruments: list[InstrumentId], trader: str = "PAPER-001"
) -> TradingNodeConfig:
    ids = frozenset(instruments)
    if venue == "binance":
        provider = BinanceInstrumentProviderConfig(load_ids=ids)
        common = {"account_type": BinanceAccountType.USDT_FUTURES,
                  "environment": BinanceEnvironment.TESTNET, "instrument_provider": provider}  # fmt: skip
        data = BinanceDataClientConfig(api_key=None, api_secret=None, **common)
        exec_ = BinanceExecClientConfig(
            api_key=None, api_secret=None, use_reduce_only=True, **common
        )
        name = BINANCE
    else:
        provider = InstrumentProviderConfig(load_ids=ids)
        common = {"product_types": (BybitProductType.LINEAR,),
                  "environment": BybitEnvironment.TESTNET, "instrument_provider": provider}  # fmt: skip
        data = BybitDataClientConfig(api_key=None, api_secret=None, **common)
        exec_ = BybitExecClientConfig(api_key=None, api_secret=None, **common)
        name = BYBIT
    config = TradingNodeConfig(
        trader_id=TraderId(trader),
        logging=LoggingConfig(log_level="INFO", log_colors=False),
        exec_engine=LiveExecEngineConfig(reconciliation=True, reconciliation_lookback_mins=1440),
        data_clients={name: data},
        exec_clients={name: exec_},
    )
    check_testnet(config)
    return config


def check_testnet(config: TradingNodeConfig) -> None:
    clients = [*config.data_clients.values(), *config.exec_clients.values()]
    for client in clients:
        environment = getattr(client, "environment", None)
        if environment not in (BinanceEnvironment.TESTNET, BybitEnvironment.TESTNET):
            raise NotTestnet(f"{type(client).__name__} is not on a testnet ({environment})")


def strategies(
    frozen: dict, venue: str, fills_path: str | None, warmup_days: int | None = None
) -> list[SpecStrategy]:
    """One strategy per symbol of the frozen record, each trading an equal share of equity."""

    symbols = frozen["symbols"]
    timeframe = frozen["timeframe"]
    stake = float(frozen.get("stake") or 1.0)
    built = []
    for base in symbols:
        instrument = instrument_id(venue, base)
        built.append(
            SpecStrategy(
                SpecStrategyConfig(
                    instrument_id=instrument,
                    bar_type=bar_type(instrument, timeframe),
                    spec=json.dumps(frozen["spec"]),
                    stake=stake,
                    capital_share=1.0 / len(symbols),
                    max_exposure=max(1.0, stake),
                    min_change=0.02,
                    warmup_days=WARMUP_DAYS[timeframe] if warmup_days is None else warmup_days,
                    fills_path=fills_path,
                    strategy_id=f"SpecStrategy-{base}",
                )
            )
        )
    return built


def build_node(frozen: dict, venue: str, fills_path: str | None) -> TradingNode:
    """The node, with factories and strategies added. Nothing connects until ``node.run()``."""

    instruments = [instrument_id(venue, base) for base in frozen["symbols"]]
    config = node_config(venue, instruments, trader=f"PAPER-{frozen['idea'].upper()}")
    node = TradingNode(config=config)
    name = BINANCE if venue == "binance" else BYBIT
    factories = {
        "binance": (BinanceLiveDataClientFactory, BinanceLiveExecClientFactory),
        "bybit": (BybitLiveDataClientFactory, BybitLiveExecClientFactory),
    }[venue]
    node.add_data_client_factory(name, factories[0])
    node.add_exec_client_factory(name, factories[1])
    for strategy in strategies(frozen, venue, fills_path):
        node.trader.add_strategy(strategy)
    return node
