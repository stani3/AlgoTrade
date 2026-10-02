"""Paper-trading set-up: testnet only, keys only by name, warm-up, and the CLI's refusals."""

import asyncio
import json

import pandas as pd
import pytest

pytest.importorskip("nautilus_trader")

from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
from research_helpers import frozen_workspace

from algotrade.live import node as live_node
from algotrade.live.adapter import SpecStrategy, SpecStrategyConfig
from algotrade.live.parity import bar_type, instrument, rounded, to_bars
from algotrade.research.incubation import start
from algotrade.research.registry import find_version
from scripts import paper_trade

FROZEN = {"idea": "i007", "spec": {"type": "ewmac"}, "timeframe": "4h", "stake": 1.5,
          "symbols": ["BTC", "ETH", "SOL"]}  # fmt: skip


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """Tests never see the user's environment keys, and get their own event loop."""

    for name in [n for names in live_node.TESTNET_KEYS.values() for n in names]:
        monkeypatch.delenv(name, raising=False)
    asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.mark.parametrize("venue", live_node.VENUES)
def test_configs_are_testnet_only(venue) -> None:
    ids = [live_node.instrument_id(venue, "BTC")]
    config = live_node.node_config(venue, ids)
    clients = [*config.data_clients.values(), *config.exec_clients.values()]
    assert len(clients) == 2 and all(c.environment.name.upper() == "TESTNET" for c in clients)
    assert live_node.missing_keys(venue) == list(live_node.TESTNET_KEYS[venue])


def test_a_mainnet_config_is_refused() -> None:
    config = live_node.node_config("binance", [live_node.instrument_id("binance", "BTC")])
    live = type(config.data_clients["BINANCE"])(environment=BinanceEnvironment.LIVE)
    tampered = type(config)(data_clients={"BINANCE": live}, exec_clients=config.exec_clients)
    with pytest.raises(live_node.NotTestnet, match="not on a testnet"):
        live_node.check_testnet(tampered)


def test_instruments_bar_types_and_keys(monkeypatch) -> None:
    assert str(live_node.instrument_id("binance", "SOL")) == "SOLUSDT-PERP.BINANCE"
    assert str(live_node.instrument_id("bybit", "BTC", "USDC")) == "BTCUSDC-LINEAR.BYBIT"
    kind = live_node.bar_type(live_node.instrument_id("bybit", "ETH"), "1d")
    assert str(kind) == "ETHUSDT-LINEAR.BYBIT-1-DAY-LAST-EXTERNAL"
    hourly = live_node.bar_type(live_node.instrument_id("binance", "BTC"), "1h")
    assert str(hourly) == "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL"
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", "placeholder")
    assert live_node.missing_keys("bybit") == ["BYBIT_TESTNET_API_SECRET"]


def test_one_strategy_per_symbol_with_an_equal_share() -> None:
    built = live_node.strategies(FROZEN, "binance", "fills.csv")
    assert [s.config.instrument_id.symbol.value for s in built] == [
        "BTCUSDT-PERP",
        "ETHUSDT-PERP",
        "SOLUSDT-PERP",
    ]
    first = built[0].config
    assert first.capital_share == pytest.approx(1 / 3) and first.stake == 1.5
    assert first.max_exposure == 1.5 and first.warmup_days == 120 and first.min_change == 0.02
    assert json.loads(first.spec) == {"type": "ewmac"} and first.fills_path == "fills.csv"
    daily = live_node.strategies({**FROZEN, "timeframe": "1d", "stake": None}, "bybit", None, 30)
    assert daily[0].config.warmup_days == 30 and daily[0].config.stake == 1.0


def test_node_is_assembled_without_connecting() -> None:
    node = live_node.build_node(FROZEN, "bybit", None)
    try:
        assert not node.is_built() and len(node.trader.strategies()) == 3
    finally:
        node.dispose()


# --- the adapter's bar handling -------------------------------------------------------------


def test_bar_close_stamps_become_open_times_and_warm_up_is_deduplicated() -> None:
    frame = pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "high": [2.0, 3.0, 4.0], "low": [1.0, 2.0, 3.0],
         "close": [2.0, 3.0, 4.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC"),
    )  # fmt: skip
    perp = instrument()
    kind = bar_type(perp.id, "4h")
    bars = to_bars(rounded(frame), kind, perp)
    strategy = SpecStrategy(
        SpecStrategyConfig(instrument_id=perp.id, bar_type=kind, spec='{"type": "ewmac"}')
    )
    assert strategy.open_time(bars[1]) == frame.index[1]
    for bar in bars[:2]:
        strategy.on_historical_data(bar)
    strategy.on_historical_data("not a bar")
    assert list(strategy.decider.history.index) == list(frame.index[:2])
    assert strategy._add(bars[1]) is False and strategy._add(bars[2]) is True
    strategy.on_stop()


# --- the command line --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    return frozen_workspace(tmp_path_factory.mktemp("paper") / "repo")


def cli(ws, *args) -> int:
    return paper_trade.main(["i001", "--root", str(ws.root), *args])


def test_refuses_until_incubation_started_and_keys_exist(frozen, capsys, monkeypatch) -> None:
    ws, _ = frozen
    assert cli(ws, "--dry-run") == 1
    assert "incubation has not been started" in capsys.readouterr().out
    start(ws, find_version(ws, "i001"), when="2025-12-01", commit=False)
    assert cli(ws, "--dry-run") == 0
    out = capsys.readouterr().out
    assert (
        "binance TESTNET" in out and "BINANCE_FUTURES_TESTNET_API_KEY" in out and "Dry run" in out
    )
    assert cli(ws) == 1  # no keys: nothing starts
    started = []
    monkeypatch.setattr(
        paper_trade, "run_node", lambda node: started.append(node) or node.dispose()
    )
    (ws.root / ".env").write_text(
        "# testnet only\nBYBIT_TESTNET_API_KEY=placeholder\nBYBIT_TESTNET_API_SECRET='placeholder'\n",
        encoding="utf-8",
    )
    assert cli(ws, "--venue", "bybit") == 0 and len(started) == 1
    assert "placeholder" not in capsys.readouterr().out  # values are never printed


def test_unfrozen_ideas_are_refused(tmp_path, capsys) -> None:
    from research_helpers import draft, make_workspace

    from algotrade.research.registry import register

    ws, criteria = make_workspace(tmp_path, market=False)
    register(ws, criteria, draft(), commit=False)
    assert cli(ws, "--dry-run") == 1
    assert "REFUSED: i001 v1 is not frozen" in capsys.readouterr().out


def test_parity_on_the_latest_bars(frozen, capsys) -> None:
    ws, _ = frozen
    assert cli(ws, "--parity") in (0, 1)
    out = capsys.readouterr().out
    assert out.count("final equity ours") == 3 and "PARITY" in out


def test_parity_verdict_and_dotenv(frozen, monkeypatch, capsys, tmp_path) -> None:
    ws, _ = frozen
    frozen_record = {"spec": {"type": "ewmac"}, "timeframe": "4h", "symbols": ["BTC"]}
    assert paper_trade.parity(ws, frozen_record, bars_per_symbol=600) is True
    assert "PARITY OK" in capsys.readouterr().out
    paper_trade.load_dotenv(tmp_path / "missing.env")


def test_run_node_builds_runs_and_always_disposes() -> None:
    calls = []

    class FakeNode:
        def build(self):
            calls.append("build")

        def run(self):
            calls.append("run")
            raise KeyboardInterrupt

        def dispose(self):
            calls.append("dispose")

    with pytest.raises(KeyboardInterrupt):
        paper_trade.run_node(FakeNode())
    assert calls == ["build", "run", "dispose"]


def test_warm_up_request_and_duplicate_bars_inside_a_backtest() -> None:
    from decimal import Decimal

    import numpy as np
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.models import FillModel
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId
    from nautilus_trader.model.objects import Money

    index = pd.date_range("2024-01-01", periods=50, freq="4h", tz="UTC")
    close = 100 + np.arange(50.0)
    frame = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close}, index=index
    )
    perp = instrument()
    kind = bar_type(perp.id, "4h")
    bars = to_bars(rounded(frame), kind, perp)
    engine = BacktestEngine(
        config=BacktestEngineConfig(trader_id=TraderId("WARMUP-001"),
                                    logging=LoggingConfig(log_level="ERROR"))
    )  # fmt: skip
    engine.add_venue(venue=perp.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                     starting_balances=[Money(1e6, USDT)], base_currency=USDT,
                     default_leverage=Decimal(10), fill_model=FillModel(), bar_execution=True)  # fmt: skip
    engine.add_instrument(perp)
    engine.add_data(bars)
    strategy = SpecStrategy(SpecStrategyConfig(instrument_id=perp.id, bar_type=kind,
                                               spec='{"type": "ewmac"}', warmup_days=2))  # fmt: skip
    engine.add_strategy(strategy)
    engine.run()
    try:
        assert len(strategy.decider.history) == 50
        strategy.on_bar(bars[10])  # an old bar again: ignored before touching the account
        assert len(strategy.decider.history) == 50
    finally:
        engine.dispose()


def test_parity_run_without_trades() -> None:
    from algotrade.live.parity import ParityRun

    run = ParityRun(ours=None, equity=pd.Series(dtype=float), positions=pd.DataFrame())
    assert run.trades().empty and list(run.trades().columns)[:2] == ["entry", "exit"]


def test_cli_entry_point(tmp_path, monkeypatch) -> None:
    import runpy
    import sys

    from research_helpers import make_workspace

    ws, _ = make_workspace(tmp_path, market=False)
    monkeypatch.setattr(sys, "argv", ["paper_trade", "i404", "--root", str(ws.root)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(paper_trade.__file__, run_name="__main__")
    assert exit_info.value.code == 1
