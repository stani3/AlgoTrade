"""The downloader against a fake exchange: symbols, quote currencies, paging and resuming."""

import os

import pandas as pd
import pytest

from algotrade.data import exchange as exchange_module
from algotrade.data.exchange import (
    MarketId,
    fetch_funding,
    fetch_ohlcv,
    funding_path,
    list_perpetuals,
    ohlcv_path,
    update_market,
)
from algotrade.data.market import load_market
from scripts import download_data

HOUR = 3_600_000
START = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)


class FakeExchange:
    """Serves hourly-spaced candles and 8-hourly funding from ``START`` until ``now``."""

    id = "binanceusdm"

    def __init__(self, now: int, listed: int = START, page: int = 1000) -> None:
        self.now, self.listed, self.page = now, listed, page
        self.ohlcv_calls: list[tuple] = []
        self.funding_calls: list[tuple] = []

    def milliseconds(self) -> int:
        return self.now

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.ohlcv_calls.append((symbol, timeframe, since))
        step = {"4h": 4 * HOUR, "1d": 24 * HOUR}[timeframe]
        first = max(since, self.listed)
        first += (-(first - self.listed)) % step  # align to the listing's bar grid
        stamps = range(first, self.now, step)
        return [[t, 100.0, 101.0, 99.0, 100.5, 10.0] for t in list(stamps)[: min(limit, self.page)]]

    def fetch_funding_rate_history(self, symbol, since, limit):
        self.funding_calls.append((symbol, since))
        first = max(since, self.listed)
        first += (-(first - self.listed)) % (8 * HOUR)
        stamps = list(range(first, self.now, 8 * HOUR))[: min(limit, self.page)]
        return [{"timestamp": t, "fundingRate": 0.0001} for t in stamps]

    def load_markets(self):
        def market(base, settle, active=True, swap=True, onboard=None):
            info = {"onboardDate": str(onboard)} if onboard else {}
            return {"base": base, "quote": settle, "settle": settle, "swap": swap,
                    "linear": True, "active": active, "info": info}  # fmt: skip

        return {
            "ETH/USDC:USDC": market("ETH", "USDC", onboard=2),
            "BTC/USDC:USDC": market("BTC", "USDC", onboard=1),
            "OLD/USDC:USDC": market("OLD", "USDC", active=False),
            "BTC/USDC": market("BTC", "USDC", swap=False),
            "BTC/USDT:USDT": market("BTC", "USDT", onboard=1),
        }


def test_market_names_follow_the_quote() -> None:
    usdt = MarketId("binanceusdm", "BTC")
    usdc = MarketId("binanceusdm", "BTC", "USDC")
    assert (usdt.name, usdt.ccxt_symbol) == ("BTCUSDT", "BTC/USDT:USDT")
    assert (usdc.name, usdc.ccxt_symbol) == ("BTCUSDC", "BTC/USDC:USDC")


def test_list_perpetuals_keeps_active_linear_swaps_in_the_quote() -> None:
    fake = FakeExchange(now=START)
    assert list_perpetuals(fake, "USDC") == ["BTC", "ETH"]
    assert list_perpetuals(fake, "USDT") == ["BTC"]


def test_fetch_ohlcv_pages_skips_the_unlisted_past_and_drops_the_forming_bar() -> None:
    now = START + 30 * 4 * HOUR + HOUR  # the 31st bar is still forming
    fake = FakeExchange(now=now, page=7)
    since = START - 2000 * 4 * HOUR  # long before listing: empty pages are skipped
    frame = fetch_ohlcv(fake, MarketId("binanceusdm", "BTC", "USDC"), "4h", since)
    assert len(frame) == 30 and frame["timestamp"].is_monotonic_increasing
    assert frame["timestamp"].iloc[-1] == pd.Timestamp(START + 29 * 4 * HOUR, unit="ms", tz="UTC")
    assert {call[0] for call in fake.ohlcv_calls} == {"BTC/USDC:USDC"}
    assert len(fake.ohlcv_calls) == 5  # pages of 7, 7, 7, 7 and 2 bars


def test_empty_pages_before_listing_are_skipped() -> None:
    class EmptyBeforeListing(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            if since < self.listed - 1000 * 24 * HOUR:
                return []
            return super().fetch_ohlcv(symbol, timeframe, since, limit)

        def fetch_funding_rate_history(self, symbol, since, limit):
            if since < self.listed - 1000 * 8 * HOUR:
                return []
            return super().fetch_funding_rate_history(symbol, since, limit)

    fake = EmptyBeforeListing(now=START + 5 * 24 * HOUR)
    market = MarketId("binanceusdm", "BTC", "USDC")
    long_ago = START - 4000 * 24 * HOUR
    assert len(fetch_ohlcv(fake, market, "1d", long_ago)) == 5
    assert len(fetch_funding(fake, market, long_ago)) == 15


def test_fetch_ohlcv_stops_when_the_exchange_stops_advancing() -> None:
    class Stuck(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            return [[START - 8 * HOUR, 1.0, 1.0, 1.0, 1.0, 1.0]]

    frame = fetch_ohlcv(Stuck(now=START + 100 * HOUR), MarketId("binanceusdm", "BTC"), "4h", START)
    assert len(frame) == 1


def test_fetch_funding_pages_and_stops() -> None:
    now = START + 10 * 8 * HOUR
    fake = FakeExchange(now=now, page=3)
    frame = fetch_funding(fake, MarketId("binanceusdm", "BTC", "USDC"), START - 5000 * 8 * HOUR)
    assert len(frame) == 10 and frame["funding_rate"].eq(0.0001).all()

    class Stuck(FakeExchange):
        def fetch_funding_rate_history(self, symbol, since, limit):
            return [{"timestamp": START - HOUR, "fundingRate": 0.0}]

    assert len(fetch_funding(Stuck(now=now), MarketId("bybit", "BTC"), START)) == 1


def test_update_market_writes_quote_files_and_resumes(tmp_path) -> None:
    market = MarketId("binanceusdm", "BTC", "USDC")
    fake = FakeExchange(now=START + 10 * 24 * HOUR)
    counts = update_market(tmp_path, market, ["4h", "1d"], exchange=fake)
    assert counts == {"4h": 60, "1d": 10, "funding": 30}
    assert ohlcv_path(tmp_path, market, "4h").name == "BTCUSDC_4h.parquet"
    assert funding_path(tmp_path, market).name == "BTCUSDC_funding.parquet"

    fake.now += 2 * 24 * HOUR
    fake.ohlcv_calls.clear()
    counts = update_market(tmp_path, market, ["4h", "1d"], exchange=fake)
    assert counts == {"4h": 72, "1d": 12, "funding": 36}
    first_since = {timeframe: since for _, timeframe, since in fake.ohlcv_calls[:1]}
    assert first_since == {"4h": START + 60 * 4 * HOUR}  # resumed after the last stored bar

    bars = load_market(tmp_path, "binanceusdm", "btc", "4h", quote="usdc")
    assert len(bars) == 72 and bars.attrs["symbol"] == "BTCUSDC"
    # The settlement at the very first bar's open belongs to the bar before it (not stored).
    assert bars["funding_rate"].sum() == pytest.approx(35 * 0.0001)
    with pytest.raises(FileNotFoundError, match="--quote USDC --symbols ETH"):
        load_market(tmp_path, "binanceusdm", "ETH", "4h", quote="USDC")


def test_empty_files_restart_from_the_beginning(tmp_path) -> None:
    market = MarketId("binanceusdm", "BTC", "USDC")
    path = ohlcv_path(tmp_path, market, "1d")
    path.parent.mkdir(parents=True)
    pd.DataFrame(columns=exchange_module.OHLCV_COLUMNS).to_parquet(path, index=False)
    fake = FakeExchange(now=START + 3 * 24 * HOUR)
    assert update_market(tmp_path, market, ["1d"], exchange=fake)["1d"] == 3


def test_download_all_usdc_perpetuals(tmp_path, monkeypatch, capsys) -> None:
    fake = FakeExchange(now=START + 2 * 24 * HOUR)
    monkeypatch.setattr(download_data, "make_exchange", lambda exchange_id: fake)
    settings = type("S", (), {"data_paths": type("P", (), {"raw": tmp_path})()})()
    monkeypatch.setattr(download_data, "load_settings", lambda: settings)

    def flaky(root, market, timeframes, exchange):
        if market.base == "ETH":
            raise download_data.ccxt.NetworkError("timeout")
        return {"1d": 2, "funding": 6}

    monkeypatch.setattr(download_data, "update_market", flaky)
    download_data.main("binanceusdm", ["ALL"], ["1d"], quote="USDC")
    out = capsys.readouterr().out
    assert "2 active USDC perpetuals on binanceusdm" in out
    assert "BTCUSDC: 1d=2, funding=6" in out and "ETHUSDC: FAILED (NetworkError: timeout)" in out


def test_make_exchange() -> None:
    assert exchange_module.make_exchange("bybit").options["defaultType"] == "swap"
    assert exchange_module.make_exchange("binanceusdm").id == "binanceusdm"
    with pytest.raises(ValueError, match="Unsupported exchange 'kraken'"):
        exchange_module.make_exchange("kraken")


def test_download_named_symbols_from_the_command_line(tmp_path, monkeypatch, capsys) -> None:
    import runpy
    import sys

    import algotrade.config

    calls = []
    settings = type("S", (), {"data_paths": type("P", (), {"raw": tmp_path})()})()
    monkeypatch.setattr(algotrade.config, "load_settings", lambda: settings)
    monkeypatch.setattr(exchange_module, "make_exchange", lambda exchange_id: object())
    monkeypatch.setattr(
        exchange_module,
        "update_market",
        lambda root, market, timeframes, exchange: calls.append((market, timeframes)) or {"4h": 1},
    )
    argv = ["download_data", "--quote", "USDC", "--symbols", "btc, sol", "--timeframes", "4h"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(download_data.__file__, run_name="__main__")
    assert [(m.name, t) for m, t in calls] == [("BTCUSDC", ["4h"]), ("SOLUSDC", ["4h"])]
    assert "SOLUSDC: 4h=1" in capsys.readouterr().out


# --- stocks, ETFs and forex ---------------------------------------------------------------


def test_asset_classes_choose_source_and_symbols() -> None:
    args = download_data.parse(["--asset-class", "bonds"])
    assert args.source == "alpaca" and args.symbols.startswith("TLT,IEF")
    args = download_data.parse(["--asset-class", "fx", "--symbols", "eurusd"])
    assert (args.source, args.symbols) == ("dukascopy", "eurusd")
    assert download_data.parse([]).source == "binanceusdm"
    assert download_data.parse(["--exchange", "bybit"]).source == "bybit"
    with pytest.raises(SystemExit):
        download_data.parse(["--asset-class", "fx", "--source", "alpaca"])
    with pytest.raises(SystemExit):
        download_data.parse(["--source", "alpaca"])
    assert download_data.split(" aapl, brk.b ,") == ["AAPL", "BRK.B"]


def test_session_markets_download_check_and_report_failures(tmp_path, monkeypatch, capsys) -> None:
    from algotrade.data import alpaca, dukascopy, rates
    from algotrade.data.files import bars_path, write_bars

    settings = type("S", (), {"data_paths": type("P", (), {"raw": tmp_path})()})()
    monkeypatch.setattr(download_data, "load_settings", lambda: settings)
    monkeypatch.setattr(alpaca, "AlpacaClient", lambda: "client")
    monkeypatch.setattr(dukascopy, "DukascopyClient", lambda: "client")

    class FakeFinancing:
        def update(self, root, source, symbol):
            if symbol == "IEF":
                raise rates.RatesError("HTTP 500 for FRED series DFF")
            return 5

    monkeypatch.setattr(download_data, "Financing", FakeFinancing)
    index = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                        index=index)  # fmt: skip
    bars.iloc[20:, :4] = 2.0  # a 100% jump for --check to report

    def update(root, symbol, timeframes, client):
        if symbol == "BAD":
            raise alpaca.AlpacaError("HTTP 404 for BAD")
        write_bars(bars, bars_path(root, "alpaca", symbol, "4h"))
        return {"4h": len(bars)} if symbol == "TLT" else {}

    monkeypatch.setattr(alpaca, "update_equity", update)
    failed = download_data.main_sessions("alpaca", ["TLT", "BAD", "IEF"], ["4h", "1d"], True)
    out = capsys.readouterr().out
    assert failed == 1
    assert "TLT: 4h=40" in out and "IEF: up to date" in out
    assert "financing: 5 charges" in out
    assert "financing: FAILED (RatesError: HTTP 500 for FRED series DFF)" in out
    assert "BAD: FAILED (AlpacaError: HTTP 404 for BAD)" in out
    assert "check 4h jump: close-to-close moves above 25%" in out

    calls = []
    monkeypatch.setattr(dukascopy, "update_fx", lambda *args: calls.append(args[1]) or {"1h": 1})
    assert download_data.main_sessions("dukascopy", ["EURUSD"], ["1h"], False) == 0
    assert calls == ["EURUSD"]


def test_session_markets_from_the_command_line(tmp_path, monkeypatch) -> None:
    env = tmp_path / "keys.env"
    env.write_text("ALGOTRADE_TEST_KEY=from-file\n# comment\n", encoding="utf-8")
    seen = []
    monkeypatch.delenv("ALGOTRADE_TEST_KEY", raising=False)
    monkeypatch.setattr(
        download_data, "main_sessions",
        lambda source, symbols, timeframes, check: seen.append(
            (source, symbols, timeframes, check, os.environ.get("ALGOTRADE_TEST_KEY"))
        ) or 2,
    )  # fmt: skip
    assert download_data.run(["--asset-class", "fx", "--check", "--env-file", str(env)]) == 2
    assert seen == [("dukascopy", ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"], ["1h", "4h", "1d"],
                     True, "from-file")]  # fmt: skip
    crypto = []
    monkeypatch.setattr(download_data, "main", lambda **kwargs: crypto.append(kwargs))
    assert download_data.run(["--symbols", "btc", "--timeframes", "1d"]) == 0
    assert crypto == [{"exchange_id": "binanceusdm", "symbols": ["BTC"], "timeframes": ["1d"],
                       "quote": "USDT"}]  # fmt: skip


def test_financing_fetches_each_rate_once(tmp_path, monkeypatch) -> None:
    from algotrade.data import rates

    fetched = []
    monkeypatch.setattr(
        rates, "fetch_series", lambda series, session: fetched.append(series) or 0.0
    )
    monkeypatch.setattr(rates, "update_us_financing", lambda root, symbol, rate: 3)
    monkeypatch.setattr(rates, "update_fx_rollovers", lambda root, pair, found: len(found))
    financing = download_data.Financing()
    assert financing.update(tmp_path, "alpaca", "TLT") == 3
    assert financing.update(tmp_path, "alpaca", "IEF") == 3
    assert financing.update(tmp_path, "dukascopy", "EURUSD") == 2
    assert financing.update(tmp_path, "dukascopy", "USDJPY") == 2
    assert fetched == ["DFF", "ECBDFR", "IRSTCI01JPM156N"]
