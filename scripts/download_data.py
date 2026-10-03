"""Download or incrementally update market history into data/raw/<source>/.

Crypto perpetuals come from the exchange (bars and funding). US stocks and ETFs come from
Alpaca (free account; ALPACA_API_KEY and ALPACA_SECRET_KEY in .env) and forex from Dukascopy
(no account), on session-anchored 1h, 4h and daily bars, with the financing of a position
(fed funds for US instruments, the rate difference for forex) from FRED.

Examples:
    python -m scripts.download_data                                   # the 10 USDT perps
    python -m scripts.download_data --quote USDC --symbols ALL        # every USDC perp
    python -m scripts.download_data --symbols BTC,ETH --timeframes 1d
    python -m scripts.download_data --asset-class bonds --check       # 10 bond ETFs
    python -m scripts.download_data --asset-class fx --check          # 4 forex pairs
    python -m scripts.download_data --source alpaca --symbols SPY,QQQ
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ccxt

from algotrade.config import load_dotenv, load_settings
from algotrade.data.exchange import (
    QUOTES,
    SUPPORTED_EXCHANGES,
    TIMEFRAMES,
    MarketId,
    list_perpetuals,
    make_exchange,
    update_market,
)

DEFAULT_UNIVERSE = "BTC,ETH,SOL,BNB,XRP,DOGE,ADA,AVAX,LINK,LTC"
SESSION_SOURCES = ("alpaca", "dukascopy")
SESSION_TIMEFRAMES = "1h,4h,1d"
# The instruments of each asset class (research/criteria.yaml lists the ones research uses).
ASSET_CLASSES = {
    "stocks": ("alpaca", "AAPL,GOOGL,MSFT,BRK.B,XOM,AMZN,META,GE,JNJ,WFC"),
    "bonds": ("alpaca", "TLT,IEF,SHY,TIP,LQD,HYG,AGG,EMB,MUB,BNDX"),
    "commodities": ("alpaca", "GLD,SLV,USO,UNG,CPER,DBA,CORN,WEAT,PPLT,DBC"),
    "fx": ("dukascopy", "EURUSD,USDJPY,GBPUSD,AUDUSD"),
}


def main(exchange_id: str, symbols: list[str], timeframes: list[str], quote: str = "USDT") -> None:
    root = load_settings().data_paths.raw
    exchange = make_exchange(exchange_id)
    if symbols == ["ALL"]:
        symbols = list_perpetuals(exchange, quote)
        print(f"{len(symbols)} active {quote} perpetuals on {exchange_id}")
    for base in symbols:
        market = MarketId(exchange=exchange_id, base=base, quote=quote)
        try:
            counts = update_market(root, market, timeframes, exchange=exchange)
        except (ccxt.BaseError, OSError) as exc:  # one bad symbol shouldn't stop the batch
            print(f"{market.name}: FAILED ({type(exc).__name__}: {exc})")
            continue
        summary = ", ".join(f"{key}={value}" for key, value in counts.items())
        print(f"{market.name}: {summary}")


def main_sessions(source: str, symbols: list[str], timeframes: list[str], check: bool) -> int:
    """Stocks and ETFs (Alpaca) or forex (Dukascopy); returns the number of failed symbols."""

    from algotrade.data import alpaca, dukascopy, rates
    from algotrade.data.files import bars_path, read_bars
    from algotrade.data.quality import check_bars

    root = load_settings().data_paths.raw
    financing = Financing()
    if source == "alpaca":
        client, update, errors = alpaca.AlpacaClient(), alpaca.update_equity, alpaca.AlpacaError
    else:
        client, update = dukascopy.DukascopyClient(), dukascopy.update_fx
        errors = dukascopy.DukascopyError
    failed = 0
    for symbol in symbols:
        try:
            counts = update(root, symbol, timeframes, client)
        except (errors, OSError, ValueError) as exc:  # one bad symbol shouldn't stop the batch
            print(f"{symbol}: FAILED ({type(exc).__name__}: {exc})")
            failed += 1
            continue
        summary = ", ".join(f"{key}={value}" for key, value in counts.items()) or "up to date"
        print(f"{symbol}: {summary}")
        try:
            print(f"  financing: {financing.update(root, source, symbol)} charges")
        except (rates.RatesError, OSError, KeyError) as exc:
            print(f"  financing: FAILED ({type(exc).__name__}: {exc})")
        if check:
            for timeframe in timeframes:
                bars = read_bars(bars_path(root, source, symbol, timeframe))
                for issue in check_bars(bars, timeframe) if bars is not None else []:
                    print(f"  check {timeframe} {issue}")
    return failed


class Financing:
    """Interest charges for the downloaded symbols; FRED series are fetched once per run."""

    def __init__(self, session=None) -> None:
        self.session = session
        self.series: dict[str, object] = {}

    def rate(self, currency: str):
        from algotrade.data import rates

        if currency not in self.series:
            self.series[currency] = rates.fetch_series(
                rates.CURRENCY_SERIES[currency], self.session
            )
        return self.series[currency]

    def update(self, root, source: str, symbol: str) -> int:
        from algotrade.data import rates

        if source == "alpaca":
            return rates.update_us_financing(root, symbol, self.rate("USD"))
        currencies = {symbol[:3]: self.rate(symbol[:3]), symbol[3:]: self.rate(symbol[3:])}
        return rates.update_fx_rollovers(root, symbol, currencies)


def class_instruments(
    asset_class: str, criteria_path: Path = Path("research/criteria.yaml")
) -> tuple[str, str]:
    """Source and comma-separated symbols of an asset class: as research/criteria.yaml lists
    them, or the defaults above when the file has no such class."""

    from algotrade.instruments import universes
    from algotrade.research.criteria import load_criteria

    if criteria_path.exists():
        spec = universes(load_criteria(criteria_path)).get(asset_class)
        if spec:
            return spec["source"], ",".join(spec["symbols"])
    return ASSET_CLASSES[asset_class]


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        "--exchange",
        dest="source",
        choices=(*SUPPORTED_EXCHANGES, *SESSION_SOURCES),
        help="Where the data comes from (default binanceusdm, or the asset class's source)",
    )
    parser.add_argument("--asset-class", choices=sorted(ASSET_CLASSES))
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols (crypto: base assets, or ALL for every active perpetual)",
    )
    parser.add_argument("--quote", choices=QUOTES, default="USDT", help="Crypto settlement")
    parser.add_argument(
        "--timeframes",
        help=f"Crypto: any of {', '.join(TIMEFRAMES)} (default 4h,1d); others: 1h,4h,1d",
    )
    parser.add_argument("--check", action="store_true", help="Report data-quality issues")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="API keys")
    args = parser.parse_args(argv)
    if args.asset_class:
        source, symbols = class_instruments(args.asset_class)
        if args.source and args.source != source:
            parser.error(f"{args.asset_class} comes from {source}, not {args.source}")
        args.source, args.symbols = source, args.symbols or symbols
    args.source = args.source or "binanceusdm"
    if args.source in SESSION_SOURCES and not args.symbols:
        parser.error(f"--symbols or --asset-class is needed for {args.source}")
    return args


def split(text: str) -> list[str]:
    return [s.strip().upper() for s in text.split(",") if s.strip()]


def run(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.source in SESSION_SOURCES:
        load_dotenv(args.env_file)
        timeframes = [t.strip() for t in (args.timeframes or SESSION_TIMEFRAMES).split(",")]
        return main_sessions(args.source, split(args.symbols), timeframes, args.check)
    main(
        exchange_id=args.source,
        symbols=split(args.symbols or DEFAULT_UNIVERSE),
        timeframes=[t.strip() for t in (args.timeframes or "4h,1d").split(",") if t.strip()],
        quote=args.quote,
    )
    return 0


if __name__ == "__main__":
    failed = run()
    if failed:
        raise SystemExit(failed)
