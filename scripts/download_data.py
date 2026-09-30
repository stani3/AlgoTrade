"""Download or incrementally update perpetual-futures history into data/raw/<exchange>/."""

from __future__ import annotations

import argparse

import ccxt

from algotrade.config import load_settings
from algotrade.data.exchange import (
    SUPPORTED_EXCHANGES,
    TIMEFRAMES,
    MarketId,
    make_exchange,
    update_market,
)

DEFAULT_UNIVERSE = "BTC,ETH,SOL,BNB,XRP,DOGE,ADA,AVAX,LINK,LTC"


def main(exchange_id: str, symbols: list[str], timeframes: list[str]) -> None:
    root = load_settings().data_paths.raw
    exchange = make_exchange(exchange_id)
    for base in symbols:
        market = MarketId(exchange=exchange_id, base=base)
        try:
            counts = update_market(root, market, timeframes, exchange=exchange)
        except (ccxt.BaseError, OSError) as exc:  # one bad symbol shouldn't stop the batch
            print(f"{market.name}: FAILED ({type(exc).__name__}: {exc})")
            continue
        summary = ", ".join(f"{key}={value}" for key, value in counts.items())
        print(f"{market.name}: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", choices=SUPPORTED_EXCHANGES, default="binanceusdm")
    parser.add_argument("--symbols", default=DEFAULT_UNIVERSE, help="Comma-separated base assets")
    parser.add_argument("--timeframes", default="4h,1d", help=f"Any of {', '.join(TIMEFRAMES)}")
    args = parser.parse_args()
    main(
        exchange_id=args.exchange,
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        timeframes=[t.strip() for t in args.timeframes.split(",") if t.strip()],
    )
