"""Load stored perpetual-futures history into bar frames ready for backtesting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .exchange import TIMEFRAMES, MarketId, funding_path, ohlcv_path


def align_funding(bar_open: pd.DatetimeIndex, timeframe: str, funding: pd.DataFrame) -> pd.Series:
    """Sum funding settlements into the bar during which they are charged.

    A settlement at time ``t`` is charged to whoever holds the position at ``t``, i.e. the bar
    whose interval ``(open, open + length]`` contains ``t``. A settlement exactly on a bar
    boundary therefore belongs to the bar that closes there, not the one that opens there.
    """

    length = pd.Timedelta(TIMEFRAMES[timeframe])
    if funding.empty:
        return pd.Series(0.0, index=bar_open, name="funding_rate")
    owner = (funding["timestamp"] - pd.Timedelta("1ms")).dt.floor(length)
    per_bar = funding.groupby(owner)["funding_rate"].sum()
    return per_bar.reindex(bar_open, fill_value=0.0).rename("funding_rate")


def load_market(root: Path, exchange: str, base: str, timeframe: str) -> pd.DataFrame:
    """Return OHLCV bars indexed by bar-open time (UTC) with a ``funding_rate`` column."""

    market = MarketId(exchange=exchange, base=base.upper())
    path = ohlcv_path(root, market, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"No {timeframe} data for {market.name} on {exchange}. "
            f"Run: python -m scripts.download_data --exchange {exchange} --symbols {market.base}"
        )
    bars = pd.read_parquet(path).set_index("timestamp").sort_index()
    fpath = funding_path(root, market)
    funding = (
        pd.read_parquet(fpath)
        if fpath.exists()
        else pd.DataFrame(columns=["timestamp", "funding_rate"])
    )
    bars["funding_rate"] = align_funding(bars.index, timeframe, funding)
    bars.attrs.update(exchange=exchange, symbol=market.name, timeframe=timeframe)
    return bars
