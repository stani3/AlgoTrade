"""Load stored history into bar frames ready for backtesting.

Crypto perpetuals carry their funding settlements; other instruments carry the financing of
the position (interest on the money behind it) in the same ``funding_rate`` column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .exchange import TIMEFRAMES, MarketId, funding_path, ohlcv_path
from .files import bars_path, funding_file, read_bars


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


def load_market(
    root: Path, exchange: str, base: str, timeframe: str, quote: str = "USDT"
) -> pd.DataFrame:
    """Return OHLCV bars indexed by bar-open time (UTC) with a ``funding_rate`` column."""

    market = MarketId(exchange=exchange, base=base.upper(), quote=quote.upper())
    path = ohlcv_path(root, market, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"No {timeframe} data for {market.name} on {exchange}. "
            f"Run: python -m scripts.download_data --exchange {exchange} --quote {market.quote} "
            f"--symbols {market.base}"
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


def align_to_bars(bar_open: pd.DatetimeIndex, charges: pd.DataFrame) -> pd.Series:
    """Sum financing charges into the bar that holds the position they are charged on.

    A charge at time ``t`` (an overnight interest charge at a session's close, a forex
    rollover at 17:00 New York) is paid on the position held from ``t`` on: the first bar
    opening at or after ``t``, whose position was decided at the close before it. Charges
    before the first bar or after the last bar's open belong to no bar here.
    """

    if charges.empty or not len(bar_open):
        return pd.Series(0.0, index=bar_open, name="funding_rate")
    stamps = pd.DatetimeIndex(charges["timestamp"])
    keep = np.asarray((stamps >= bar_open[0]) & (stamps <= bar_open[-1]))
    owner = bar_open.searchsorted(stamps[keep], side="left")
    rates = charges["funding_rate"].to_numpy(dtype="float64")[keep]
    sums = np.bincount(owner, weights=rates, minlength=len(bar_open))
    return pd.Series(sums, index=bar_open, name="funding_rate")


def load_instrument(root: Path, instrument, timeframe: str) -> pd.DataFrame:
    """Bars of an :class:`algotrade.instruments.Instrument`, with its ``funding_rate``.

    Crypto goes through :func:`load_market` exactly as before. Other instruments start at the
    instrument's first date and carry their financing charges (zero until downloaded).
    """

    if instrument.is_crypto:
        return load_market(root, instrument.source, instrument.symbol, timeframe, instrument.quote)
    bars = read_bars(bars_path(root, instrument.source, instrument.symbol, timeframe))
    if bars is None:
        raise FileNotFoundError(
            f"No {timeframe} data for {instrument.symbol} from {instrument.source}. Run: "
            f"python -m scripts.download_data --asset-class {instrument.asset_class}"
        )
    if instrument.start is not None:
        bars = bars[bars.index >= instrument.start]
    charges = read_bars(funding_file(root, instrument.source, instrument.symbol))
    charges = (
        charges.reset_index()
        if charges is not None
        else pd.DataFrame({"timestamp": pd.DatetimeIndex([], tz="UTC"), "funding_rate": []})
    )
    bars["funding_rate"] = align_to_bars(bars.index, charges)
    bars.attrs.update(exchange=instrument.source, symbol=instrument.symbol, timeframe=timeframe)
    return bars
