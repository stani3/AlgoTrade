"""Load stored perpetual-futures history into bar frames ready for backtesting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .exchange import TIMEFRAMES, MarketId, funding_path, ohlcv_path

# How a settlement's recorded timestamp is turned into the instant it was charged.
# ``raw_timestamp`` is what every research result before ``data.funding_alignment`` used; it stays
# the default so those results keep reproducing.
FUNDING_ALIGNMENTS = ("raw_timestamp", "nearest_second")
LEGACY_FUNDING_ALIGNMENT = "raw_timestamp"


def align_funding(
    bar_open: pd.DatetimeIndex,
    timeframe: str,
    funding: pd.DataFrame,
    alignment: str = LEGACY_FUNDING_ALIGNMENT,
) -> pd.Series:
    """Sum funding settlements into the bar during which they are charged.

    A settlement at time ``t`` is charged to whoever holds the position at ``t``, i.e. the bar
    whose interval ``(open, open + length]`` contains ``t``. A settlement exactly on a bar
    boundary therefore belongs to the bar that closes there, not the one that opens there.

    Binance records settlements 0-47 ms after the hour (08:00:00.023, say), so ``alignment``
    decides what ``t`` is:

    * ``nearest_second``: the recorded timestamp rounded to the nearest second, so every
      settlement is charged to the bar that closes at its settlement hour.
    * ``raw_timestamp`` (legacy): the recorded timestamp as it is, which charges the ~45% of
      settlements stamped a few milliseconds late one bar late, to the bar that opens at the
      settlement hour (on 1h, 4h and 1d bars alike).
    """

    if alignment not in FUNDING_ALIGNMENTS:
        raise ValueError(
            f"unknown funding alignment {alignment!r}; expected one of {FUNDING_ALIGNMENTS}"
        )
    length = pd.Timedelta(TIMEFRAMES[timeframe])
    if funding.empty:
        return pd.Series(0.0, index=bar_open, name="funding_rate")
    charged = funding["timestamp"]
    if alignment == "nearest_second":
        charged = charged.dt.round("1s")
    owner = (charged - pd.Timedelta("1ms")).dt.floor(length)
    per_bar = funding.groupby(owner)["funding_rate"].sum()
    return per_bar.reindex(bar_open, fill_value=0.0).rename("funding_rate")


def load_market(
    root: Path,
    exchange: str,
    base: str,
    timeframe: str,
    quote: str = "USDT",
    funding_alignment: str = LEGACY_FUNDING_ALIGNMENT,
) -> pd.DataFrame:
    """Return OHLCV bars indexed by bar-open time (UTC) with a ``funding_rate`` column.

    ``funding_alignment`` is passed to :func:`align_funding`.
    """

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
    bars["funding_rate"] = align_funding(bars.index, timeframe, funding, funding_alignment)
    bars.attrs.update(
        exchange=exchange,
        symbol=market.name,
        timeframe=timeframe,
        funding_alignment=funding_alignment,
    )
    return bars
