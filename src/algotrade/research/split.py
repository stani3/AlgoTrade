"""The one way research code reads market data: development bars end before the holdout.

Feasibility and validation only ever call :func:`load_dev_bars`. The holdout stage calls
:func:`open_holdout` first, which journals the look and refuses a second one unless forced;
only then may :func:`load_full_bars` be used.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from algotrade.data.market import load_instrument, load_market
from algotrade.instruments import CRYPTO, registry, symbols_for

from .criteria import Criteria
from .journal import Journal, VersionState
from .workspace import Workspace


class HoldoutViolation(RuntimeError):
    """Development code asked for holdout data, or a second holdout look was attempted."""


def dev_end(criteria: Criteria) -> pd.Timestamp:
    return pd.Timestamp(criteria.get("data.dev_end"), tz="UTC")


def _stamp(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


@dataclass(frozen=True)
class DataWindow:
    symbol: str
    timeframe: str
    start: str
    end: str
    rows: int
    sha: str

    def to_dict(self) -> dict:
        return asdict(self)


def bars_hash(bars: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(bars.index.asi8, dtype=np.int64).tobytes())
    digest.update(str(bars.index.dtype).encode())
    for column in ("open", "high", "low", "close", "funding_rate"):
        if column in bars:
            digest.update(bars[column].to_numpy(dtype="float64").tobytes())
    return digest.hexdigest()[:16]


def window(bars: pd.DataFrame, symbol: str, timeframe: str) -> DataWindow:
    return DataWindow(
        symbol=symbol,
        timeframe=timeframe,
        start=str(bars.index[0]) if len(bars) else "",
        end=str(bars.index[-1]) if len(bars) else "",
        rows=len(bars),
        sha=bars_hash(bars),
    )


def load_dev_bars(
    ws: Workspace,
    criteria: Criteria,
    symbol: str,
    timeframe: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Bars opening before the holdout cutoff (and inside ``start``/``end`` when given)."""

    cutoff = dev_end(criteria)
    if end is not None and _stamp(end) > cutoff:
        raise HoldoutViolation(
            f"requested data up to {end}, but development data ends at {cutoff:%Y-%m-%d}"
        )
    bars = _load(ws, criteria, symbol, timeframe)
    stop = cutoff if end is None else _stamp(end)
    bars = bars[bars.index < stop]
    if start is not None:
        bars = bars[bars.index >= _stamp(start)]
    return bars


def dev_universe(
    ws: Workspace, criteria: Criteria, timeframe: str, symbols: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    names = symbols or symbols_for(criteria, (CRYPTO,))
    return {s: load_dev_bars(ws, criteria, s, timeframe) for s in names}


def open_holdout(
    journal: Journal, version: VersionState, force: bool = False, reason: str = ""
) -> None:
    """Record a holdout look for this idea version; a second look needs ``force``."""

    if version.holdout_looks and not force:
        raise HoldoutViolation(
            f"{version.idea} v{version.version} has already looked at the holdout "
            f"({version.holdout_looks}x); another look needs an explicit --force from the user"
        )
    journal.append(
        "holdout_look",
        idea=version.idea,
        version=version.version,
        look=version.holdout_looks + 1,
        forced=bool(version.holdout_looks),
        reason=reason,
    )


def load_full_bars(ws: Workspace, criteria: Criteria, symbol: str, timeframe: str) -> pd.DataFrame:
    """All stored bars. Only for the holdout stage (after :func:`open_holdout`) and incubation."""

    return _load(ws, criteria, symbol, timeframe)


def _load(ws: Workspace, criteria: Criteria, symbol: str, timeframe: str) -> pd.DataFrame:
    """A registered instrument from its source; any other symbol as a perpetual on the
    criteria's exchange (as before instruments were registered)."""

    instrument = registry(criteria).get(symbol)
    if instrument is None:
        return load_market(ws.raw_data, criteria.get("data.exchange"), symbol, timeframe)
    return load_instrument(ws.raw_data, instrument, timeframe)
