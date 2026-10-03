"""Tested strategies never change behaviour: every committed fingerprint must reproduce exactly.

A failure means code (a strategy, an indicator, the engine) changed how an already-tested
strategy trades. Revisions must add a defaulted parameter or a new class instead. Each
fingerprint is recomputed on the data it was made from: the funding alignment its sidecar
records (fingerprints from before ``data.funding_alignment`` existed used the legacy one).
"""

from pathlib import Path

import pandas as pd
import pytest

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.data.market import LEGACY_FUNDING_ALIGNMENT
from algotrade.research import fingerprint
from algotrade.research.criteria import load_criteria
from algotrade.research.split import FUNDING_ALIGNMENT, dev_universe
from algotrade.research.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
WS = Workspace(ROOT)
STORE = fingerprint.FingerprintStore(WS.fingerprints_dir)
KEYS = STORE.keys()


@pytest.fixture(scope="module")
def universes():
    criteria = load_criteria(WS.criteria_path)
    cache = {}

    def get(timeframe: str, symbols: tuple[str, ...], alignment: str):
        key = timeframe, symbols, alignment
        if key not in cache:
            pinned = criteria.with_value(FUNDING_ALIGNMENT, alignment)
            try:
                cache[key] = dev_universe(WS, pinned, timeframe, list(symbols))
            except FileNotFoundError:
                pytest.skip("market data not downloaded")
        return cache[key]

    return get


@pytest.mark.parametrize("key", KEYS)
def test_committed_fingerprints_reproduce(key, universes) -> None:
    stored, meta = STORE.load(key)
    costs = EXCHANGE_COSTS[meta.get("exchange", "binanceusdm")]
    alignment = meta.get("funding_alignment", LEGACY_FUNDING_ALIGNMENT)
    universe = universes(meta["timeframe"], tuple(meta["symbols"]), alignment)
    fresh = fingerprint.compute(meta["spec"], universe, costs)
    pd.testing.assert_frame_equal(fresh, stored, check_freq=False, rtol=0, atol=0)
