import numpy as np
import pandas as pd
import pytest

from algotrade.config import DATA_ROOT_ENV


@pytest.fixture(autouse=True)
def _no_data_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests build their own data folders; a shell's ALGOTRADE_DATA_ROOT must not leak in."""

    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)


def make_bars(n: int = 500, freq: str = "4h", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    spread = np.abs(rng.normal(0, 0.01, n))
    index = pd.date_range("2022-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": np.concatenate(([close[0]], close[:-1])),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": 1.0,
            "funding_rate": rng.normal(1e-4, 2e-4, n),
        },
        index=index,
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def long_bars() -> pd.DataFrame:
    """About 167 days of 4h bars, for rules that need months of history before they can trade
    (i004 ranks funding against its trailing year once a quarter of that year exists)."""
    return make_bars(n=1000)
