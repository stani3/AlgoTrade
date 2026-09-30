import numpy as np
import pandas as pd
import pytest


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
