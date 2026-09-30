import pandas as pd

from algotrade.data.market import align_funding


def funding(*stamps: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(list(stamps), utc=True), "funding_rate": [1.0] * len(stamps)}
    )


def test_settlement_on_boundary_belongs_to_the_bar_that_closes_there() -> None:
    bars = pd.date_range("2024-01-01 00:00", periods=6, freq="4h", tz="UTC")
    aligned = align_funding(bars, "4h", funding("2024-01-01 08:00", "2024-01-01 16:00"))
    # 08:00 settles at the close of the 04:00 bar; 16:00 at the close of the 12:00 bar.
    assert aligned.tolist() == [0, 1, 0, 1, 0, 0]


def test_daily_bar_collects_its_three_settlements() -> None:
    bars = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    settlements = funding(
        "2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 00:00", "2024-01-02 08:00"
    )
    assert align_funding(bars, "1d", settlements).tolist() == [3, 1]
