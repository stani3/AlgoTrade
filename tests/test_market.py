import pandas as pd
import pytest

from algotrade.data.exchange import TIMEFRAMES, MarketId, funding_path, ohlcv_path
from algotrade.data.market import align_funding, load_market


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


def test_no_funding_history_means_zero_funding() -> None:
    bars = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    empty = pd.DataFrame(columns=["timestamp", "funding_rate"])
    assert align_funding(bars, "4h", empty).tolist() == [0.0, 0.0, 0.0]
    assert align_funding(bars, "4h", empty, "nearest_second").tolist() == [0.0, 0.0, 0.0]


# Binance stamps its settlements 0-47 ms after the hour, stored at millisecond resolution. The
# last one is a genuine mid-bar settlement (stamped late too). Rates are powers of ten so every
# sum shows which settlements a bar collected.
SETTLEMENTS = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(
            [
                "2024-01-01 08:00:00.000",
                "2024-01-01 16:00:00.023",
                "2024-01-02 00:00:00.047",
                "2024-01-02 10:30:00.023",
            ],
            utc=True,
        ).astype("datetime64[ms, UTC]"),
        "funding_rate": [1.0, 10.0, 100.0, 1000.0],
    }
)


def bar_opens(timeframe: str) -> pd.DatetimeIndex:
    step = pd.Timedelta(TIMEFRAMES[timeframe])
    return pd.date_range("2024-01-01", "2024-01-03", freq=step, tz="UTC", inclusive="left")


def charged(timeframe: str, alignment: str) -> dict[str, float]:
    """Bars (by open, ``MM-DD HH:MM``) charged any funding, with the sum each was charged."""

    aligned = align_funding(bar_opens(timeframe), timeframe, SETTLEMENTS, alignment)
    assert aligned.sum() == 1111.0  # every settlement is charged exactly once
    return {f"{t:%m-%d %H:%M}": rate for t, rate in aligned[aligned != 0].items()}


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("1h", {"01-01 07:00": 1, "01-01 15:00": 10, "01-01 23:00": 100, "01-02 10:00": 1000}),
        ("4h", {"01-01 04:00": 1, "01-01 12:00": 10, "01-01 20:00": 100, "01-02 08:00": 1000}),
        ("1d", {"01-01 00:00": 111, "01-02 00:00": 1000}),
    ],
)
def test_late_stamps_are_charged_to_the_bar_closing_at_the_settlement(timeframe, expected) -> None:
    # 08:00:00.000, 16:00:00.023 and 00:00:00.047 all settle at the close of the bar that ends on
    # the hour; 10:30:00.023 falls inside a bar and stays there.
    assert charged(timeframe, "nearest_second") == expected


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("1h", {"01-01 07:00": 1, "01-01 16:00": 10, "01-02 00:00": 100, "01-02 10:00": 1000}),
        ("4h", {"01-01 04:00": 1, "01-01 16:00": 10, "01-02 00:00": 100, "01-02 08:00": 1000}),
        ("1d", {"01-01 00:00": 11, "01-02 00:00": 1100}),
    ],
)
def test_raw_timestamp_alignment_charges_late_stamps_one_bar_late(timeframe, expected) -> None:
    # The legacy alignment, kept for results recorded with it: a few milliseconds past the hour
    # puts a settlement on the bar that opens at its settlement hour.
    assert charged(timeframe, "raw_timestamp") == expected


def test_raw_timestamp_is_the_default() -> None:
    opens = bar_opens("1h")
    pd.testing.assert_series_equal(
        align_funding(opens, "1h", SETTLEMENTS),
        align_funding(opens, "1h", SETTLEMENTS, "raw_timestamp"),
    )


def test_unknown_alignment_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown funding alignment 'nearest_minute'"):
        align_funding(bar_opens("4h"), "4h", SETTLEMENTS, "nearest_minute")


def test_load_market_aligns_funding_as_asked(tmp_path) -> None:
    market = MarketId(exchange="binanceusdm", base="BTC")
    path = ohlcv_path(tmp_path, market, "4h")
    path.parent.mkdir(parents=True)
    opens = bar_opens("4h")
    prices = {column: 1.0 for column in ("open", "high", "low", "close", "volume")}
    pd.DataFrame({"timestamp": opens, **prices}).to_parquet(path, index=False)
    SETTLEMENTS.to_parquet(funding_path(tmp_path, market), index=False)

    legacy = load_market(tmp_path, "binanceusdm", "BTC", "4h")
    fixed = load_market(tmp_path, "binanceusdm", "BTC", "4h", funding_alignment="nearest_second")
    assert legacy.attrs["funding_alignment"] == "raw_timestamp"
    assert fixed.attrs["funding_alignment"] == "nearest_second"
    pd.testing.assert_series_equal(
        fixed["funding_rate"],
        align_funding(opens, "4h", SETTLEMENTS, "nearest_second"),
        check_names=False,
        check_freq=False,
    )
    assert fixed["funding_rate"][pd.Timestamp("2024-01-01 12:00", tz="UTC")] == 10.0
    assert legacy["funding_rate"][pd.Timestamp("2024-01-01 16:00", tz="UTC")] == 10.0
