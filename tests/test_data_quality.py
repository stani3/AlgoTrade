import numpy as np
import pandas as pd

from algotrade.data.files import bars_path, read_bars, write_bars
from algotrade.data.quality import Issue, check_bars
from algotrade.data.sessions import session_buckets, us_equity_sessions


def equity_bars(timeframe: str = "1h") -> pd.DataFrame:
    sessions = us_equity_sessions("2023-03-01", "2023-03-31")
    index = pd.DatetimeIndex(session_buckets(sessions, timeframe)["start"])
    close = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.003, len(index))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
         "volume": 500.0},
        index=index,
    )  # fmt: skip


def names(issues: list[Issue]) -> list[str]:
    return [issue.check for issue in issues]


def test_clean_bars_have_no_issues() -> None:
    assert check_bars(equity_bars(), "1h") == []
    bars = equity_bars()
    crypto = bars.set_axis(pd.date_range("2023-01-01", periods=len(bars), freq="4h", tz="UTC"))
    assert check_bars(crypto, "4h") == []


def test_each_problem_is_reported() -> None:
    bars = equity_bars()
    broken = bars.copy()
    broken.iloc[5, broken.columns.get_loc("high")] = broken["open"].iloc[5] * 0.5
    broken.iloc[20:, :4] *= 1.3  # an unadjusted split-sized jump
    broken.iloc[30:40, :4] = broken["close"].iloc[29]
    broken.iloc[30:40, 4] = 0.0  # ten flat bars without trades
    broken = pd.concat([broken.drop(broken.index[50:53]), broken.iloc[[60]]])  # gap + duplicate
    found = check_bars(broken, "1h", "us_equity")
    assert names(found) == ["duplicates", "order", "ohlc", "jump", "filled", "missing"]
    assert "3 bars missing" in str(found[-1])
    outside = pd.concat(
        [bars, bars.iloc[[0]].set_axis([bars.index[0] - pd.Timedelta(hours=3)])]
    ).sort_index()
    assert names(check_bars(outside, "1h", "us_equity")) == ["outside"]
    many = equity_bars().iloc[:40]
    many.iloc[1:, :4] = many.iloc[1:, :4].mul(np.cumprod(np.full(39, 1.5)), axis=0)
    jump = check_bars(many, "1h", "us_equity")[0]
    assert jump.check == "jump" and "and 34 more" in jump.detail
    assert names(check_bars(bars.iloc[:0], "1h")) == ["empty"]


def test_files_round_trip(tmp_path) -> None:
    bars = equity_bars()
    path = bars_path(tmp_path, "alpaca", "TLT", "1h")
    assert read_bars(path) is None
    assert write_bars(bars, path) == len(bars)
    pd.testing.assert_frame_equal(read_bars(path), bars, check_freq=False, check_names=False)
    write_bars(bars.iloc[:0], path)
    assert read_bars(path) is None


def test_funding_files_sit_next_to_the_bars(tmp_path) -> None:
    from algotrade.data.files import funding_file

    assert funding_file(tmp_path, "dukascopy", "EURUSD") == (
        tmp_path / "dukascopy" / "EURUSD_funding.parquet"
    )
