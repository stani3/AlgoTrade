import pandas as pd
import pytest
from http_fakes import FakeResponse, FakeSession

from algotrade.data import rates
from algotrade.data.files import bars_path, funding_file, write_bars
from algotrade.data.market import align_to_bars
from algotrade.data.rates import (
    RatesError,
    daily,
    fetch_series,
    fx_rollovers,
    parse_fred,
    update_fx_rollovers,
    update_us_financing,
    us_financing,
)
from algotrade.data.sessions import session_buckets, us_equity_sessions

NY = "America/New_York"


def constant(value: float, start: str = "2023-01-01") -> pd.Series:
    return pd.Series([value], index=pd.DatetimeIndex([start]))


def test_fred_files() -> None:
    parsed = parse_fred("observation_date,DFF\n2024-01-01,5.33\n2024-01-02,.\n2024-01-03,5.31\n")
    assert parsed.name == "DFF"
    assert parsed.iloc[0] == pytest.approx(0.0533) and pd.isna(parsed.iloc[1])
    with pytest.raises(RatesError, match="unexpected"):
        parse_fred("a,b,c\n1,2,3\n")
    ok = FakeSession(lambda url, params: FakeResponse(200, b"observation_date,X\n2024-01-01,1\n"))
    assert fetch_series("X", ok).iloc[0] == 0.01
    assert ok.calls[0][0].endswith("id=X")
    with pytest.raises(RatesError, match="HTTP 500"):
        fetch_series("X", FakeSession(lambda url, params: FakeResponse(500)))
    monthly = pd.Series([0.01, 0.02], index=pd.DatetimeIndex(["2024-01-01", "2024-02-01"]))
    held = daily(monthly, pd.Timestamp("2024-02-10"))
    assert held["2024-01-31"] == 0.01 and held["2024-02-10"] == 0.02


def test_us_financing_charges_overnight_and_weekends() -> None:
    sessions = us_equity_sessions("2024-03-27", "2024-04-02")  # Good Friday 29 March is closed
    charges = us_financing(sessions, constant(0.036))
    local = pd.DatetimeIndex(charges["timestamp"]).tz_convert(NY)
    assert local.strftime("%m-%d %H:%M").tolist() == ["03-27 16:00", "03-28 16:00", "04-01 16:00"]
    assert charges["funding_rate"].tolist() == pytest.approx(
        [0.036 / 360, 0.036 * 4 / 360, 0.036 / 360]
    )
    assert us_financing(sessions.iloc[:1], constant(0.03)).empty


def test_long_pays_and_short_receives_the_overnight_rate() -> None:
    from algotrade.backtest.costs import CostModel
    from algotrade.backtest.engine import run_backtest

    sessions = us_equity_sessions("2024-01-02", "2024-01-31")
    index = pd.DatetimeIndex(session_buckets(sessions, "1d")["start"])
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                         "volume": 1.0}, index=index)  # fmt: skip
    bars["funding_rate"] = align_to_bars(index, us_financing(sessions, constant(0.05)))
    free = CostModel(0.0, 0.0)
    long = run_backtest(bars, pd.Series(1.0, index=index), free)
    short = run_backtest(bars, pd.Series(-1.0, index=index), free)
    days = (index[-1].normalize() - index[0].normalize()).days
    assert long.equity.iloc[-1] < 1 < short.equity.iloc[-1]
    assert 1 - long.equity.iloc[-1] == pytest.approx(0.05 * days / 360, rel=0.01)


def test_fx_rollovers_triple_on_wednesday_and_follow_new_york() -> None:
    usd, eur = constant(0.05), constant(0.03)
    charges = fx_rollovers(eur, usd, pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-15 17:00"))
    local = pd.DatetimeIndex(charges["timestamp"]).tz_convert(NY)
    assert set(local.strftime("%H:%M")) == {"17:00"}
    assert len(charges) == 10
    wednesday = charges["funding_rate"][local.dayofweek == 2]
    assert wednesday.tolist() == pytest.approx([3 * 0.02 / 360] * 2)
    assert {t.hour for t in pd.DatetimeIndex(charges["timestamp"])} == {21, 22}  # DST on 10 March
    # long EURUSD earns EUR and pays USD: a cost while USD rates are higher
    assert (charges["funding_rate"] > 0).all()
    weekend = fx_rollovers(eur, usd, pd.Timestamp("2024-03-09"), pd.Timestamp("2024-03-10"))
    assert weekend.empty


def test_updates_extend_the_files_and_never_rewrite(tmp_path, monkeypatch) -> None:
    sessions = us_equity_sessions("2024-01-02", "2024-02-29")
    daily_bars = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.DatetimeIndex(session_buckets(sessions, "1d")["start"]),
    )
    write_bars(daily_bars.iloc[:20], bars_path(tmp_path, "alpaca", "TLT", "1d"))
    first = update_us_financing(tmp_path, "TLT", constant(0.05))
    stored = pd.read_parquet(funding_file(tmp_path, "alpaca", "TLT"))
    write_bars(daily_bars, bars_path(tmp_path, "alpaca", "TLT", "1d"))
    revised = constant(0.07)  # FRED revised its history: stored charges must not change
    more = update_us_financing(tmp_path, "TLT", revised)
    after = pd.read_parquet(funding_file(tmp_path, "alpaca", "TLT"))
    assert first == 19 and more == len(daily_bars) - 1
    pd.testing.assert_frame_equal(after.iloc[:first], stored)
    assert after["funding_rate"].iloc[-1] == pytest.approx(0.07 / 360)
    assert update_us_financing(tmp_path, "IEF", constant(0.05)) == 0

    hours = pd.date_range("2024-03-03 22:00", "2024-03-08 21:00", freq="1h", tz="UTC")
    write_bars(daily_bars.iloc[:1].reindex(hours, method="ffill"),
               bars_path(tmp_path, "dukascopy", "EURUSD", "1h"))  # fmt: skip
    count = update_fx_rollovers(tmp_path, "EURUSD", {"EUR": constant(0.03), "USD": constant(0.05)})
    assert count == 4  # Monday to Thursday; Friday's 17:00 is after the last stored hour
    assert update_fx_rollovers(tmp_path, "USDJPY", {}) == 0
    assert set(rates.CURRENCY_SERIES) == {"USD", "EUR", "JPY", "GBP", "AUD"}
