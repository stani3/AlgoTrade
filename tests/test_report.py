"""Davey-style performance report: trade statistics, drawdowns, periods, Monte Carlo, HTML."""

import numpy as np
import pandas as pd
import pytest
from bracket_helpers import random_bars
from conftest import make_bars

from algotrade.backtest.costs import CostModel
from algotrade.backtest.engine import run_backtest
from algotrade.backtest.report import (
    DAVEY_CRITERIA,
    MC_KEYS,
    TRADE_KEYS,
    annual_analysis,
    build_report,
    drawdown_stats,
    max_streak,
    monte_carlo,
    monthly_returns,
    side_stats,
    trade_summary,
)
from algotrade.backtest.report_html import fmt, render_html, write_report
from algotrade.strategies import BreakoutBracket

COSTS = CostModel(fee_bps=5.0, slippage_bps=3.0)


def daily(values, start="2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def trade_frame(returns, money, bars, direction, is_open=None) -> pd.DataFrame:
    n = len(returns)
    return pd.DataFrame(
        {
            "direction": direction,
            "return": returns,
            "pnl_usd": money,
            "bars": bars,
            "open": is_open if is_open is not None else [False] * n,
        }
    )


TRADES = trade_frame(
    returns=[0.10, -0.05, 0.02, -0.05, -0.01, 0.0],
    money=[100.0, -50.0, 20.0, -50.0, -10.0, 0.0],
    bars=[5, 3, 4, 2, 1, 1],
    direction=[1, 1, -1, -1, 1, -1],
)


def test_side_stats_by_hand() -> None:
    stats = side_stats(TRADES)
    assert stats["net_profit"] == pytest.approx(10.0)
    assert stats["gross_profit"] == pytest.approx(120.0)
    assert stats["gross_loss"] == pytest.approx(-110.0)
    assert stats["profit_factor"] == pytest.approx(120 / 110)
    assert (stats["trades"], stats["winners"], stats["losers"], stats["even"]) == (6, 2, 3, 1)
    assert stats["win_rate"] == pytest.approx(2 / 6)
    assert stats["avg_trade"] == pytest.approx(0.01 / 6)
    assert stats["avg_trade_usd"] == pytest.approx(10 / 6)
    assert stats["avg_win"] == pytest.approx(0.06)
    assert stats["avg_loss"] == pytest.approx(-0.11 / 3)
    assert stats["win_loss_ratio"] == pytest.approx(0.06 / (0.11 / 3))
    assert stats["largest_win"] == pytest.approx(0.10)
    assert stats["largest_loss"] == pytest.approx(-0.05)
    assert stats["largest_win_share"] == pytest.approx(100 / 120)
    assert stats["largest_loss_share"] == pytest.approx(50 / 110)
    assert stats["max_consec_wins"] == 1
    assert stats["max_consec_losses"] == 2
    assert stats["avg_bars"] == pytest.approx(16 / 6)
    assert stats["avg_bars_win"] == pytest.approx(4.5)
    assert stats["avg_bars_loss"] == pytest.approx(2.0)


def test_side_stats_edge_cases() -> None:
    only_wins = side_stats(trade_frame([0.1, 0.2], [10.0, 20.0], [1, 1], [1, 1]))
    assert only_wins["profit_factor"] == float("inf")
    assert np.isnan(only_wins["win_loss_ratio"]) and np.isnan(only_wins["largest_loss"])
    empty = side_stats(trade_frame([], [], [], []))
    assert empty["trades"] == 0 and np.isnan(empty["profit_factor"])
    assert np.isnan(empty["win_rate"]) and empty["net_profit"] == 0


def test_trade_summary_splits_sides_and_skips_open_trades() -> None:
    trades = pd.concat(
        [TRADES, trade_frame([0.5], [500.0], [9], [1], is_open=[True])], ignore_index=True
    )
    table = trade_summary(trades)
    assert list(table.index) == TRADE_KEYS
    assert list(table.columns) == ["All trades", "Long trades", "Short trades"]
    assert table.loc["trades"].tolist() == [6, 3, 3]
    assert table.loc["net_profit", "Long trades"] == pytest.approx(40.0)
    assert table.loc["net_profit", "Short trades"] == pytest.approx(-30.0)
    assert table.loc["largest_win", "All trades"] == pytest.approx(0.10)  # open 50% excluded


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], 0), ([False], 0), ([True, True, False, True], 2), ([False, True, True, True], 3)],
)
def test_max_streak(flags, expected) -> None:
    assert max_streak(np.array(flags, dtype=bool)) == expected


def test_drawdown_stats_by_hand() -> None:
    equity = daily([1.1, 0.99, 1.0, 1.2, 0.9, 1.0])
    stats = drawdown_stats(equity, capital=100.0)
    assert stats["max_drawdown"] == pytest.approx(0.9 / 1.2 - 1)
    assert stats["max_drawdown_usd"] == pytest.approx(30.0)
    assert stats["dd_peak"] == equity.index[3]
    assert stats["dd_trough"] == equity.index[4]
    assert pd.isna(stats["dd_recovery"])
    assert stats["longest_drawdown_days"] == pytest.approx(2.0)
    assert stats["time_underwater"] == pytest.approx(4 / 6)


def test_drawdown_recovery_and_starting_capital_peak() -> None:
    recovered = drawdown_stats(daily([1.0, 1.2, 0.6, 1.3]), capital=1.0)
    assert recovered["max_drawdown"] == pytest.approx(-0.5)
    assert recovered["dd_recovery"] == daily([0] * 4).index[3]
    # Losing from the first bar is a drawdown from the starting capital.
    losing = drawdown_stats(daily([0.9, 0.95]), capital=1.0)
    assert losing["max_drawdown"] == pytest.approx(-0.1)
    assert losing["dd_peak"] == daily([0]).index[0]
    rising = drawdown_stats(daily([1.0, 1.1, 1.2]), capital=1.0)
    assert rising["max_drawdown"] == 0 and pd.isna(rising["dd_trough"])


def test_annual_and_monthly_returns_compound_consistently() -> None:
    rng = np.random.default_rng(3)
    equity = daily(np.cumprod(1 + rng.normal(0.001, 0.02, 800)), start="2023-03-15")
    no_trades = trade_frame([], [], [], []).assign(exit=pd.Series(dtype="datetime64[ns, UTC]"))
    annual = annual_analysis(equity, no_trades, capital=1000.0)
    monthly = monthly_returns(equity, annual)
    assert list(annual.index) == [2023, 2024, 2025]
    assert np.prod(1 + annual["return"]) - 1 == pytest.approx(equity.iloc[-1] - 1)
    assert annual["net_profit"].sum() == pytest.approx((equity.iloc[-1] - 1) * 1000)
    months = monthly.drop(columns="Year")
    for year, row in months.iterrows():
        assert np.prod(1 + row.dropna()) - 1 == pytest.approx(monthly.loc[year, "Year"])
    assert months.loc[2023, ["Jan", "Feb"]].isna().all()
    assert (annual["max_drawdown"] <= 0).all()


def test_annual_trades_are_counted_by_exit_year() -> None:
    equity = daily(np.ones(400), start="2024-06-01")
    trades = trade_frame([0.1, -0.1, 0.2], [1.0, -1.0, 2.0], [1, 1, 1], [1, 1, 1])
    trades["exit"] = pd.to_datetime(["2024-07-01", "2025-01-02", "2025-03-01"], utc=True)
    annual = annual_analysis(equity, trades, capital=1.0)
    assert annual["trades"].tolist() == [1, 2]
    assert annual["win_rate"].tolist() == [1.0, 0.5]


def test_monte_carlo_all_winners_never_draws_down() -> None:
    mc = monte_carlo(np.full(20, 0.01), trades_per_run=12, runs=200)
    row = mc.table.loc[1.0]
    assert row["risk_of_ruin"] == 0 and row["median_max_dd"] == 0
    assert row["median_return"] == pytest.approx(1.01**12 - 1)
    assert row["prob_profit"] == 1 and row["return_dd"] == float("inf")
    assert mc.paths.shape == (200, 13) and len(mc.max_dd) == 200


def test_monte_carlo_ruin_stops_trading() -> None:
    mc = monte_carlo(np.full(5, -0.1), trades_per_run=20, runs=50, ruin=0.5)
    row = mc.table.loc[1.0]
    assert row["risk_of_ruin"] == 1.0
    # 0.9^7 is the first value at or below 0.5; after that the run stops trading.
    assert row["median_return"] == pytest.approx(0.9**7 - 1)
    np.testing.assert_allclose(mc.paths[:, 7:], 0.9**7)


def test_monte_carlo_sizes_share_draws_and_scale_risk() -> None:
    returns = np.random.default_rng(1).normal(0.005, 0.04, 300)
    mc = monte_carlo(returns, trades_per_run=50, runs=500, multipliers=(0.5, 2.0), seed=4)
    assert list(mc.table.index) == [0.5, 1.0, 2.0]  # 1x is always included
    assert list(mc.table.columns) == MC_KEYS
    dd = mc.table["median_max_dd"]
    assert dd[0.5] < dd[1.0] < dd[2.0]
    again = monte_carlo(returns, trades_per_run=50, runs=500, multipliers=(0.5, 2.0), seed=4)
    pd.testing.assert_frame_equal(mc.table, again.table)


def test_davey_checks_use_the_thresholds() -> None:
    mc = monte_carlo(np.full(20, 0.01), trades_per_run=12, runs=50)
    checks = mc.checks()
    assert [ok for *_, ok in checks] == [True, True, True]
    mc.table.loc[1.0, "median_max_dd"] = DAVEY_CRITERIA["median_max_dd"]
    mc.table.loc[1.0, "return_dd"] = 1.0
    assert [ok for *_, ok in mc.checks()] == [True, False, False]


def continuous_target(bars: pd.DataFrame) -> pd.Series:
    """Long/short with resizing every bar, so trades include reversals and rebalancing."""

    close = bars["close"]
    score = (close / close.rolling(10).mean() - 1) * 20
    return score.clip(-1, 1).fillna(0.0)


@pytest.mark.parametrize("seed", range(4))
def test_trade_money_adds_up_to_net_profit_for_the_vectorized_engine(seed) -> None:
    bars = make_bars(n=600, seed=seed)
    result = run_backtest(bars, continuous_target(bars), COSTS)
    report = build_report(result, capital=5000.0, mc_runs=100)
    assert report.trades["direction"].nunique() == 2
    assert report.trades["pnl_usd"].sum() == pytest.approx(report.account["net_profit"])
    costs = result.ledger["trading_cost"] * result.equity.shift(1, fill_value=1.0) * 5000
    assert report.account["costs_paid"] == pytest.approx(costs.sum())


@pytest.mark.parametrize("seed", range(4))
def test_trade_money_adds_up_to_net_profit_for_brackets(seed) -> None:
    bars = random_bars(seed, n=800)
    strategy = BreakoutBracket(lookback=5, rsi_length=5, atr_length=5, kill_drawdown=1.0)
    report = build_report(strategy.simulate(bars, COSTS), capital=10_000.0, mc_runs=100)
    assert len(report.trades) > 10
    assert report.trades["pnl_usd"].sum() == pytest.approx(report.account["net_profit"])
    closed = report.trades[~report.trades["open"]]
    summary = report.trade_summary["All trades"]
    assert summary["net_profit"] == pytest.approx(closed["pnl_usd"].sum())


def test_build_report_sections(bars) -> None:
    result = run_backtest(bars, continuous_target(bars), COSTS)
    benchmark = run_backtest(bars, pd.Series(1.0, index=bars.index), COSTS)
    report = build_report(result, benchmark=benchmark, mc_runs=100)
    assert report.name == "backtest"
    assert report.equity.iloc[0] == pytest.approx(result.equity.iloc[0] * 10_000)
    assert report.monte_carlo is not None
    assert {"benchmark_cagr", "benchmark_max_drawdown"} <= set(report.account)
    assert report.account["ending_equity"] == pytest.approx(10_000 + report.account["net_profit"])
    assert list(report.monthly.columns)[-1] == "Year"


def test_kill_switch_is_flagged_and_monte_carlo_uses_the_active_period() -> None:
    bars = random_bars(2, n=1500)
    strategy = BreakoutBracket(lookback=5, rsi_length=5, atr_length=5, kill_drawdown=0.15)
    result = strategy.simulate(bars, COSTS)
    killed = result.meta["killed_at"]
    report = build_report(result, mc_runs=100)
    assert report.account["stopped_at"] == killed
    assert any("Trading stopped" in note for note in report.notes)
    closed = report.trades[~report.trades["open"]]
    years = (killed - bars.index[0]) / pd.Timedelta(days=365.25)
    if report.monte_carlo is not None:
        assert report.monte_carlo.trades_per_run == max(round(len(closed) / years), 1)


def test_few_trades_skip_monte_carlo(bars) -> None:
    target = pd.Series(0.0, index=bars.index)
    target.iloc[100:150] = 1.0
    report = build_report(run_backtest(bars, target, COSTS))
    assert report.monte_carlo is None
    assert any("Monte Carlo skipped" in note for note in report.notes)
    page = render_html([report], "one trade", {})
    assert "Monte Carlo skipped" in page


def test_open_trade_is_noted(bars) -> None:
    target = pd.Series(0.0, index=bars.index)
    target.iloc[-20:] = 1.0
    report = build_report(run_backtest(bars, target, COSTS))
    assert any("Open trade" in note for note in report.notes)


def test_no_trades_renders(bars) -> None:
    report = build_report(run_backtest(bars, pd.Series(0.0, index=bars.index), COSTS))
    page = render_html([report], "flat", {})
    assert "No closed trades" in page and "No trades." in page


def test_write_report(tmp_path) -> None:
    reports = []
    for seed, symbol in enumerate(["AAAUSDT", "BBB/USDT"]):
        bars = random_bars(seed, n=800)
        bars.attrs["symbol"] = symbol
        result = BreakoutBracket(lookback=5, rsi_length=5, atr_length=5).simulate(bars, COSTS)
        benchmark = run_backtest(bars, pd.Series(1.0, index=bars.index), COSTS)
        reports.append(build_report(result, benchmark=benchmark, mc_runs=100))
    page = write_report(reports, tmp_path / "out", "test strategy", {"Spec": '{"a": 1}'})
    text = page.read_text(encoding="utf-8")
    for heading in ("Overview", "Trade analysis", "Annual analysis", "Monthly returns",
                    "Monte Carlo (Davey)", "List of trades", "How to read this report"):  # fmt: skip
        assert heading in text
    assert "AAAUSDT" in text and "BBB/USDT" in text
    assert text.count("<img") == 2 * 3  # equity, trades and Monte Carlo charts per market
    assert "&quot;a&quot;" in text  # settings are escaped
    assert (tmp_path / "out" / "trades_AAAUSDT.csv").exists()
    assert (tmp_path / "out" / "trades_BBB_USDT.csv").exists()
    summary = pd.read_csv(tmp_path / "out" / "summary.csv", index_col=0)
    assert list(summary.index) == ["AAAUSDT", "BBB/USDT"]


@pytest.mark.parametrize(
    ("value", "kind", "text"),
    [
        (1234.4, "usd", "$1,234"),
        (-1234.6, "usd", "-$1,235"),
        (0.1234, "pct", "12.3%"),
        (0.01234, "pct2", "1.23%"),
        (1.5, "ratio", "1.50"),
        (float("inf"), "ratio", "∞"),
        (float("nan"), "pct", "–"),
        (pd.NaT, "date", "–"),
        (pd.Timestamp("2024-05-06 08:00", tz="UTC"), "date", "2024-05-06"),
        (12.0, "int", "12"),
        (45.4, "days", "45 days"),
    ],
)
def test_fmt(value, kind, text) -> None:
    assert fmt(value, kind) == text
