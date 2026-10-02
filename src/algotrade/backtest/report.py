"""Strategy performance report in the style of Davey's *Building Winning Algorithmic Trading
Systems*: a TradeStation-like trade summary split into all/long/short trades, account and drawdown
statistics, annual and monthly period analysis, and Davey's Monte Carlo check on the trade list.

Trades are sized as a fraction of current equity, so the account compounds. Per-trade figures are
therefore given as a percentage of the equity at entry; money figures follow the compounded
account that starts with ``capital``. Drawdowns are measured close to close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import BacktestResult
from .metrics import YEAR, summarize

# Davey's rules of thumb for the Monte Carlo test (one year of trades, at the size you will trade).
DAVEY_CRITERIA = {"risk_of_ruin": 0.10, "median_max_dd": 0.40, "return_dd": 2.0}
MIN_MC_TRADES = 10

# key -> (label, kind); kind selects the formatting in report_html.
FIELDS: dict[str, tuple[str, str]] = {
    # trade summary
    "net_profit": ("Total net profit", "usd"),
    "gross_profit": ("Gross profit", "usd"),
    "gross_loss": ("Gross loss", "usd"),
    "profit_factor": ("Profit factor", "ratio"),
    "trades": ("Total number of trades", "int"),
    "win_rate": ("Percent profitable", "pct"),
    "winners": ("Winning trades", "int"),
    "losers": ("Losing trades", "int"),
    "even": ("Even trades", "int"),
    "avg_trade": ("Avg trade net profit (% of equity)", "pct2"),
    "avg_trade_usd": ("Avg trade net profit", "usd"),
    "avg_win": ("Avg winning trade", "pct2"),
    "avg_loss": ("Avg losing trade", "pct2"),
    "win_loss_ratio": ("Ratio avg win : avg loss", "ratio"),
    "largest_win": ("Largest winning trade", "pct2"),
    "largest_loss": ("Largest losing trade", "pct2"),
    "largest_win_share": ("Largest winner as % of gross profit", "pct"),
    "largest_loss_share": ("Largest loser as % of gross loss", "pct"),
    "max_consec_wins": ("Max consecutive winning trades", "int"),
    "max_consec_losses": ("Max consecutive losing trades", "int"),
    "avg_bars": ("Avg bars in total trades", "num"),
    "avg_bars_win": ("Avg bars in winning trades", "num"),
    "avg_bars_loss": ("Avg bars in losing trades", "num"),
    # account
    "capital": ("Starting capital", "usd"),
    "ending_equity": ("Ending equity", "usd"),
    "total_return": ("Total return", "pct"),
    "cagr": ("Annual rate of return (CAGR)", "pct"),
    "ann_vol": ("Annual volatility", "pct"),
    "sharpe": ("Sharpe ratio", "ratio"),
    "sortino": ("Sortino ratio", "ratio"),
    "max_drawdown": ("Max drawdown (close to close)", "pct"),
    "max_drawdown_usd": ("Max drawdown ($)", "usd"),
    "dd_peak": ("Max drawdown: peak", "date"),
    "dd_trough": ("Max drawdown: trough", "date"),
    "dd_recovery": ("Max drawdown: recovered", "date"),
    "longest_drawdown_days": ("Longest time below a previous high", "days"),
    "time_underwater": ("Time spent in drawdown", "pct"),
    "return_on_account": ("Return on account (net profit / max DD $)", "ratio"),
    "calmar": ("CAGR / max drawdown", "ratio"),
    "exposure": ("Time in market", "pct"),
    "trades_per_year": ("Trades per year", "num"),
    "costs_paid": ("Commission + slippage paid", "usd"),
    "funding_paid": ("Funding paid (negative = received)", "usd"),
    "cost_drag_per_year": ("Commission + slippage per year (% of equity)", "pct"),
    "funding_drag_per_year": ("Funding per year (% of equity)", "pct"),
    "stopped_at": ("Trading stopped (kill switch / liquidation)", "date"),
    "benchmark_cagr": ("Buy & hold CAGR", "pct"),
    "benchmark_sharpe": ("Buy & hold Sharpe", "ratio"),
    "benchmark_max_drawdown": ("Buy & hold max drawdown", "pct"),
    # monte carlo
    "risk_of_ruin": ("Risk of ruin", "pct"),
    "median_max_dd": ("Median max drawdown", "pct"),
    "median_return": ("Median return", "pct"),
    "return_dd": ("Return / drawdown", "ratio"),
    "prob_profit": ("Probability of profit", "pct"),
    "p05_return": ("5th percentile return", "pct"),
    "p95_max_dd": ("95th percentile max drawdown", "pct"),
}

TRADE_KEYS = list(FIELDS)[: list(FIELDS).index("capital")]
MC_KEYS = list(FIELDS)[list(FIELDS).index("risk_of_ruin") :]


@dataclass
class MonteCarlo:
    """Davey-style Monte Carlo on the closed-trade returns: one year of trades per run."""

    table: pd.DataFrame  # index: position-size multiplier, columns: MC_KEYS
    paths: np.ndarray  # sample of equity paths at 1x size, shape (runs, trades + 1)
    max_dd: np.ndarray  # max drawdown of every run at 1x size
    trades_per_run: int
    runs: int
    ruin: float

    def checks(
        self, limits: dict | None = None, size: float = 1.0
    ) -> list[tuple[str, float, str, bool]]:
        """Davey's three pass/fail tests at ``size`` (default: the size actually traded, 1x).

        ``limits`` overrides ``DAVEY_CRITERIA`` (the research pipeline passes its own).
        """

        row = self.table.loc[size]
        limits = limits or DAVEY_CRITERIA
        return [
            ("Risk of ruin", row["risk_of_ruin"], f"< {limits['risk_of_ruin']:.0%}",
             bool(row["risk_of_ruin"] < limits["risk_of_ruin"])),
            ("Median max drawdown", row["median_max_dd"], f"< {limits['median_max_dd']:.0%}",
             bool(row["median_max_dd"] < limits["median_max_dd"])),
            ("Return / drawdown", row["return_dd"], f"> {limits['return_dd']:.1f}",
             bool(row["return_dd"] > limits["return_dd"])),
        ]  # fmt: skip


@dataclass
class PerformanceReport:
    name: str
    meta: dict
    trades: pd.DataFrame  # closed and open trades with money columns
    trade_summary: pd.DataFrame  # index: TRADE_KEYS, columns: All / Long / Short
    account: dict
    annual: pd.DataFrame
    monthly: pd.DataFrame  # index: year, columns: Jan..Dec + Year
    equity: pd.Series  # in money
    drawdown: pd.Series  # fraction below the running peak
    monte_carlo: MonteCarlo | None
    benchmark: pd.Series | None = None  # buy & hold equity in money
    notes: list[str] = field(default_factory=list)


def max_streak(flags: np.ndarray) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def trade_list(result: BacktestResult, capital: float) -> pd.DataFrame:
    """Trades with ``equity_at_entry`` and ``pnl_usd`` added, in money of the compounded account.

    Money P&L is exact: closed and open trades add up to the account's net profit.
    """

    trades = result.trades.copy()
    if trades.empty:
        return trades.assign(equity_at_entry=pd.Series(dtype=float), pnl_usd=pd.Series(dtype=float))
    trades["return"] = trades["return"].astype(float)
    trades["open"] = trades["open"].astype(bool)
    equity_before = result.equity.shift(1, fill_value=1.0).to_numpy() * capital
    start = result.ledger.index.get_indexer(trades["entry"])
    trades["equity_at_entry"] = equity_before[start]
    if "pnl" in trades:  # the bracket simulator tracks money P&L per trade
        trades["pnl_usd"] = trades["pnl"].astype(float) * capital
    else:
        trades["pnl_usd"] = rebalanced_trade_money(result, trades, start, equity_before)
    return trades


def rebalanced_trade_money(
    result: BacktestResult, trades: pd.DataFrame, start: np.ndarray, equity_before: np.ndarray
) -> np.ndarray:
    """Money P&L of trades from the vectorized engine, bar by bar.

    Uses the same cost attribution as ``extract_trades``: the entry side and every resize belong
    to the trade, and the exit is charged on the first bar after it.
    """

    ledger = result.ledger
    position = ledger["position"].to_numpy()
    rate = result.costs.rate
    carry = (ledger["gross"] - ledger["funding_cost"]).to_numpy() * equity_before
    resize = np.abs(np.diff(position, prepend=0.0)) * rate * equity_before
    carry_sum = np.concatenate(([0.0], np.cumsum(carry)))
    resize_sum = np.concatenate(([0.0], np.cumsum(resize)))
    end = start + trades["bars"].to_numpy(dtype=int)  # one past the last bar held
    entry_cost = np.abs(position[start]) * rate * equity_before[start]
    after = np.minimum(end, len(position) - 1)
    exit_cost = np.where(
        trades["open"].to_numpy(), 0.0, np.abs(position[end - 1]) * rate * equity_before[after]
    )
    held = carry_sum[end] - carry_sum[start]
    resizing = resize_sum[end] - resize_sum[start + 1]
    return held - entry_cost - resizing - exit_cost


def side_stats(trades: pd.DataFrame) -> dict[str, float]:
    """TradeStation's trade analysis for one set of closed trades."""

    returns = trades["return"].to_numpy(dtype=float)
    money = trades["pnl_usd"].to_numpy(dtype=float)
    bars = trades["bars"].to_numpy(dtype=float)
    win, loss = returns > 0, returns < 0
    gross_profit, gross_loss = money[win].sum(), money[loss].sum()

    def mean(values: np.ndarray) -> float:
        return float(values.mean()) if len(values) else float("nan")

    if gross_loss < 0:
        profit_factor = gross_profit / -gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else float("nan")
    avg_win, avg_loss = mean(returns[win]), mean(returns[loss])
    return {
        "net_profit": float(money.sum()),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(profit_factor),
        "trades": len(returns),
        "win_rate": float(win.mean()) if len(returns) else float("nan"),
        "winners": int(win.sum()),
        "losers": int(loss.sum()),
        "even": int(len(returns) - win.sum() - loss.sum()),
        "avg_trade": mean(returns),
        "avg_trade_usd": mean(money),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": avg_win / -avg_loss if win.any() and loss.any() else float("nan"),
        "largest_win": float(returns.max()) if win.any() else float("nan"),
        "largest_loss": float(returns.min()) if loss.any() else float("nan"),
        "largest_win_share": float(money[win].max() / gross_profit) if win.any() else float("nan"),
        "largest_loss_share": float(money[loss].min() / gross_loss) if loss.any() else float("nan"),
        "max_consec_wins": max_streak(win),
        "max_consec_losses": max_streak(loss),
        "avg_bars": mean(bars),
        "avg_bars_win": mean(bars[win]),
        "avg_bars_loss": mean(bars[loss]),
    }


def trade_summary(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades[~trades["open"]] if len(trades) else trades
    sides = {
        "All trades": closed,
        "Long trades": closed[closed["direction"] > 0] if len(closed) else closed,
        "Short trades": closed[closed["direction"] < 0] if len(closed) else closed,
    }
    return pd.DataFrame({name: side_stats(part) for name, part in sides.items()}).loc[TRADE_KEYS]


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fraction below the running peak; the starting capital (1.0) counts as the first peak."""

    return equity / equity.cummax().clip(lower=1.0) - 1.0


def drawdown_stats(equity: pd.Series, capital: float) -> dict:
    peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / peak - 1.0
    empty = {
        "max_drawdown": 0.0,
        "max_drawdown_usd": 0.0,
        "dd_peak": pd.NaT,
        "dd_trough": pd.NaT,
        "dd_recovery": pd.NaT,
        "longest_drawdown_days": 0.0,
        "time_underwater": 0.0,
    }
    if equity.empty or drawdown.min() >= 0:
        return empty

    trough = drawdown.idxmin()
    peak_value = peak.loc[trough]
    before = equity.loc[:trough]
    at_peak = before[before >= peak_value]
    after = equity.loc[trough:]
    recovered = after[after >= peak_value]

    # Time since the last high, measured on every bar spent below it.
    days = ((equity.index - equity.index[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    at_high = (drawdown >= 0).to_numpy()
    last_high = np.maximum.accumulate(np.where(at_high, days, 0.0))

    return {
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_usd": float(((peak - equity) * capital).max()),
        "dd_peak": at_peak.index[-1] if len(at_peak) else equity.index[0],
        "dd_trough": trough,
        "dd_recovery": recovered.index[0] if len(recovered) else pd.NaT,
        "longest_drawdown_days": float((days - last_high)[~at_high].max()),
        "time_underwater": float((~at_high).mean()),
    }


def account_summary(
    result: BacktestResult, capital: float, benchmark: BacktestResult | None = None
) -> dict:
    stats = summarize(result)
    equity = result.equity
    ledger = result.ledger
    equity_before = equity.shift(1, fill_value=1.0)
    drawdown = drawdown_stats(equity, capital)
    net_profit = float((equity.iloc[-1] - 1.0) * capital)
    max_dd = drawdown["max_drawdown"]
    account = {
        "capital": capital,
        "ending_equity": float(equity.iloc[-1] * capital),
        "net_profit": net_profit,
        "total_return": stats["total_return"],
        "cagr": stats["cagr"],
        "ann_vol": stats["ann_vol"],
        "sharpe": stats["sharpe"],
        "sortino": stats["sortino"],
        **drawdown,
        "return_on_account": (
            net_profit / drawdown["max_drawdown_usd"] if drawdown["max_drawdown_usd"] else np.nan
        ),
        "calmar": stats["cagr"] / -max_dd if max_dd < 0 else np.nan,
        "exposure": stats["exposure"],
        "trades_per_year": stats["trades_per_year"],
        "costs_paid": float((ledger["trading_cost"] * equity_before).sum() * capital),
        "funding_paid": float((ledger["funding_cost"] * equity_before).sum() * capital),
        "cost_drag_per_year": stats["cost_drag_per_year"],
        "funding_drag_per_year": stats["funding_drag_per_year"],
        "stopped_at": result.meta.get("killed_at", result.meta.get("ruined_at", pd.NaT)),
    }
    if benchmark is not None:
        bench = summarize(benchmark)
        account["benchmark_cagr"] = bench["cagr"]
        account["benchmark_sharpe"] = bench["sharpe"]
        account["benchmark_max_drawdown"] = float(drawdown_series(benchmark.equity).min())
    return account


def annual_analysis(equity: pd.Series, trades: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Return, drawdown and trades for each calendar year (trades counted by exit date)."""

    closed = trades[~trades["open"]] if len(trades) else trades
    rows = {}
    start = 1.0
    for year, path in equity.groupby(equity.index.year):
        values = path.to_numpy()
        peak = np.maximum.accumulate(np.concatenate(([start], values)))[1:]
        in_year = closed[closed["exit"].dt.year == year] if len(closed) else closed
        rows[year] = {
            "return": values[-1] / start - 1.0,
            "net_profit": (values[-1] - start) * capital,
            "max_drawdown": min(float((values / peak - 1.0).min()), 0.0),
            "trades": len(in_year),
            "win_rate": float((in_year["return"] > 0).mean()) if len(in_year) else np.nan,
        }
        start = values[-1]
    frame = pd.DataFrame(rows).T
    frame.index.name = "year"
    return frame.astype({"trades": int})


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_returns(equity: pd.Series, annual: pd.DataFrame) -> pd.DataFrame:
    month_end = equity.groupby([equity.index.year, equity.index.month]).last()
    returns = month_end / month_end.shift(1, fill_value=1.0) - 1.0
    table = returns.unstack()
    table = table.reindex(columns=range(1, 13))
    table.columns = MONTHS
    table.index.name = "year"
    table["Year"] = annual["return"]
    return table


def monte_carlo(
    returns: np.ndarray,
    trades_per_run: int,
    runs: int = 2500,
    ruin: float = 0.5,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    seed: int = 0,
    keep_paths: int = 200,
) -> MonteCarlo:
    """Resample ``trades_per_run`` trades with replacement, ``runs`` times (Davey's method).

    A run is ruined when equity falls to ``1 - ruin`` of the start; it then stops trading.
    Each multiplier rescales every trade (the analogue of Davey's starting-equity table for a
    fraction-of-equity account): 2.0 means trading twice the size.
    """

    returns = np.asarray(returns, dtype=float)
    multipliers = tuple(sorted({*multipliers, 1.0}))
    rng = np.random.default_rng(seed)
    draws = rng.choice(returns, size=(runs, max(trades_per_run, 1)))
    floor = 1.0 - ruin
    rows, paths, drawdowns = {}, None, None
    for size in multipliers:
        scaled = np.maximum(draws * size, -1.0)
        equity = np.cumprod(1.0 + scaled, axis=1)
        equity = np.concatenate((np.ones((runs, 1)), equity), axis=1)
        ruined = np.maximum.accumulate(equity <= floor, axis=1)
        first = ruined.argmax(axis=1)
        frozen = equity[np.arange(runs), first][:, None]
        equity = np.where(ruined, frozen, equity)
        max_dd = (1.0 - equity / np.maximum.accumulate(equity, axis=1)).max(axis=1)
        final = equity[:, -1] - 1.0
        median_dd, median_return = float(np.median(max_dd)), float(np.median(final))
        rows[size] = {
            "risk_of_ruin": float(ruined[:, -1].mean()),
            "median_max_dd": median_dd,
            "median_return": median_return,
            "return_dd": median_return / median_dd if median_dd > 0 else float("inf"),
            "prob_profit": float((final > 0).mean()),
            "p05_return": float(np.quantile(final, 0.05)),
            "p95_max_dd": float(np.quantile(max_dd, 0.95)),
        }
        if size == 1.0:
            paths, drawdowns = equity[:keep_paths], max_dd
    table = pd.DataFrame(rows).T[MC_KEYS]
    table.index.name = "size"
    return MonteCarlo(table, paths, drawdowns, max(trades_per_run, 1), runs, ruin)


def build_report(
    result: BacktestResult,
    capital: float = 10_000.0,
    benchmark: BacktestResult | None = None,
    mc_runs: int = 2500,
    ruin: float = 0.5,
    seed: int = 0,
) -> PerformanceReport:
    trades = trade_list(result, capital)
    account = account_summary(result, capital, benchmark)
    annual = annual_analysis(result.equity, trades, capital)
    closed = trades[~trades["open"]] if len(trades) else trades

    notes = []
    mc = None
    if len(closed) >= MIN_MC_TRADES:
        # Trades per year while the strategy was still trading (it may have been stopped).
        index = result.ledger.index
        stopped = account["stopped_at"]
        last = index[-1] if pd.isna(stopped) else stopped
        active_years = max((last - index[0]) / YEAR, 1 / 365.25)
        per_year = max(round(len(closed) / active_years), 1)
        mc = monte_carlo(closed["return"].to_numpy(), per_year, mc_runs, ruin, seed=seed)
    else:
        notes.append(f"Monte Carlo skipped: {len(closed)} closed trades (need {MIN_MC_TRADES}).")
    stopped = account["stopped_at"]
    if not pd.isna(stopped):
        flat = float((result.ledger.index > stopped).mean())
        notes.append(
            f"Trading stopped on {stopped:%Y-%m-%d} (kill switch or liquidation): the last "
            f"{flat:.0%} of the period is flat, and the Monte Carlo only sees trades before then, "
            "so it understates the risk."
        )
    if len(trades) and trades["open"].any():
        open_trade = trades[trades["open"]].iloc[-1]
        notes.append(
            f"Open trade at the end of the data: {open_trade['pnl_usd']:+,.0f} "
            f"({open_trade['return']:+.2%}), not included in the trade statistics."
        )

    meta = dict(result.meta)
    index = result.ledger.index
    meta.update(start=index[0], end=index[-1], bars=len(index), costs=result.costs)
    return PerformanceReport(
        name=str(meta.get("symbol", "backtest")),
        meta=meta,
        trades=trades,
        trade_summary=trade_summary(trades),
        account=account,
        annual=annual,
        monthly=monthly_returns(result.equity, annual),
        equity=result.equity * capital,
        drawdown=drawdown_series(result.equity),
        monte_carlo=mc,
        benchmark=None if benchmark is None else benchmark.equity * capital,
        notes=notes,
    )
