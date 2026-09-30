"""Performance statistics for backtest results. Crypto trades 24/7, so a year is 365.25 days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult

YEAR = pd.Timedelta(days=365.25)


def periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return float("nan")
    step = pd.Series(index).diff().median()
    return YEAR / step


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def sharpe_ratio(returns: pd.Series, ppy: float) -> float:
    std = returns.std()
    return 0.0 if std == 0 or np.isnan(std) else float(returns.mean() / std * np.sqrt(ppy))


def summarize(result: BacktestResult) -> dict[str, float]:
    ledger = result.ledger
    returns = result.returns
    equity = result.equity
    ppy = periods_per_year(ledger.index)
    years = len(ledger) / ppy

    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    downside = returns[returns < 0].std()
    sortino = 0.0 if not downside > 0 else float(returns.mean() / downside * np.sqrt(ppy))
    mdd = max_drawdown(equity)

    trades = result.trades
    closed = trades[~trades["open"]] if len(trades) else trades
    wins = closed["return"][closed["return"] > 0]
    losses = closed["return"][closed["return"] < 0]

    return {
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": float(returns.std() * np.sqrt(ppy)),
        "sharpe": sharpe_ratio(returns, ppy),
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": 0.0 if mdd == 0 else cagr / -mdd,
        "trades": float(len(closed)),
        "trades_per_year": len(closed) / years,
        "win_rate": float(len(wins) / len(closed)) if len(closed) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        "avg_trade": float(closed["return"].mean()) if len(closed) else 0.0,
        "avg_bars_held": float(closed["bars"].mean()) if len(closed) else 0.0,
        "exposure": float((ledger["position"] != 0).mean()),
        "turnover_per_year": float(ledger["turnover"].sum() / years),
        "cost_drag_per_year": float(ledger["trading_cost"].sum() / years),
        "funding_drag_per_year": float(ledger["funding_cost"].sum() / years),
    }
