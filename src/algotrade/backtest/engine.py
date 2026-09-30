"""Vectorized single-instrument backtester with fees, slippage and funding.

Timing convention (the one thing a backtester must get right):

* ``target[t]`` is the exposure a strategy wants, computed from data up to and including the
  close of bar ``t``.
* It is executed at that close and held through bar ``t + 1``, so ``position[t] = target[t-1]``.
* Bar ``t`` earns ``position[t] * (close[t] / close[t-1] - 1)``.
* Changing exposure costs ``|position[t] - position[t-1]| * cost_rate``, charged to bar ``t``.
* Funding settled inside bar ``t`` is paid on ``position[t]`` (longs pay positive funding).

Exposure is a signed fraction of equity (1.0 = fully long 1x), rebalanced every bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class BacktestResult:
    """Per-bar ledger plus derived trade list for one instrument."""

    ledger: pd.DataFrame
    trades: pd.DataFrame
    costs: CostModel
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.ledger["net"]

    @property
    def equity(self) -> pd.Series:
        return self.ledger["equity"]


def run_backtest(
    bars: pd.DataFrame,
    target: pd.Series,
    costs: CostModel | None = None,
    max_leverage: float = 1.0,
) -> BacktestResult:
    """Simulate holding ``target`` exposure on ``bars`` (indexed by bar-open time)."""

    costs = costs or CostModel()
    target = target.reindex(bars.index).astype("float64").fillna(0.0)
    target = target.clip(-max_leverage, max_leverage)

    close = bars["close"].to_numpy(dtype="float64")
    tgt = target.to_numpy().copy()
    position = np.concatenate(([0.0], tgt[:-1]))
    bar_return = np.zeros_like(close)
    bar_return[1:] = close[1:] / close[:-1] - 1.0

    turnover = np.abs(np.diff(position, prepend=0.0))
    gross = position * bar_return
    trading_cost = turnover * costs.rate
    if costs.include_funding and "funding_rate" in bars:
        funding_cost = position * bars["funding_rate"].to_numpy(dtype="float64")
    else:
        funding_cost = np.zeros_like(close)
    net = gross - trading_cost - funding_cost

    meta = dict(bars.attrs)
    ruin = np.flatnonzero(net <= -1.0)
    if ruin.size:
        # Equity hit zero: the account is liquidated and stays flat. Without this, a loss
        # beyond -100% flips equity negative and compounding produces nonsense.
        k = ruin[0]
        net[k] = -1.0
        tgt[k:] = 0.0
        for column in (net, position, turnover, gross, trading_cost, funding_cost):
            column[k + 1 :] = 0.0
        meta["ruined_at"] = bars.index[k]

    ledger = pd.DataFrame(
        {
            "close": close,
            "target": tgt,
            "position": position,
            "turnover": turnover,
            "gross": gross,
            "trading_cost": trading_cost,
            "funding_cost": funding_cost,
            "net": net,
            "equity": np.cumprod(1.0 + net),
        },
        index=bars.index,
    )
    return BacktestResult(
        ledger=ledger,
        trades=extract_trades(ledger, costs.rate),
        costs=costs,
        meta=meta,
    )


TRADE_COLUMNS = ["entry", "exit", "direction", "bars", "return", "open"]


def extract_trades(ledger: pd.DataFrame, cost_rate: float) -> pd.DataFrame:
    """Group consecutive bars with the same position sign into trades.

    A trade's return is its additive P&L in equity fractions: bar P&L while held, minus the cost
    of entering, resizing and exiting. The entry cost of a reversal is attributed to the new
    trade and the exit cost to the old one, so both sides are charged fairly.
    """

    position = ledger["position"].to_numpy()
    sign = np.sign(position)
    if not sign.any():
        return pd.DataFrame(columns=TRADE_COLUMNS)

    run_id = np.concatenate(([0], np.cumsum(sign[1:] != sign[:-1])))
    pnl = (ledger["gross"] - ledger["funding_cost"]).to_numpy()
    resize = np.abs(np.diff(position, prepend=0.0)) * cost_rate

    frame = pd.DataFrame(
        {"run": run_id, "sign": sign, "pnl": pnl, "resize": resize, "pos": position}
    )
    frame = frame[frame["sign"] != 0]
    first_rows = frame.groupby("run").head(1).index
    # The first bar's turnover may include closing an opposite position; charge only this side.
    frame.loc[first_rows, "resize"] = np.abs(frame.loc[first_rows, "pos"]) * cost_rate

    grouped = frame.groupby("run")
    first = grouped.head(1)
    last = grouped.tail(1)
    is_open = (last["run"] == run_id[-1]).to_numpy()
    exit_cost = np.where(is_open, 0.0, np.abs(last["pos"].to_numpy()) * cost_rate)

    index = ledger.index
    step = index[1] - index[0] if len(index) > 1 else pd.Timedelta(0)
    trades = pd.DataFrame(
        {
            "entry": index[first.index],
            "exit": index[last.index] + step,
            "direction": first["sign"].astype(int).to_numpy(),
            "bars": grouped.size().to_numpy(),
            "return": grouped["pnl"].sum().to_numpy()
            - grouped["resize"].sum().to_numpy()
            - exit_cost,
            "open": is_open,
        }
    )
    return trades
