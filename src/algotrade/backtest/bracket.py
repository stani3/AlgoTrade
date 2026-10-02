"""Bar-by-bar simulator for bracket trades: market entry, fixed stop and profit target.

Use this instead of the vectorized engine when exits happen *inside* a bar (stops, targets) or
when rules depend on how earlier trades went (cooldowns, kill switches).

Fill rules, in the order they are applied on every bar ``i``:

1. A kill switch tripped at the previous close exits the open position at ``open[i]``.
2. An entry signalled at the previous close fills at ``open[i]``. Quantity is fixed at
   ``leverage * equity / fill`` until exit. Stop and target sit ``stop_dist`` / ``target_dist``
   (read on the signal bar) away from the fill price and never move.
3. Exits, checked on every bar the trade is open including the entry bar:
   a bar that opens beyond the stop or the target exits at the open (gap); otherwise if the
   bar's range touches both levels the stop is assumed first (conservative); otherwise the
   level that was touched fills exactly.
4. The position is marked to the close and pays funding if still open at the close.
5. Kill switch: if equity <= (1 - kill_drawdown) * peak equity, trading stops for good.
6. A new signal at the close is accepted only when flat, not killed, and at least
   ``cooldown_win`` (last trade made money after costs) or ``cooldown_loss`` (otherwise) bars
   after the last exit bar. A zero-P&L trade counts as a loser. Signals while in a trade are
   ignored (no reversal, no pyramiding); if long and short fire together, the long is taken.

Costs: every fill pays ``fee_rate`` on its notional. Market and stop fills also slip by
``slip_rate`` against the trade; target fills are limit orders and do not slip.
If equity reaches zero the account is liquidated at that close and stays flat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from .costs import CostModel
from .engine import TRADE_COLUMNS, BacktestResult

EXIT_REASONS = ("stop", "target", "kill", "liquidated", "open")
STOP, TARGET, KILL, LIQUIDATED, OPEN = range(5)


@dataclass(frozen=True)
class BracketSignals:
    """Per-bar entry conditions and bracket distances, all known at that bar's close."""

    long: pd.Series
    short: pd.Series
    stop_dist: pd.Series
    target_dist: pd.Series


@njit(cache=True)
def _simulate(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    funding: np.ndarray,
    long_sig: np.ndarray,
    short_sig: np.ndarray,
    stop_dist: np.ndarray,
    target_dist: np.ndarray,
    cooldown_win: int,
    cooldown_loss: int,
    kill_drawdown: float,
    fee_rate: float,
    slip_rate: float,
    leverage: float,
):
    n = close.shape[0]
    gross = np.zeros(n)
    cost = np.zeros(n)
    fund = np.zeros(n)
    traded = np.zeros(n)
    equity_out = np.zeros(n)
    qty_out = np.zeros(n)

    t_entry = np.zeros(n, dtype=np.int64)
    t_exit = np.zeros(n, dtype=np.int64)
    t_dir = np.zeros(n, dtype=np.int64)
    t_qty = np.zeros(n)
    t_entry_px = np.zeros(n)
    t_exit_px = np.zeros(n)
    t_reason = np.zeros(n, dtype=np.int64)
    t_pnl = np.zeros(n)
    t_equity = np.zeros(n)
    count = 0

    equity = 1.0
    peak = 1.0
    killed = False
    killed_bar = -1
    kill_pending = False
    pos = 0
    qty = 0.0
    mark = 0.0
    stop_lvl = 0.0
    target_lvl = 0.0
    pending = 0
    pending_stop = 0.0
    pending_target = 0.0
    last_exit = -(10**12)
    cooldown = 0
    # running totals for the open trade
    entry_bar = 0
    entry_px = 0.0
    entry_fee = 0.0
    entry_equity = 0.0
    trade_funding = 0.0

    for i in range(n):
        g = 0.0
        c = 0.0
        f = 0.0
        o = open_[i]
        exit_raw = np.nan
        exit_px = 0.0
        reason = -1

        # 1. kill-switch exit at the open
        if kill_pending and pos != 0:
            exit_raw = o
            exit_px = o * (1.0 - pos * slip_rate)
            reason = KILL
        kill_pending = False

        # 2. pending entry fills at the open
        if pending != 0 and reason < 0:
            pos = pending
            entry_px = o * (1.0 + pos * slip_rate)
            qty = leverage * equity / entry_px
            entry_fee = qty * entry_px * fee_rate
            c += entry_fee + qty * o * slip_rate
            traded[i] += qty * entry_px
            mark = o
            stop_lvl = entry_px - pos * pending_stop
            target_lvl = entry_px + pos * pending_target
            entry_bar = i
            entry_equity = equity
            trade_funding = 0.0
        pending = 0

        # 3. stop / target
        if pos != 0 and reason < 0:
            if pos > 0:
                if o <= stop_lvl:
                    exit_raw, exit_px, reason = o, o * (1.0 - slip_rate), STOP
                elif o >= target_lvl:
                    exit_raw, exit_px, reason = o, o, TARGET
                elif low[i] <= stop_lvl:
                    exit_raw, exit_px, reason = stop_lvl, stop_lvl * (1.0 - slip_rate), STOP
                elif high[i] >= target_lvl:
                    exit_raw, exit_px, reason = target_lvl, target_lvl, TARGET
            else:
                if o >= stop_lvl:
                    exit_raw, exit_px, reason = o, o * (1.0 + slip_rate), STOP
                elif o <= target_lvl:
                    exit_raw, exit_px, reason = o, o, TARGET
                elif high[i] >= stop_lvl:
                    exit_raw, exit_px, reason = stop_lvl, stop_lvl * (1.0 + slip_rate), STOP
                elif low[i] <= target_lvl:
                    exit_raw, exit_px, reason = target_lvl, target_lvl, TARGET

        if reason >= 0:
            exit_fee = qty * exit_px * fee_rate
            g += pos * qty * (exit_raw - mark)
            c += exit_fee + qty * abs(exit_raw - exit_px)
            traded[i] += qty * exit_px
            pnl = pos * qty * (exit_px - entry_px) - entry_fee - exit_fee - trade_funding
            t_entry[count] = entry_bar
            t_exit[count] = i
            t_dir[count] = pos
            t_qty[count] = qty
            t_entry_px[count] = entry_px
            t_exit_px[count] = exit_px
            t_reason[count] = reason
            t_pnl[count] = pnl
            t_equity[count] = entry_equity
            count += 1
            cooldown = cooldown_win if pnl > 0 else cooldown_loss
            last_exit = i
            pos = 0
            qty = 0.0

        # 4. mark to the close, funding on positions held at the close
        if pos != 0:
            g += pos * qty * (close[i] - mark)
            mark = close[i]
            f = pos * qty * close[i] * funding[i]
            trade_funding += f

        new_equity = equity + g - c - f
        if new_equity <= 0.0:
            new_equity = 0.0
            if reason >= 0:
                # A gap blew through the stop and took more than the account: the trade's
                # loss is capped at the equity it started with.
                t_pnl[count - 1] = -t_equity[count - 1]
                t_reason[count - 1] = LIQUIDATED
            if pos != 0:
                t_entry[count] = entry_bar
                t_exit[count] = i
                t_dir[count] = pos
                t_qty[count] = qty
                t_entry_px[count] = entry_px
                t_exit_px[count] = close[i]
                t_reason[count] = LIQUIDATED
                t_pnl[count] = -entry_equity
                t_equity[count] = entry_equity
                count += 1
                pos = 0
                qty = 0.0
                last_exit = i
            if not killed:
                killed = True
                killed_bar = i

        gross[i] = g
        cost[i] = c
        fund[i] = f
        equity = new_equity
        equity_out[i] = equity
        qty_out[i] = pos * qty

        # 5. kill switch
        peak = max(peak, equity)
        if not killed and equity <= (1.0 - kill_drawdown) * peak:
            killed = True
            killed_bar = i
            kill_pending = pos != 0

        # 6. new signals
        if pos == 0 and not killed and i - last_exit >= cooldown:
            if long_sig[i] and stop_dist[i] > 0.0 and target_dist[i] > 0.0:
                pending = 1
            elif short_sig[i] and stop_dist[i] > 0.0 and target_dist[i] > 0.0:
                pending = -1
            pending_stop = stop_dist[i]
            pending_target = target_dist[i]

    if pos != 0:
        pnl = pos * qty * (close[n - 1] - entry_px) - entry_fee - trade_funding
        t_entry[count] = entry_bar
        t_exit[count] = n - 1
        t_dir[count] = pos
        t_qty[count] = qty
        t_entry_px[count] = entry_px
        t_exit_px[count] = close[n - 1]
        t_reason[count] = OPEN
        t_pnl[count] = pnl
        t_equity[count] = entry_equity
        count += 1

    trades = (
        t_entry[:count],
        t_exit[:count],
        t_dir[:count],
        t_qty[:count],
        t_entry_px[:count],
        t_exit_px[:count],
        t_reason[:count],
        t_pnl[:count],
        t_equity[:count],
    )
    return gross, cost, fund, traded, equity_out, qty_out, trades, killed_bar


def simulate_bracket(
    bars: pd.DataFrame,
    signals: BracketSignals,
    costs: CostModel | None = None,
    leverage: float = 1.0,
    cooldown_win: int = 0,
    cooldown_loss: int = 0,
    kill_drawdown: float = 1.0,
) -> BacktestResult:
    """Run the bracket simulation and return a ledger compatible with ``run_backtest``."""

    costs = costs or CostModel()

    def floats(values: pd.Series | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype="float64")

    def flags(condition: pd.Series) -> np.ndarray:
        return condition.reindex(bars.index).fillna(False).to_numpy(dtype=bool)

    def distances(values: pd.Series) -> np.ndarray:
        return values.reindex(bars.index).fillna(0.0).to_numpy(dtype="float64")

    has_funding = costs.include_funding and "funding_rate" in bars
    funding = floats(bars["funding_rate"]) if has_funding else np.zeros(len(bars))
    gross, cost, fund, traded, equity, qty, trades, killed_bar = _simulate(
        floats(bars["open"]),
        floats(bars["high"]),
        floats(bars["low"]),
        floats(bars["close"]),
        funding,
        flags(signals.long),
        flags(signals.short),
        distances(signals.stop_dist),
        distances(signals.target_dist),
        int(cooldown_win),
        int(cooldown_loss),
        float(kill_drawdown),
        costs.fee_bps / 10_000,
        costs.slippage_bps / 10_000,
        float(leverage),
    )

    close = floats(bars["close"])
    prev_equity = np.concatenate(([1.0], equity[:-1]))
    safe_prev = np.where(prev_equity > 0, prev_equity, np.inf)
    safe_now = np.where(equity > 0, equity, np.inf)
    position = qty * close / safe_now
    ledger = pd.DataFrame(
        {
            "close": close,
            "target": position,
            "position": position,
            "turnover": traded / safe_prev,
            "gross": gross / safe_prev,
            "trading_cost": cost / safe_prev,
            "funding_cost": fund / safe_prev,
            "net": (equity - prev_equity) / safe_prev,
            "equity": equity,
        },
        index=bars.index,
    )

    meta = dict(bars.attrs)
    if killed_bar >= 0:
        meta["killed_at"] = bars.index[killed_bar]
    return BacktestResult(
        ledger=ledger, trades=_trade_frame(bars.index, trades), costs=costs, meta=meta
    )


def _trade_frame(index: pd.DatetimeIndex, trades: tuple) -> pd.DataFrame:
    entry, exit_, direction, qty, entry_px, exit_px, reason, pnl, entry_equity = trades
    columns = [*TRADE_COLUMNS, "qty", "entry_price", "exit_price", "exit_reason", "pnl"]
    if len(entry) == 0:
        return pd.DataFrame(columns=columns)
    # A kill exit fills at the open, so that bar is not held; every other exit is intrabar.
    held = exit_ - entry + np.where(reason == KILL, 0, 1)
    return pd.DataFrame(
        {
            "entry": index[entry],
            "exit": index[exit_],
            "direction": direction,
            "bars": held,
            "return": pnl / entry_equity,
            "open": reason == OPEN,
            "qty": qty,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "exit_reason": [EXIT_REASONS[r] for r in reason],
            "pnl": pnl,
        },
        columns=columns,
    )
