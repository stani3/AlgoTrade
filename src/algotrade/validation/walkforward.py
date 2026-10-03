"""Rolling walk-forward analysis over the development period.

Each window optimises the pre-registered grid on ``in_sample_years`` of data (plateau choice, as
in feasibility) and then trades the chosen parameters on the next ``out_of_sample_months``.
The out-of-sample pieces are stitched into one track record per symbol: that is what the
strategy would have earned had it been re-optimised on schedule, never seeing its future.

Every out-of-sample window starts flat. Indicators are computed on all the history before the
window, so nothing needs warming up, but positions and kill-switch state are not carried in. A
position still open when a window ends is closed at that close and pays the exit cost.

Walk-forward efficiency (Pardo) compares the annualised out-of-sample return with the
annualised in-sample return of the chosen parameters: a robust system keeps at least about half.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from algotrade.backtest.bracket import BracketSignals, simulate_bracket
from algotrade.backtest.costs import CostModel, costs_for
from algotrade.backtest.engine import BacktestResult, extract_trades, run_backtest
from algotrade.backtest.metrics import YEAR, periods_per_year, sharpe_ratio
from algotrade.parallel import Pool, run
from algotrade.research.dedup import grid_specs, set_path
from algotrade.strategies import BracketStrategy, from_spec

from .optimize import assemble_board, choose, grid_symbol


@dataclass(frozen=True)
class Window:
    in_sample_start: pd.Timestamp
    out_of_sample_start: pd.Timestamp
    out_of_sample_end: pd.Timestamp


def make_windows(
    first: pd.Timestamp,
    end: pd.Timestamp,
    in_sample_years: float,
    out_of_sample_months: int,
    min_days: int = 28,
) -> list[Window]:
    """Back-to-back out-of-sample windows from ``first + in_sample_years`` (rounded up to a
    month start) until ``end``; a final window shorter than ``min_days`` is dropped."""

    in_sample = pd.DateOffset(months=max(round(in_sample_years * 12), 1))
    start = (first + in_sample).ceil("D")
    if start != start + pd.offsets.MonthBegin(0):
        start = start + pd.offsets.MonthBegin(1)
    windows = []
    while start < end:
        stop = min(start + pd.DateOffset(months=out_of_sample_months), end)
        if (stop - start).days >= min_days:
            windows.append(Window(start - in_sample, start, stop))
        start = stop
    return windows


def out_of_sample(
    spec: dict, history: pd.DataFrame, start: pd.Timestamp, costs: CostModel
) -> BacktestResult:
    """Trade ``spec`` from ``start`` on, flat at ``start``, signals computed on all ``history``."""

    strategy = from_spec(spec)
    part = history[history.index >= start]
    if isinstance(strategy, BracketStrategy):
        signals = strategy.signals(history)
        sliced = BracketSignals(
            *(
                s.reindex(part.index)
                for s in (signals.long, signals.short, signals.stop_dist, signals.target_dist)
            )
        )
        return simulate_bracket(
            part, sliced, costs,
            cooldown_win=strategy.cooldown_win, cooldown_loss=strategy.cooldown_loss,
            kill_drawdown=strategy.kill_drawdown, max_bars=getattr(strategy, "max_bars", 0),
        )  # fmt: skip
    return run_backtest(part, strategy.target_position(history).reindex(part.index), costs)


def stitch(pieces: list[BacktestResult], costs: CostModel) -> BacktestResult:
    """Join consecutive out-of-sample results into one compounded track record.

    A position still open at the end of a piece is closed at that close: the exit (fee and
    slippage on the position) is charged on the next piece's first bar, exactly as one
    continuous engine run that went flat there would charge it, and an open bracket trade is
    marked closed with that cost in its P&L. The last piece's open position stays open.
    """

    ledgers, bracket = [], []
    carried = 0.0
    slip = costs.slippage_bps / 10_000
    for number, piece in enumerate(pieces):
        ledger = piece.ledger.copy()
        if carried:
            first = ledger.index[0]
            ledger.at[first, "turnover"] += carried
            ledger.at[first, "trading_cost"] += carried * costs.rate
            ledger.at[first, "net"] -= carried * costs.rate
        last = number == len(pieces) - 1
        held = abs(float(ledger["position"].iloc[-1]))
        if "exit_reason" in piece.trades:
            trades = piece.trades.copy()
            equity = piece.equity.to_numpy()
            before = pd.Series(np.concatenate(([1.0], equity[:-1])), index=piece.ledger.index)
            entry_equity = before.reindex(pd.DatetimeIndex(trades["entry"])).to_numpy()
            open_ = trades["open"].astype(bool).to_numpy()
            if not last and open_.any():
                trades.loc[open_, "pnl"] -= held * costs.rate * equity[-1]
                trades.loc[open_, "exit_price"] *= 1 - trades.loc[open_, "direction"] * slip
                trades.loc[open_, "open"] = False
                trades.loc[open_, "exit_reason"] = "window end"
            trades["return"] = trades["pnl"] / entry_equity
            bracket.append(trades)
        ledgers.append(ledger)
        carried = 0.0 if last else held
    ledger = pd.concat(ledgers)
    ledger["equity"] = np.cumprod(1.0 + ledger["net"].to_numpy())
    if bracket:
        trades = pd.concat(bracket, ignore_index=True)
        before = ledger["equity"].shift(1, fill_value=1.0)
        trades["pnl"] = (
            trades["return"] * before.reindex(pd.DatetimeIndex(trades["entry"])).to_numpy()
        )
    else:
        trades = extract_trades(ledger, costs.rate)
    return BacktestResult(ledger=ledger, trades=trades, costs=costs, meta=dict(pieces[0].meta))


def _annualised(total_return: float, days: float) -> float:
    if days <= 0 or total_return <= -1:
        return -1.0
    return float((1.0 + total_return) ** (YEAR.days / days) - 1.0)


@dataclass
class WalkForward:
    windows: pd.DataFrame
    results: dict[str, BacktestResult]

    @property
    def sharpes(self) -> pd.Series:
        return pd.Series(
            {
                s: sharpe_ratio(r.returns, periods_per_year(r.ledger.index))
                for s, r in self.results.items()
            }
        )

    @property
    def oos_median_sharpe(self) -> float:
        return float(self.sharpes.median()) if len(self.results) else 0.0

    @property
    def profitable_windows(self) -> float:
        return float((self.windows["oos_median_return"] > 0).mean()) if len(self.windows) else 0.0

    @property
    def efficiency(self) -> float:
        """Annualised out-of-sample return over annualised in-sample return (0 if IS <= 0)."""

        if not len(self.windows):
            return 0.0
        in_sample = float(self.windows["is_median_cagr"].mean())
        out = float(self.windows["oos_annualised"].mean())
        return out / in_sample if in_sample > 0 else 0.0


@dataclass(frozen=True)
class InSampleTask:
    bars: pd.DataFrame
    costs: CostModel
    specs: list[dict]
    windows: list[Window]
    min_in_sample_share: float


def in_sample_symbol(task: InSampleTask) -> list[list[tuple[dict, bool]] | None]:
    """Per window, every grid configuration on this symbol's in-sample bars, or None when the
    symbol has too little history in that window to take part."""

    out = []
    bars = task.bars
    for window in task.windows:
        part = bars[
            (bars.index >= window.in_sample_start) & (bars.index < window.out_of_sample_start)
        ]
        span = window.out_of_sample_start - window.in_sample_start
        if len(part) and (part.index[-1] - part.index[0]) >= span * task.min_in_sample_share:
            out.append(grid_symbol(task.specs, part, task.costs))
        else:
            out.append(None)
    return out


@dataclass(frozen=True)
class OutOfSampleTask:
    bars: pd.DataFrame
    costs: CostModel
    plan: list[tuple[Window, dict]]  # each window with the parameters chosen for it


def out_of_sample_symbol(
    task: OutOfSampleTask,
) -> tuple[list[tuple[float, float] | None], BacktestResult | None]:
    """Per window, this symbol's out-of-sample (return, Sharpe) or None if it has no bars there;
    and the stitched out-of-sample record."""

    per_window, pieces = [], []
    for window, spec in task.plan:
        history = task.bars[task.bars.index < window.out_of_sample_end]
        if not (history.index >= window.out_of_sample_start).any():
            per_window.append(None)
            continue
        result = out_of_sample(spec, history, window.out_of_sample_start, task.costs)
        pieces.append(result)
        per_window.append(
            (
                float(result.equity.iloc[-1] - 1.0),
                sharpe_ratio(result.returns, periods_per_year(result.ledger.index)),
            )
        )
    return per_window, stitch(pieces, task.costs) if pieces else None


def walk_forward(
    base: dict,
    grid: dict[str, list],
    universe: dict[str, pd.DataFrame],
    costs: CostModel | Mapping[str, CostModel],
    end: pd.Timestamp,
    in_sample_years: float,
    out_of_sample_months: int,
    min_in_sample_share: float = 0.5,
    pool: Pool | None = None,
) -> WalkForward:
    """Re-optimise on each in-sample window, trade the next out-of-sample one, stitch.

    Symbols run in parallel twice: every window's grid in-sample, then (once the parent has
    chosen each window's parameters across symbols) every window out-of-sample.
    """

    keys = list(grid)
    first = min(bars.index[0] for bars in universe.values())
    windows = make_windows(first, end, in_sample_years, out_of_sample_months)
    specs = grid_specs(base, grid)
    in_tasks = [
        InSampleTask(bars, costs_for(costs, symbol), specs, windows, min_in_sample_share)
        for symbol, bars in universe.items()
    ]
    cells = dict(zip(universe, run(in_sample_symbol, in_tasks, pool), strict=True))

    plan, bests = [], []
    for number, window in enumerate(windows):
        in_sample = {s: c[number] for s, c in cells.items() if c[number] is not None}
        if not in_sample:
            continue
        best = choose(assemble_board(grid, specs, in_sample), keys)
        spec = base
        for key in keys:
            value = best[key]
            spec = set_path(spec, key, value.item() if hasattr(value, "item") else value)
        plan.append((window, spec))
        bests.append((best, len(in_sample)))

    out_tasks = [
        OutOfSampleTask(bars, costs_for(costs, symbol), plan) for symbol, bars in universe.items()
    ]
    found = dict(zip(universe, run(out_of_sample_symbol, out_tasks, pool), strict=True))

    rows = []
    for number, ((window, _), (best, is_symbols)) in enumerate(zip(plan, bests, strict=True)):
        scored = [f[0][number] for f in found.values() if f[0][number] is not None]
        returns = [r for r, _ in scored]
        sharpes = [sh for _, sh in scored]
        days = (window.out_of_sample_end - window.out_of_sample_start).days
        median_return = float(np.median(returns)) if returns else 0.0
        rows.append(
            {
                "is_start": window.in_sample_start,
                "oos_start": window.out_of_sample_start,
                "oos_end": window.out_of_sample_end,
                **{key: best[key] for key in keys},
                "is_symbols": is_symbols,
                "is_median_sharpe": float(best["median_sharpe"]),
                "is_median_cagr": float(best["median_cagr"]),
                "oos_symbols": len(returns),
                "oos_median_sharpe": float(np.median(sharpes)) if sharpes else 0.0,
                "oos_median_return": median_return,
                "oos_annualised": _annualised(median_return, days),
            }
        )
    results = {symbol: f[1] for symbol, f in found.items() if f[1] is not None}
    return WalkForward(windows=pd.DataFrame(rows), results=results)
