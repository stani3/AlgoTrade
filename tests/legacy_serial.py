"""Frozen copies of the serial research code as it was before parallelism and multi-asset support.

The parallel and per-symbol versions in ``src`` must give exactly these numbers on crypto. Do
not change this module to follow the production code: it is the reference.

Copied verbatim (apart from names) from commit 9dff4b7: ``run_entry_test`` (validation/
entry_test.py), ``monkey_test`` (validation/monkey.py), ``evaluate`` and ``run_grid``
(validation/optimize.py), ``walk_forward`` (validation/walkforward.py) and the compute part of
``feasibility`` (research/feasibility.py, everything but recording and the HTML report).
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from algotrade.backtest.bracket import BracketSignals, simulate_bracket
from algotrade.backtest.costs import CostModel
from algotrade.backtest.engine import BacktestResult
from algotrade.backtest.metrics import aggregate, periods_per_year, sharpe_ratio, summarize
from algotrade.backtest.runner import backtest
from algotrade.indicators import atr
from algotrade.research.criteria import Criteria, at_least
from algotrade.research.dedup import grid_specs, set_path, spec_hash
from algotrade.strategies import BracketStrategy, Strategy, from_spec
from algotrade.validation.diagnostics import diagnostics
from algotrade.validation.entry_test import (
    FIXED_SIZE,
    SCORINGS,
    EntryTest,
    cell_profit,
    entry_signals,
)
from algotrade.validation.fast import bracket_returns, position_returns, sharpe
from algotrade.validation.monkey import MonkeyTest, _bracket_rules
from algotrade.validation.optimize import choose, profitable_share
from algotrade.validation.walkforward import (
    WalkForward,
    _annualised,
    make_windows,
    out_of_sample,
    stitch,
)


def legacy_entry_test(
    strategy: Strategy,
    universe: dict[str, pd.DataFrame],
    costs: CostModel,
    exit_bars: list[int],
    stop_atr: float,
    target_atr: float,
    atr_length: int,
    scoring: str = "compounded",
) -> EntryTest:
    if scoring not in SCORINGS:
        raise ValueError(f"entry test scoring must be one of {sorted(SCORINGS)}, not {scoring!r}")
    rows = []
    for symbol, bars in universe.items():
        long, short = entry_signals(strategy, bars)
        volatility = atr(bars["high"], bars["low"], bars["close"], atr_length)
        never = pd.Series(np.inf, index=bars.index)
        exits = {f"{n} bars": (never, never, n) for n in exit_bars}
        exits[f"{stop_atr:g}/{target_atr:g} ATR bracket"] = (
            stop_atr * volatility,
            target_atr * volatility,
            0,
        )
        for name, (stop, target, max_bars) in exits.items():
            signals = BracketSignals(long, short, stop, target)
            whole = simulate_bracket(bars, signals, costs, max_bars=max_bars)
            small = simulate_bracket(bars, signals, costs, leverage=FIXED_SIZE, max_bars=max_bars)
            closed = small.trades[~small.trades["open"].astype(bool)]
            per_unit = closed["return"] / FIXED_SIZE
            fixed = cell_profit(per_unit, float(small.equity.iloc[-1]), "fixed")
            compounded = cell_profit(per_unit, float(whole.equity.iloc[-1]), "compounded")
            profit = fixed if scoring == "fixed" else compounded
            rows.append(
                {
                    "exit": name,
                    "symbol": symbol,
                    "trades": len(closed),
                    "win_rate": float((per_unit > 0).mean()) if len(closed) else 0.0,
                    "avg_trade": float(per_unit.mean()) if len(closed) else 0.0,
                    "net_return": compounded,
                    "fixed_return": fixed,
                    "profitable": bool(profit > 0 and len(closed) > 0),
                }
            )
    return EntryTest(pd.DataFrame(rows))


def legacy_monkey_test(
    strategy: Strategy,
    universe: dict[str, pd.DataFrame],
    costs: CostModel,
    runs: int,
    seed: int = 0,
) -> MonkeyTest:
    rng = np.random.default_rng(seed)
    real = []
    monkeys = np.zeros((runs, len(universe)))
    for column, bars in enumerate(universe.values()):
        ppy = periods_per_year(bars.index)
        n = len(bars)
        if isinstance(strategy, BracketStrategy):
            signals = strategy.signals(bars)
            long = signals.long.fillna(False).to_numpy(dtype=bool)
            short = signals.short.fillna(False).to_numpy(dtype=bool)
            stop, target = signals.stop_dist.to_numpy(), signals.target_dist.to_numpy()
            rules = _bracket_rules(strategy)
            real.append(
                sharpe(bracket_returns(bars, long, short, stop, target, costs, **rules), ppy)
            )
            p_long, p_short = long.mean(), short.mean()
            for run in range(runs):
                draw = rng.random(n)
                fake_long = draw < p_long
                fake_short = (draw >= p_long) & (draw < p_long + p_short)
                returns = bracket_returns(bars, fake_long, fake_short, stop, target, costs, **rules)
                monkeys[run, column] = sharpe(returns, ppy)
        else:
            exposure = strategy.target_position(bars).to_numpy(dtype="float64")
            real.append(sharpe(position_returns(bars, exposure, costs), ppy))
            low, high = max(n // 10, 1), max(n - n // 10, 2)
            for run in range(runs):
                shifted = np.roll(exposure, int(rng.integers(low, high)))
                monkeys[run, column] = sharpe(position_returns(bars, shifted, costs), ppy)
    return MonkeyTest(real=float(np.median(real)), monkeys=np.median(monkeys, axis=1))


def legacy_evaluate(
    spec: dict, universe: dict[str, pd.DataFrame], costs: CostModel, max_leverage: float = 1.0
) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    strategy = from_spec(spec)
    results = {s: backtest(strategy, bars, costs, max_leverage) for s, bars in universe.items()}
    return results, pd.DataFrame({s: summarize(r) for s, r in results.items()}).T


def legacy_run_grid(
    base: dict,
    grid: dict[str, list],
    universe: dict[str, pd.DataFrame],
    costs: CostModel,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    rows = []
    combos = list(itertools.product(*grid.values())) if grid else [()]
    for combo, spec in zip(combos, grid_specs(base, grid), strict=True):
        results, per_symbol = legacy_evaluate(spec, universe, costs, max_leverage)
        rows.append(
            {
                **dict(zip(grid, combo, strict=True)),
                **aggregate(per_symbol),
                "median_trades": float(per_symbol["trades"].median()),
                "killed": float(np.mean(["killed_at" in r.meta for r in results.values()])),
                "spec_hash": spec_hash(spec),
                "spec": json.dumps(spec, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def legacy_walk_forward(
    base: dict,
    grid: dict[str, list],
    universe: dict[str, pd.DataFrame],
    costs: CostModel,
    end: pd.Timestamp,
    in_sample_years: float,
    out_of_sample_months: int,
    min_in_sample_share: float = 0.5,
) -> WalkForward:
    keys = list(grid)
    first = min(bars.index[0] for bars in universe.values())
    windows = make_windows(first, end, in_sample_years, out_of_sample_months)
    rows, pieces = [], {symbol: [] for symbol in universe}
    for window in windows:
        in_sample = {}
        for symbol, bars in universe.items():
            part = bars[
                (bars.index >= window.in_sample_start) & (bars.index < window.out_of_sample_start)
            ]
            span = window.out_of_sample_start - window.in_sample_start
            if len(part) and (part.index[-1] - part.index[0]) >= span * min_in_sample_share:
                in_sample[symbol] = part
        if not in_sample:
            continue
        board = legacy_run_grid(base, grid, in_sample, costs)
        best = choose(board, keys)
        spec = base
        for key in keys:
            value = best[key]
            spec = set_path(spec, key, value.item() if hasattr(value, "item") else value)
        returns, sharpes = [], []
        for symbol, bars in universe.items():
            history = bars[bars.index < window.out_of_sample_end]
            if not (history.index >= window.out_of_sample_start).any():
                continue
            result = out_of_sample(spec, history, window.out_of_sample_start, costs)
            pieces[symbol].append(result)
            returns.append(float(result.equity.iloc[-1] - 1.0))
            sharpes.append(sharpe_ratio(result.returns, periods_per_year(result.ledger.index)))
        days = (window.out_of_sample_end - window.out_of_sample_start).days
        median_return = float(np.median(returns)) if returns else 0.0
        rows.append(
            {
                "is_start": window.in_sample_start,
                "oos_start": window.out_of_sample_start,
                "oos_end": window.out_of_sample_end,
                **{key: best[key] for key in keys},
                "is_symbols": len(in_sample),
                "is_median_sharpe": float(best["median_sharpe"]),
                "is_median_cagr": float(best["median_cagr"]),
                "oos_symbols": len(returns),
                "oos_median_sharpe": float(np.median(sharpes)) if sharpes else 0.0,
                "oos_median_return": median_return,
                "oos_annualised": _annualised(median_return, days),
            }
        )
    results = {symbol: stitch(parts, costs) for symbol, parts in pieces.items() if parts}
    return WalkForward(windows=pd.DataFrame(rows), results=results)


def _plain(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    shown = frame.round(digits).reset_index()
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "---|" * len(shown.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def legacy_feasibility(
    card_spec: dict,
    grid: dict[str, list],
    universe: dict[str, pd.DataFrame],
    costs: CostModel,
    criteria: Criteria,
    seed: int = 0,
) -> dict:
    """What ``feasibility`` recorded: checks, metrics, notes, the chosen cell and CSV texts."""

    feas = "feasibility."
    strategy = from_spec(card_spec)
    scoring = criteria.get(feas + "entry_scoring", "compounded")
    entry = legacy_entry_test(
        strategy,
        universe,
        costs,
        criteria.get(feas + "entry_exit_bars"),
        criteria.get(feas + "entry_stop_atr"),
        criteria.get(feas + "entry_target_atr"),
        criteria.get(feas + "entry_atr_length"),
        scoring,
    )
    _, per_symbol = legacy_evaluate(card_spec, universe, costs)
    monkeys = legacy_monkey_test(
        strategy, universe, costs, criteria.get(feas + "monkey_runs"), seed
    )
    keys = list(grid)
    board = legacy_run_grid(card_spec, grid, universe, costs)
    best = choose(board, keys)
    spec = card_spec
    for key in keys:
        spec = set_path(spec, key, _plain(best[key]))
    chosen_results, chosen_per_symbol = legacy_evaluate(spec, universe, costs)

    median_trades = float(per_symbol["trades"].median())
    checks = [
        at_least("entry test: share of (exit, symbol) cells profitable", entry.profitable_share,
                 criteria.get(feas + "entry_min_profitable_share"), "{:.0%}"),
        at_least("core system: median Sharpe across symbols", float(per_symbol["sharpe"].median()),
                 criteria.get(feas + "core_min_median_sharpe")),
        at_least("core system: share of symbols with Sharpe > 0",
                 float((per_symbol["sharpe"] > 0).mean()),
                 criteria.get(feas + "core_min_positive_share"), "{:.0%}"),
        at_least("core system: median closed trades per symbol", median_trades,
                 criteria.get(feas + "min_trades_per_symbol")),
        at_least("monkey test: share of random monkeys beaten", monkeys.percentile,
                 criteria.get(feas + "monkey_min_percentile"), "{:.0%}"),
    ]  # fmt: skip
    if keys:
        checks.append(
            at_least(
                "limited optimisation: share of grid combinations profitable",
                profitable_share(board),
                criteria.get(feas + "optimise_min_profitable_share"),
                "{:.0%}",
            )
        )
    found = diagnostics(chosen_results, universe)
    csvs = {f"diagnostics_{name}.csv": frame.to_csv() for name, frame in found.items()}
    csvs["entry_test.csv"] = entry.table.to_csv(index=False)
    csvs["grid.csv"] = board.drop(columns=["spec"]).to_csv(index=False)
    csvs["core_by_symbol.csv"] = per_symbol.to_csv()
    notes = [
        f"Entry test by exit ({SCORINGS[scoring]}):\n\n" + _markdown(entry.by_exit()),
        "Long against short (chosen parameters):\n\n" + _markdown(found["sides"]),
        "Regimes (chosen parameters):\n\n" + _markdown(found["regimes"].set_index(["kind", "regime"])),
        "Excursions (MAE/MFE):\n\n" + _markdown(found["excursions"]),
    ]  # fmt: skip
    return {
        "checks": [c.to_dict() for c in checks],
        "metrics": {
            "median Sharpe at the chosen parameters": float(chosen_per_symbol["sharpe"].median()),
            "plateau score of the chosen cell": float(best["plateau"]),
            "monkey median Sharpe (median run)": float(pd.Series(monkeys.monkeys).median()),
            "median CAGR at the chosen parameters": float(chosen_per_symbol["cagr"].median()),
            "worst drawdown at the chosen parameters": float(
                chosen_per_symbol["max_drawdown"].min()
            ),
        },
        "trials": len(board),
        "notes": notes,
        "chosen": {k: _plain(best[k]) for k in keys},
        "chosen_spec": spec,
        "csvs": csvs,
    }
