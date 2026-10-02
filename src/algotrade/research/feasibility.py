"""``research feasibility``: Davey's limited testing, on development data only.

1. Entry test: the strategy's entries with neutral exits (fixed-bar and an ATR bracket).
2. Core system at the card's own parameters, across the whole symbol universe.
3. Monkey test: does it beat random trading with the same habits?
4. Limited optimisation over the pre-registered grid, and the plateau the parameters for
   validation come from.
5. Diagnostics (sides, symbols, years, regimes, holding times, MAE/MFE) for the agent to read,
   and the Davey performance report at the chosen parameters.

Every configuration evaluated is added to the trial ledger.
"""

from __future__ import annotations

import json

import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.engine import run_backtest
from algotrade.backtest.report import build_report
from algotrade.backtest.report_html import write_report
from algotrade.strategies import from_spec
from algotrade.validation.diagnostics import diagnostics
from algotrade.validation.entry_test import run_entry_test
from algotrade.validation.monkey import monkey_test
from algotrade.validation.optimize import choose, evaluate, profitable_share, run_grid

from . import results, vcs
from .buildcheck import code_files
from .cards import card_hash, read_card
from .criteria import Criteria, at_least
from .dedup import set_path, spec_hash
from .journal import TrialLedger, VersionState
from .registry import Refused
from .split import dev_end, dev_universe, window
from .workspace import Workspace


def require_ready(
    ws: Workspace, version: VersionState, stage: str, after: str, repeat: bool = False
) -> None:
    """The previous stage passed, the card is untouched and the idea's files are committed.

    ``repeat`` allows running ``stage`` again (a holdout look the user forced), even when that
    stage's own earlier verdict closed the idea.
    """

    if version.failed and not (repeat and version.status == f"failed:{stage}"):
        raise Refused(f"{version.idea} v{version.version} is closed ({version.status})")
    if version.stages.get(after, {}).get("verdict") != "PASS":
        raise Refused(f"{version.idea} v{version.version} needs a passing {after} stage first")
    if stage in version.stages and not repeat:
        raise Refused(f"{version.idea} v{version.version} already has a {stage} result")
    path = ws.root / version.card_path
    if card_hash(path) != version.card_hash:
        raise Refused(f"{version.card_path} was edited after registration")
    files = [path] + [p for _, module, tests in code_files(ws, version) for p in (module, tests)]
    vcs.require_clean(ws.root, files, f"{version.idea} v{version.version}")


def plain(value: object) -> object:
    """Numpy scalars from a results table as plain Python values (for specs and JSON)."""

    return value.item() if hasattr(value, "item") else value


def chosen_spec(base: dict, row: pd.Series, keys: list[str]) -> dict:
    spec = base
    for key in keys:
        spec = set_path(spec, key, plain(row[key]))
    return spec


def _markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    shown = frame.round(digits).reset_index()
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "---|" * len(shown.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def feasibility(
    ws: Workspace,
    criteria: Criteria,
    version: VersionState,
    commit: bool = True,
    report: bool = True,
    seed: int = 0,
) -> results.StageResult:
    require_ready(ws, version, "feasibility", after="build")
    card = read_card(ws.root / version.card_path)
    universe = dev_universe(ws, criteria, card.timeframe)
    costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
    feas = "feasibility."
    strategy = from_spec(card.spec)

    entry = run_entry_test(
        strategy,
        universe,
        costs,
        criteria.get(feas + "entry_exit_bars"),
        criteria.get(feas + "entry_stop_atr"),
        criteria.get(feas + "entry_target_atr"),
        criteria.get(feas + "entry_atr_length"),
    )
    _, per_symbol = evaluate(card.spec, universe, costs)
    monkeys = monkey_test(strategy, universe, costs, criteria.get(feas + "monkey_runs"), seed)
    keys = list(card.optimise)
    board = run_grid(card.spec, card.optimise, universe, costs)
    best = choose(board, keys)
    spec = chosen_spec(card.spec, best, keys)
    chosen_results, chosen_per_symbol = evaluate(spec, universe, costs)

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

    folder = ws.stage_dir(version.idea, version.slug, version.version, "feasibility")
    folder.mkdir(parents=True, exist_ok=True)
    found = diagnostics(chosen_results, universe)
    for name, frame in found.items():
        frame.to_csv(folder / f"diagnostics_{name}.csv")
    entry.table.to_csv(folder / "entry_test.csv", index=False)
    board.drop(columns=["spec"]).to_csv(folder / "grid.csv", index=False)
    per_symbol.to_csv(folder / "core_by_symbol.csv")

    ledger_rows = [
        {
            "idea": version.idea,
            "version": version.version,
            "stage": "feasibility",
            "kind": "grid" if keys else "base",
            "label": f"{version.idea} v{version.version}",
            "spec_hash": row["spec_hash"],
            "timeframe": card.timeframe,
            "metric": row["median_sharpe"],
            "observations": int(pd.Series([len(b) for b in universe.values()]).median()),
            "window": f"dev, before {dev_end(criteria):%Y-%m-%d}",
            "spec": row["spec"],
        }
        for _, row in board.iterrows()
    ]
    base_hash = spec_hash(card.spec)
    if base_hash not in set(board["spec_hash"]):
        ledger_rows.append(
            {
                **ledger_rows[0],
                "kind": "base",
                "spec_hash": base_hash,
                "metric": float(per_symbol["sharpe"].median()),
                "spec": json.dumps(card.spec, sort_keys=True),
            }
        )
    TrialLedger(ws.trials_path).add(ledger_rows)

    report_path = None
    if report:
        reports = []
        for symbol, bars in universe.items():
            hold = run_backtest(bars, pd.Series(1.0, index=bars.index), costs)
            reports.append(build_report(chosen_results[symbol], benchmark=hold, seed=seed))
        settings = {
            "Idea": f"{version.idea} v{version.version}: {version.title}",
            "Strategy spec (chosen from the grid plateau)": json.dumps(spec),
            "Data": f"{criteria.get('data.exchange')} {card.timeframe}, development period "
            f"(before {dev_end(criteria):%Y-%m-%d})",
            "Costs": f"fee {costs.fee_bps} bps + slippage {costs.slippage_bps} bps, funding on",
        }
        page = write_report(
            reports, ws.report_dir(version.idea, version.version, "feasibility"),
            f"{version.idea} v{version.version} feasibility", settings,
        )  # fmt: skip
        report_path = ws.relative(page)

    windows = [window(bars, symbol, card.timeframe) for symbol, bars in universe.items()]
    notes = [
        "Entry test by exit:\n\n" + _markdown(entry.by_exit()),
        "Long against short (chosen parameters):\n\n" + _markdown(found["sides"]),
        "Regimes (chosen parameters):\n\n" + _markdown(found["regimes"].set_index(["kind", "regime"])),
        "Excursions (MAE/MFE):\n\n" + _markdown(found["excursions"]),
    ]  # fmt: skip
    result = results.StageResult(
        version.idea,
        version.version,
        "feasibility",
        checks,
        metrics={
            "median Sharpe at the chosen parameters": float(chosen_per_symbol["sharpe"].median()),
            "plateau score of the chosen cell": float(best["plateau"]),
            "monkey median Sharpe (median run)": float(pd.Series(monkeys.monkeys).median()),
            "median CAGR at the chosen parameters": float(chosen_per_symbol["cagr"].median()),
            "worst drawdown at the chosen parameters": float(
                chosen_per_symbol["max_drawdown"].min()
            ),
        },
        trials=len(board),
        notes=notes,
        provenance=results.provenance(ws, criteria, version, spec_hash(spec), windows),
        report=report_path,
        extra={"chosen": {k: plain(best[k]) for k in keys}, "chosen_spec": spec},
    )
    results.record(ws, version, result, commit=commit)
    return result
