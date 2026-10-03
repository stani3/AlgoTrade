"""``research feasibility``: Davey's limited testing, on development data only.

1. Entry test: the strategy's entries with neutral exits (fixed-bar and an ATR bracket).
2. Core system at the card's own parameters, across the whole symbol universe.
3. Monkey test: does it beat random trading with the same habits?
4. Limited optimisation over the pre-registered grid, and the plateau the parameters for
   validation come from.
5. Diagnostics (sides, symbols, years, regimes, holding times, MAE/MFE) for the agent to read,
   and the Davey performance report at the chosen parameters.

Every configuration evaluated is added to the trial ledger. Symbols are tested in parallel
(:mod:`algotrade.parallel`); the results do not depend on the number of workers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from algotrade.backtest.costs import CostModel
from algotrade.backtest.report_html import write_report
from algotrade.instruments import (
    costs_for_symbols,
    costs_text,
    describe,
    registry,
    universe_key,
)
from algotrade.parallel import Pool
from algotrade.validation.diagnostics import by_class, diagnostics
from algotrade.validation.entry_test import SCORINGS
from algotrade.validation.limited import (
    ChosenRun,
    EntrySettings,
    LimitedTest,
    run_chosen,
    run_limited,
)
from algotrade.validation.optimize import choose, profitable_share

from . import results, vcs
from .buildcheck import code_files
from .cards import card_hash, read_card
from .criteria import Check, Criteria, at_least
from .dedup import set_path, spec_hash
from .journal import TrialLedger, VersionState
from .registry import Refused
from .split import dev_end, dev_universe, position_caps, version_symbols, window
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


def entry_settings(criteria: Criteria) -> EntrySettings:
    feas = "feasibility."
    return EntrySettings(
        exit_bars=criteria.get(feas + "entry_exit_bars"),
        stop_atr=criteria.get(feas + "entry_stop_atr"),
        target_atr=criteria.get(feas + "entry_target_atr"),
        atr_length=criteria.get(feas + "entry_atr_length"),
        scoring=criteria.get(feas + "entry_scoring", "compounded"),  # older criteria: compounded
    )


@dataclass
class FeasibilityRun:
    """Everything feasibility measures, before anything is written."""

    limited: LimitedTest
    best: pd.Series
    spec: dict  # the chosen parameters
    chosen: ChosenRun
    checks: list[Check]
    diagnostics: dict[str, pd.DataFrame]
    metrics: dict
    notes: list[str]
    keys: list[str]

    @property
    def csvs(self) -> dict[str, str]:
        """The CSV files written next to ``result.json``, by name."""

        out = {
            f"diagnostics_{name}.csv": frame.to_csv() for name, frame in self.diagnostics.items()
        }
        out["entry_test.csv"] = self.limited.entry.table.to_csv(index=False)
        out["grid.csv"] = self.limited.board.drop(columns=["spec"]).to_csv(index=False)
        out["core_by_symbol.csv"] = self.limited.per_symbol.to_csv()
        return out

    @property
    def chosen_params(self) -> dict:
        return {k: plain(self.best[k]) for k in self.keys}


def run_feasibility(
    card_spec: dict,
    grid: dict[str, list],
    universe: dict[str, pd.DataFrame],
    costs: CostModel | Mapping[str, CostModel],
    criteria: Criteria,
    seed: int = 0,
    report: bool = True,
    workers: int | None = None,
    max_leverage: float | Mapping[str, float] = 1.0,
) -> FeasibilityRun:
    """Limited testing over ``universe``: the checks, metrics and notes feasibility records."""

    feas = "feasibility."
    entry_rules = entry_settings(criteria)
    keys = list(grid)
    with Pool(workers) as pool:
        limited = run_limited(
            card_spec, grid, universe, costs, entry_rules, criteria.get(feas + "monkey_runs"),
            seed, pool, max_leverage,
        )  # fmt: skip
        best = choose(limited.board, keys)
        spec = chosen_spec(card_spec, best, keys)
        chosen = run_chosen(spec, universe, costs, report, seed, pool, max_leverage)
    entry, per_symbol, monkeys, board = (
        limited.entry, limited.per_symbol, limited.monkeys, limited.board,
    )  # fmt: skip

    checks = [
        at_least("entry test: share of (exit, symbol) cells profitable", entry.profitable_share,
                 criteria.get(feas + "entry_min_profitable_share"), "{:.0%}"),
        at_least("core system: median Sharpe across symbols", float(per_symbol["sharpe"].median()),
                 criteria.get(feas + "core_min_median_sharpe")),
        at_least("core system: share of symbols with Sharpe > 0",
                 float((per_symbol["sharpe"] > 0).mean()),
                 criteria.get(feas + "core_min_positive_share"), "{:.0%}"),
        at_least("core system: median closed trades per symbol",
                 float(per_symbol["trades"].median()),
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
    found = diagnostics(chosen.results, universe)
    notes = [
        f"Entry test by exit ({SCORINGS[entry_rules.scoring]}):\n\n" + _markdown(entry.by_exit()),
        "Long against short (chosen parameters):\n\n" + _markdown(found["sides"]),
        "Regimes (chosen parameters):\n\n" + _markdown(found["regimes"].set_index(["kind", "regime"])),
        "Excursions (MAE/MFE):\n\n" + _markdown(found["excursions"]),
    ]  # fmt: skip
    chosen_by_symbol = chosen.per_symbol
    metrics = {
        "median Sharpe at the chosen parameters": float(chosen_by_symbol["sharpe"].median()),
        "plateau score of the chosen cell": float(best["plateau"]),
        "monkey median Sharpe (median run)": float(pd.Series(monkeys.monkeys).median()),
        "median CAGR at the chosen parameters": float(chosen_by_symbol["cagr"].median()),
        "worst drawdown at the chosen parameters": float(chosen_by_symbol["max_drawdown"].min()),
    }
    return FeasibilityRun(limited, best, spec, chosen, checks, found, metrics, notes, keys)


def feasibility(
    ws: Workspace,
    criteria: Criteria,
    version: VersionState,
    commit: bool = True,
    report: bool = True,
    seed: int = 0,
    workers: int | None = None,
) -> results.StageResult:
    require_ready(ws, version, "feasibility", after="build")
    card = read_card(ws.root / version.card_path)
    universe = dev_universe(ws, criteria, card.timeframe, version_symbols(criteria, version))
    costs = costs_for_symbols(criteria, list(universe))
    caps = position_caps(criteria, list(universe), card.spec)
    found = run_feasibility(card.spec, card.optimise, universe, costs, criteria, seed, report,
                            workers, caps)  # fmt: skip
    board, per_symbol = found.limited.board, found.limited.per_symbol

    folder = ws.stage_dir(version.idea, version.slug, version.version, "feasibility")
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in found.csvs.items():
        (folder / name).write_text(text, encoding="utf-8", newline="")
    notes, extra = list(found.notes), {}
    if len(version.universe) > 1:  # evidence per asset class, for the agent to read
        classes = {s: i.asset_class for s, i in registry(criteria).items() if s in universe}
        table = by_class(per_symbol, classes, found.limited.entry.table)
        table.to_csv(folder / "core_by_class.csv")
        notes.append("By asset class (the card's parameters):\n\n" + _markdown(table))
    if version.universe != ("crypto",):
        extra["universe"] = list(version.universe)

    ledger_rows = [
        {
            "idea": version.idea,
            "version": version.version,
            "stage": "feasibility",
            "kind": "grid" if found.keys else "base",
            "label": f"{version.idea} v{version.version}",
            "spec_hash": row["spec_hash"],
            "timeframe": card.timeframe,
            "metric": row["median_sharpe"],
            "observations": int(pd.Series([len(b) for b in universe.values()]).median()),
            "window": f"dev, before {dev_end(criteria):%Y-%m-%d}",
            "spec": row["spec"],
            "universe": universe_key(version.universe),
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
        settings = {
            "Idea": f"{version.idea} v{version.version}: {version.title}",
            "Strategy spec (chosen from the grid plateau)": json.dumps(found.spec),
            "Data": f"{describe(criteria, list(universe), card.timeframe)}, development period "
            f"(before {dev_end(criteria):%Y-%m-%d})",
            "Costs": costs_text(costs),
        }
        page = write_report(
            found.chosen.reports, ws.report_dir(version.idea, version.version, "feasibility"),
            f"{version.idea} v{version.version} feasibility", settings, found.chosen.sections,
        )  # fmt: skip
        report_path = ws.relative(page)

    windows = [window(bars, symbol, card.timeframe) for symbol, bars in universe.items()]
    result = results.StageResult(
        version.idea,
        version.version,
        "feasibility",
        found.checks,
        metrics=found.metrics,
        trials=len(board),
        notes=notes,
        provenance=results.provenance(ws, criteria, version, spec_hash(found.spec), windows),
        report=report_path,
        extra={"chosen": found.chosen_params, "chosen_spec": found.spec, **extra},
    )
    results.record(ws, version, result, commit=commit)
    return result
