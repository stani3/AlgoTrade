"""``research report``: rebuild a stage's HTML report from what was committed.

The HTML reports are not committed (they are large); the spec, parameters, data window hashes
and headline metrics are. This recomputes the stage from those, refuses if the market data no
longer hashes the same, checks the headline metric matches the committed result, and writes the
report again. No holdout look is journaled: nothing new is learned.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
from algotrade.backtest.report_html import render_reports, write_report
from algotrade.instruments import costs_for_symbols
from algotrade.parallel import Pool
from algotrade.validation.optimize import evaluate

from .cards import read_card
from .criteria import Criteria
from .holdout import holdout_results, load_result
from .journal import VersionState
from .registry import Refused
from .split import dev_end, dev_universe, position_caps, version_symbols, window
from .validate import deflation, run_walk_forward
from .workspace import Workspace

HEADLINE = {
    "feasibility": "median Sharpe at the chosen parameters",
    "validation": "out-of-sample annualised Sharpe of the median symbol",
}


def _check_data(recorded: list[dict], current: list) -> None:
    now = {w.symbol: w.sha for w in current}
    changed = [w["symbol"] for w in recorded if now.get(w["symbol"]) != w["sha"]]
    if changed:
        raise Refused(
            "market data changed since the result was recorded (hash mismatch for "
            f"{', '.join(changed)}); the report cannot be reproduced"
        )


def _check_metric(expected: float, actual: float, name: str) -> None:
    if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12):
        raise Refused(f"reproduced {name} {actual!r} differs from the committed {expected!r}")


def reproduce(
    ws: Workspace,
    criteria: Criteria,
    version: VersionState,
    stage: str,
    workers: int | None = None,
) -> str:
    with Pool(workers) as pool:
        return _reproduce(ws, criteria, version, stage, pool)


def _reproduce(
    ws: Workspace, criteria: Criteria, version: VersionState, stage: str, pool: Pool
) -> str:
    recorded = load_result(ws, version, stage)
    card = read_card(ws.root / version.card_path)
    title = f"{version.idea} v{version.version} {stage} (reproduced)"
    symbols = version_symbols(criteria, version)
    if stage == "feasibility":
        universe = dev_universe(ws, criteria, card.timeframe, symbols)
        _check_data(
            recorded["provenance"]["data"],
            [window(b, s, card.timeframe) for s, b in universe.items()],
        )
        found, per_symbol = evaluate(
            recorded["chosen_spec"], universe, costs_for_symbols(criteria, list(universe)),
            position_caps(criteria, list(universe), recorded["chosen_spec"]),
        )  # fmt: skip
        _check_metric(
            recorded["metrics"][HEADLINE[stage]],
            float(per_symbol["sharpe"].median()),
            HEADLINE[stage],
        )
        pairs = {s: (universe[s], r) for s, r in found.items()}
        settings = {"Strategy spec": json.dumps(recorded["chosen_spec"])}
    elif stage == "validation":
        universe = dev_universe(ws, criteria, card.timeframe, symbols)
        _check_data(
            recorded["provenance"]["data"],
            [window(b, s, card.timeframe) for s, b in universe.items()],
        )
        wf, _ = run_walk_forward(ws, criteria, card, pool, symbols)
        _check_metric(
            recorded["metrics"][HEADLINE[stage]],
            deflation(ws, wf.results)["annualised_sharpe"],
            HEADLINE[stage],
        )
        pairs = {s: (universe[s], r) for s, r in wf.results.items()}
        settings = {"Base spec": json.dumps(card.spec), "Walk-forward": "as recorded"}
    elif stage == "holdout":
        found = holdout_results(ws, criteria, recorded["spec"], card.timeframe, symbols)
        start = dev_end(criteria)
        _check_data(
            recorded["provenance"]["data"],
            [window(b[b.index >= start], s, card.timeframe) for s, (b, _) in found.items()],
        )
        sharpes = [
            sharpe_ratio(r.returns, periods_per_year(r.ledger.index)) for _, r in found.values()
        ]
        _check_metric(
            recorded["checks"][0]["value"],
            float(pd.Series(sharpes).median()),
            "holdout median Sharpe",
        )
        pairs = found
        settings = {"Strategy spec": json.dumps(recorded["spec"])}
    else:
        raise Refused(f"no report to reproduce for stage '{stage}'")
    costs = costs_for_symbols(criteria, list(pairs))
    reports, sections = render_reports(
        [(r, bars, costs[s]) for s, (bars, r) in pairs.items()], 0, pool
    )
    page = write_report(
        reports, ws.report_dir(version.idea, version.version, stage), title, settings, sections
    )
    return ws.relative(page)
