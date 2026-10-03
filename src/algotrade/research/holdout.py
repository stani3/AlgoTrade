"""``research holdout``: one look at data the research never saw, then ``freeze``.

The strategy that passed validation (the card's spec with the parameters feasibility chose
from the plateau) is traded from the first holdout bar to the end of the data, starting flat
with indicators warmed up on the history before. The look is journaled before any data is
loaded and a second one is refused unless the user forces it.

It passes when the median symbol is still profitable on a risk-adjusted basis and its drawdown
stays inside the 95th percentile of drawdowns of same-length paths bootstrapped (in 10-day
blocks) from the walk-forward's out-of-sample bar returns.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
from algotrade.backtest.report import build_report
from algotrade.backtest.report_html import write_report
from algotrade.validation.bands import block_bootstrap
from algotrade.validation.walkforward import out_of_sample

from . import results
from .criteria import Criteria, above, at_most
from .dedup import spec_hash
from .feasibility import require_ready
from .journal import Journal, VersionState
from .registry import Refused
from .split import dev_end, for_version, load_full_bars, open_holdout, window
from .validate import benchmark
from .workspace import Workspace

BOOTSTRAP_BLOCK_DAYS = 10


def _bars_per_day(found: dict) -> float:
    rates = [periods_per_year(result.ledger.index) / 365.25 for _, result in found.values()]
    return float(np.median(rates)) if rates else 1.0


def load_result(ws: Workspace, version: VersionState, stage: str) -> dict:
    path = ws.stage_dir(version.idea, version.slug, version.version, stage) / "result.json"
    if not path.exists():
        raise Refused(f"{version.idea} v{version.version} has no {stage} result")
    return json.loads(path.read_text(encoding="utf-8"))


def holdout_results(ws: Workspace, criteria: Criteria, spec: dict, timeframe: str) -> dict:
    costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
    start = dev_end(criteria)
    out = {}
    for symbol in criteria.get("data.symbols"):
        bars = load_full_bars(ws, criteria, symbol, timeframe)
        if (bars.index >= start).sum() > 1:
            out[symbol] = (bars, out_of_sample(spec, bars, start, costs))
    return out


def holdout(
    ws: Workspace,
    criteria: Criteria,
    version: VersionState,
    force: bool = False,
    reason: str = "",
    commit: bool = True,
    report: bool = True,
    seed: int = 0,
) -> results.StageResult:
    require_ready(ws, version, "holdout", after="validation", repeat=force)
    criteria = for_version(ws, criteria, version)
    feasibility = load_result(ws, version, "feasibility")
    validation = ws.stage_dir(version.idea, version.slug, version.version, "validation")
    oos = pd.read_parquet(validation / "oos_returns.parquet")
    spec = feasibility["chosen_spec"]
    open_holdout(Journal(ws.journal_path), version, force=force, reason=reason)
    card_timeframe = version.timeframe
    found = holdout_results(ws, criteria, spec, card_timeframe)
    costs = EXCHANGE_COSTS[criteria.get("data.exchange")]

    sharpes, drawdowns, trades = [], [], []
    for _, result in found.values():
        sharpes.append(sharpe_ratio(result.returns, periods_per_year(result.ledger.index)))
        drawdowns.append(float(-(result.equity / result.equity.cummax() - 1.0).min()))
        trades.append(int((~result.trades["open"].astype(bool)).sum()))
    median_sharpe = float(np.median(sharpes)) if sharpes else 0.0
    median_dd = float(np.median(drawdowns)) if drawdowns else 0.0
    checks = [
        above("holdout median Sharpe across symbols", median_sharpe,
              criteria.get("holdout.min_sharpe")),
    ]  # fmt: skip

    # What a run this long should look like: block bootstrap of the out-of-sample bar returns.
    failures = []
    band = None
    lengths = [len(result.ledger) for _, result in found.values()]
    series = [oos[column].dropna().to_numpy() for column in oos.columns]
    block = max(round(BOOTSTRAP_BLOCK_DAYS * _bars_per_day(found)), 1)
    try:
        bands = block_bootstrap(
            series, int(np.median(lengths)) if lengths else 0,
            criteria.get("validation.monte_carlo_runs"), block, seed,
        )  # fmt: skip
        band = bands.drawdown(criteria.get("holdout.max_drawdown_percentile"))
        checks.append(
            at_most(
                "holdout median max drawdown within the bootstrap band", median_dd, band, "{:.0%}"
            )
        )
    except ValueError:
        failures.append("not enough out-of-sample data or holdout bars to build the drawdown band")

    folder = ws.stage_dir(version.idea, version.slug, version.version, "holdout")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sharpe": sharpes, "max_drawdown": drawdowns, "closed_trades": trades},
        index=list(found),
    ).to_csv(folder / "holdout_by_symbol.csv")

    report_path = None
    if report and found:
        reports = [
            build_report(r, benchmark=benchmark(bars, r.ledger.index, costs), seed=seed)
            for bars, r in found.values()
        ]
        settings = {
            "Idea": f"{version.idea} v{version.version}: {version.title}",
            "Strategy spec": json.dumps(spec),
            "Data": f"holdout: {criteria.get('data.exchange')} {card_timeframe} from "
            f"{dev_end(criteria):%Y-%m-%d} to the end of the data, starting flat",
        }
        page = write_report(
            reports, ws.report_dir(version.idea, version.version, "holdout"),
            f"{version.idea} v{version.version} holdout", settings,
        )  # fmt: skip
        report_path = ws.relative(page)

    windows = [
        window(bars[bars.index >= dev_end(criteria)], symbol, card_timeframe)
        for symbol, (bars, _) in found.items()
    ]
    result = results.StageResult(
        version.idea,
        version.version,
        "holdout",
        checks,
        failures=failures,
        metrics={
            "symbols with holdout data": len(found),
            "median closed trades per symbol": float(np.median(trades)) if trades else 0.0,
            "bootstrap drawdown band for this length (95th percentile)": band if band else 0.0,
        },
        notes=[
            f"Look number {version.holdout_looks + 1}" + (f" (forced: {reason})" if force else "")
        ],
        provenance=results.provenance(ws, criteria, version, spec_hash(spec), windows),
        report=report_path,
        extra={"spec": spec},
    )
    results.record(ws, version, result, commit=commit)
    return result
