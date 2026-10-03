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

from algotrade.backtest.costs import cap_for
from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
from algotrade.backtest.report_html import render_reports, write_report
from algotrade.calendars import infer_calendar
from algotrade.instruments import CRYPTO, costs_for_symbols, describe, symbols_for
from algotrade.parallel import Pool
from algotrade.validation.bands import block_bootstrap
from algotrade.validation.walkforward import out_of_sample

from . import results
from .criteria import Criteria, above, at_most
from .dedup import spec_hash
from .feasibility import require_ready
from .journal import Journal, VersionState
from .registry import Refused
from .split import (
    dev_end,
    for_version,
    load_full_bars,
    open_holdout,
    position_caps,
    version_symbols,
    window,
)
from .workspace import Workspace

BOOTSTRAP_BLOCK_DAYS = 10


def _bars_per_day(found: dict) -> float:
    rates = [periods_per_year(result.ledger.index) / 365.25 for _, result in found.values()]
    return float(np.median(rates)) if rates else 1.0


def calendar_groups(found: dict) -> dict[str, list[str]]:
    """Symbols by trading calendar (one group unless the symbols trade on different ones)."""

    groups: dict[str, list[str]] = {}
    for symbol, (_, result) in found.items():
        groups.setdefault(infer_calendar(result.ledger.index), []).append(symbol)
    return groups or {"24/7": []}


def load_result(ws: Workspace, version: VersionState, stage: str) -> dict:
    path = ws.stage_dir(version.idea, version.slug, version.version, stage) / "result.json"
    if not path.exists():
        raise Refused(f"{version.idea} v{version.version} has no {stage} result")
    return json.loads(path.read_text(encoding="utf-8"))


def holdout_results(
    ws: Workspace,
    criteria: Criteria,
    spec: dict,
    timeframe: str,
    symbols: list[str] | None = None,
) -> dict:
    """``spec`` traded from the holdout cutoff on every symbol (default: crypto) with data."""

    symbols = symbols or symbols_for(criteria, (CRYPTO,))
    costs = costs_for_symbols(criteria, symbols)
    caps = position_caps(criteria, symbols, spec)
    start = dev_end(criteria)
    out = {}
    for symbol in symbols:
        bars = load_full_bars(ws, criteria, symbol, timeframe)
        if (bars.index >= start).sum() > 1:
            trade = out_of_sample(spec, bars, start, costs[symbol], cap_for(caps, symbol))
            out[symbol] = (bars, trade)
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
    workers: int | None = None,
) -> results.StageResult:
    require_ready(ws, version, "holdout", after="validation", repeat=force)
    criteria = for_version(ws, criteria, version)
    feasibility = load_result(ws, version, "feasibility")
    validation = ws.stage_dir(version.idea, version.slug, version.version, "validation")
    oos = pd.read_parquet(validation / "oos_returns.parquet")
    spec = feasibility["chosen_spec"]
    open_holdout(Journal(ws.journal_path), version, force=force, reason=reason)
    card_timeframe = version.timeframe
    found = holdout_results(ws, criteria, spec, card_timeframe, version_symbols(criteria, version))
    costs = costs_for_symbols(criteria, list(found))

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

    # What a run this long should look like: block bootstrap of the out-of-sample bar returns,
    # per trading calendar when the symbols do not share one (bars per day differ).
    failures = []
    band = None
    groups = calendar_groups(found)
    for calendar, members in groups.items():
        part = {s: found[s] for s in members}
        lengths = [len(result.ledger) for _, result in part.values()]
        columns = [c for c in oos.columns if c in members] if len(groups) > 1 else oos.columns
        series = [oos[column].dropna().to_numpy() for column in columns]
        block = max(round(BOOTSTRAP_BLOCK_DAYS * _bars_per_day(part)), 1)
        name = "holdout median max drawdown within the bootstrap band"
        if len(groups) > 1:
            name += f" ({calendar})"
        group_dd = float(np.median([drawdowns[list(found).index(s)] for s in members]))
        try:
            bands = block_bootstrap(
                series, int(np.median(lengths)) if lengths else 0,
                criteria.get("validation.monte_carlo_runs"), block, seed,
            )  # fmt: skip
            group_band = bands.drawdown(criteria.get("holdout.max_drawdown_percentile"))
            band = group_band if band is None else max(band, group_band)
            checks.append(at_most(name, group_dd if len(groups) > 1 else median_dd, group_band,
                                  "{:.0%}"))  # fmt: skip
        except ValueError:
            failures.append(
                "not enough out-of-sample data or holdout bars to build the drawdown band"
                + (f" ({calendar})" if len(groups) > 1 else "")
            )

    folder = ws.stage_dir(version.idea, version.slug, version.version, "holdout")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sharpe": sharpes, "max_drawdown": drawdowns, "closed_trades": trades},
        index=list(found),
    ).to_csv(folder / "holdout_by_symbol.csv")

    report_path = None
    if report and found:
        with Pool(workers) as pool:
            reports, sections = render_reports(
                [(r, bars, costs[s]) for s, (bars, r) in found.items()], seed, pool
            )
        settings = {
            "Idea": f"{version.idea} v{version.version}: {version.title}",
            "Strategy spec": json.dumps(spec),
            "Data": f"holdout: {describe(criteria, list(found), card_timeframe)} from "
            f"{dev_end(criteria):%Y-%m-%d} to the end of the data, starting flat",
        }
        page = write_report(
            reports, ws.report_dir(version.idea, version.version, "holdout"),
            f"{version.idea} v{version.version} holdout", settings, sections,
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
