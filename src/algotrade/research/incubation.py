"""Incubation (Davey): watch a frozen strategy on data that did not exist when it was built.

``research incubate`` (only after the user approved) records the start. From then on two things
are tracked:

* **Strategy quality** - a shadow forward test: the frozen spec run by our own engine on the real
  (mainnet) bars downloaded since the start, compared with what a period that long should look
  like (block-bootstrapped from the walk-forward's out-of-sample bar returns). Testnet prices are
  not a real market, so they are not used for this.
* **Execution** - the testnet paper-trading node logs every fill with the price the strategy
  expected; slippage is compared with the cost model.

``research incubation-report`` scores both. Until ``incubation.min_days`` and
``incubation.min_trades`` are reached the verdict is "continue" (unless something already broke
its band). Passing incubation never means going live: that is always the user's own decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
from algotrade.backtest.report import build_report
from algotrade.backtest.report_html import write_report
from algotrade.validation.bands import block_bootstrap
from algotrade.validation.walkforward import out_of_sample

from . import results, vcs
from .criteria import Check, Criteria, at_least, at_most
from .dedup import spec_hash
from .holdout import BOOTSTRAP_BLOCK_DAYS
from .index import write_index
from .journal import Journal, VersionState, now
from .registry import Refused
from .split import load_full_bars, window
from .validate import benchmark
from .workspace import Workspace

FILL_COLUMNS = ["ts", "symbol", "side", "quantity", "price", "expected_price"]


def frozen_strategy(ws: Workspace, version: VersionState) -> dict:
    path = ws.version_dir(version.idea, version.slug, version.version) / "frozen.json"
    if version.frozen is None or not path.exists():
        raise Refused(f"{version.idea} v{version.version} is not frozen")
    return json.loads(path.read_text(encoding="utf-8"))


def incubation_dir(ws: Workspace, version: VersionState) -> Path:
    return ws.stage_dir(version.idea, version.slug, version.version, "incubation")


def start(
    ws: Workspace, version: VersionState, when: str | None = None, commit: bool = True
) -> dict:
    """Record that incubation started (the user approved it). ``when`` defaults to now."""

    frozen = frozen_strategy(ws, version)
    if version.incubation_start is not None:
        raise Refused(
            f"{version.idea} v{version.version} has been incubating since {version.incubation_start}"
        )
    moment = pd.Timestamp(when or now())
    moment = moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
    record = {"start": moment.isoformat(), "spec": frozen["spec"], "timeframe": frozen["timeframe"],
              "stake": frozen["stake"], "symbols": frozen["symbols"], "tag": frozen["tag"]}  # fmt: skip
    folder = incubation_dir(ws, version)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "started.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    Journal(ws.journal_path).append(
        "incubation_started", idea=version.idea, version=version.version, start=record["start"]
    )
    if commit:
        vcs.commit(
            ws.root,
            [folder / "started.json", ws.journal_path, write_index(ws)],
            f"research({version.idea}-{version.slug} v{version.version}): incubation started",
        )
    return record


def read_fills(folder: Path) -> pd.DataFrame:
    path = folder / "fills.csv"
    if not path.exists():
        return pd.DataFrame(columns=FILL_COLUMNS)
    return pd.read_csv(path)


def slippage_bps(fills: pd.DataFrame) -> pd.Series:
    """Cost of each fill against the price the strategy expected, in basis points (positive =
    worse than expected for that side)."""

    side = np.sign(fills["side"].astype(float))
    return side * (fills["price"] - fills["expected_price"]) / fills["expected_price"] * 10_000


@dataclass
class IncubationReport:
    result: results.StageResult
    status: str  # continue | pass | fail
    days: float
    trades: float
    extra: dict = field(default_factory=dict)


def incubation_report(
    ws: Workspace,
    criteria: Criteria,
    version: VersionState,
    commit: bool = True,
    report: bool = True,
    seed: int = 0,
) -> IncubationReport:
    if version.incubation_start is None:
        raise Refused(f"{version.idea} v{version.version} is not incubating")
    if "incubation" in version.stages:
        raise Refused(f"{version.idea} v{version.version} already has an incubation verdict")
    frozen = frozen_strategy(ws, version)
    started = pd.Timestamp(version.incubation_start)
    costs = EXCHANGE_COSTS[frozen["exchange"]]
    forward = {}
    for symbol in frozen["symbols"]:
        bars = load_full_bars(ws, criteria, symbol, frozen["timeframe"])
        if (bars.index >= started).sum() > 1:
            forward[symbol] = (bars, out_of_sample(frozen["spec"], bars, started, costs))

    inc = "incubation."
    checks: list[Check] = []
    notes = []
    days, trades = 0.0, 0.0
    if forward:
        last = max(result.ledger.index[-1] for _, result in forward.values())
        days = (last - started) / pd.Timedelta(days=1)
        trades = float(
            np.median([(~r.trades["open"].astype(bool)).sum() for _, r in forward.values()])
        )
        totals = [float(r.equity.iloc[-1] - 1.0) for _, r in forward.values()]
        drawdowns = [
            float(-(r.equity / r.equity.cummax() - 1.0).min()) for _, r in forward.values()
        ]
        sharpes = [
            sharpe_ratio(r.returns, periods_per_year(r.ledger.index)) for _, r in forward.values()
        ]
        validation = ws.stage_dir(version.idea, version.slug, version.version, "validation")
        oos = pd.read_parquet(validation / "oos_returns.parquet")
        per_day = float(
            np.median([periods_per_year(r.ledger.index) / 365.25 for _, r in forward.values()])
        )
        bands = block_bootstrap(
            [oos[c].dropna().to_numpy() for c in oos.columns],
            int(np.median([len(r.ledger) for _, r in forward.values()])),
            criteria.get("validation.monte_carlo_runs"),
            max(round(BOOTSTRAP_BLOCK_DAYS * per_day), 1),
            seed,
        )
        checks += [
            at_least("forward median return vs the 5th percentile of same-length paths",
                     float(np.median(totals)), bands.total_return(criteria.get(inc + "min_return_percentile")), "{:.1%}"),
            at_most("forward median max drawdown vs the 95th percentile of same-length paths",
                    float(np.median(drawdowns)), bands.drawdown(criteria.get(inc + "max_drawdown_percentile")), "{:.1%}"),
        ]  # fmt: skip
        notes.append(f"Shadow forward test on mainnet bars up to {last:%Y-%m-%d %H:%M} UTC; "
                     f"median forward Sharpe {float(np.median(sharpes)):.2f}.")  # fmt: skip
    else:
        notes.append("No bars after the start yet: run scripts.download_data to update the data.")

    folder = incubation_dir(ws, version)
    folder.mkdir(parents=True, exist_ok=True)
    fills = read_fills(folder)
    if len(fills):
        slip = float(slippage_bps(fills).median())
        limit = costs.slippage_bps * criteria.get(inc + "max_slippage_ratio")
        checks.append(at_most("median testnet slippage per fill (bps)", slip, limit, "{:.1f}"))
        notes.append(f"{len(fills)} testnet fills logged.")
    else:
        notes.append("No testnet fills logged yet (fills.csv); execution is not scored.")

    enough = days >= criteria.get(inc + "min_days") and trades >= criteria.get(inc + "min_trades")
    broken = any(not c.passed for c in checks)
    status = "fail" if broken else ("pass" if enough else "continue")
    if not enough and not broken:
        notes.append(
            f"Continue: {days:.0f} of {criteria.get(inc + 'min_days')} days and {trades:.0f} of "
            f"{criteria.get(inc + 'min_trades')} trades so far."
        )

    report_path = None
    if report and forward:
        reports = [build_report(r, benchmark=benchmark(b, r.ledger.index, costs), seed=seed)
                   for b, r in forward.values()]  # fmt: skip
        settings = {"Idea": f"{version.idea} v{version.version}: {version.title}",
                    "Frozen spec": json.dumps(frozen["spec"]),
                    "Data": f"shadow forward test from {started:%Y-%m-%d} on mainnet bars"}  # fmt: skip
        page = write_report(reports, ws.report_dir(version.idea, version.version, "incubation"),
                            f"{version.idea} v{version.version} incubation", settings)  # fmt: skip
        report_path = ws.relative(page)

    windows = [
        window(b[b.index >= started], s, frozen["timeframe"]) for s, (b, _) in forward.items()
    ]
    result = results.StageResult(
        version.idea, version.version, "incubation", checks,
        metrics={"days since start": days, "median closed trades per symbol": trades},
        notes=notes, report=report_path,
        provenance=results.provenance(ws, criteria, version, spec_hash(frozen["spec"]), windows),
        extra={"status": status},
    )  # fmt: skip
    if status == "continue":
        results.write(ws, version, result)
        if commit:
            subject = (f"research({version.idea}-{version.slug} v{version.version}): incubation "
                       f"check-in - continue ({days:.0f} days, {trades:.0f} trades)")  # fmt: skip
            vcs.commit(ws.root, [folder], subject)
    else:
        results.record(ws, version, result, commit=commit)
    return IncubationReport(result, status, days, trades)
