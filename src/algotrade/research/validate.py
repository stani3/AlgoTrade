"""``research validate``: walk-forward, deflated Sharpe ratio and the Monte Carlo stake.

Still development data only. The walk-forward re-optimises the pre-registered grid every six
months on the previous two years and trades the next six months; its stitched out-of-sample
record is what gets judged:

* walk-forward efficiency, out-of-sample median Sharpe and share of profitable windows;
* the deflated Sharpe ratio of the median symbol, against every configuration in the trial
  ledger (the whole research programme, not just this idea);
* Davey's Monte Carlo on the out-of-sample trades, which sets the stake.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.engine import BacktestResult, run_backtest
from algotrade.backtest.metrics import periods_per_year
from algotrade.backtest.report import MIN_MC_TRADES
from algotrade.backtest.report_html import render_reports, write_report
from algotrade.parallel import Pool
from algotrade.validation.overfitting import deflated_sharpe, expected_max_sharpe, moments
from algotrade.validation.sizing import choose_stake, closed_trade_returns, limits, trades_per_year
from algotrade.validation.walkforward import WalkForward, walk_forward

from . import results
from .cards import read_card
from .criteria import Criteria, above, at_least, below
from .dedup import spec_hash
from .feasibility import require_ready
from .journal import TrialLedger, VersionState
from .split import dev_end, dev_universe, window
from .workspace import Workspace


def median_symbol_moments(results_by_symbol: dict[str, BacktestResult]) -> dict:
    """Sharpe, length, skew and kurtosis of the median symbol (per bar, not annualised)."""

    rows = [moments(r.returns.to_numpy()) for r in results_by_symbol.values()]
    frame = pd.DataFrame(rows, columns=["sharpe", "observations", "skew", "kurtosis"])
    return {
        "sharpe": float(frame["sharpe"].median()),
        "observations": int(frame["observations"].median()),
        "skew": float(frame["skew"].median()),
        "kurtosis": float(frame["kurtosis"].median()),
    }


def deflation(ws: Workspace, results_by_symbol: dict[str, BacktestResult]) -> dict:
    """Deflated Sharpe ratio of the median symbol against the whole trial ledger.

    Trial Sharpe ratios in the ledger are annualised medians across symbols; they are converted
    to the per-bar scale of this record before their variance is used.
    """

    ppy = float(np.median([periods_per_year(r.ledger.index) for r in results_by_symbol.values()]))
    trials = TrialLedger(ws.trials_path).distinct()
    metric = pd.to_numeric(trials["metric"], errors="coerce").dropna()
    count = len(trials)
    variance = float(metric.var(ddof=1)) / ppy if len(metric) > 1 else 0.0
    stats = median_symbol_moments(results_by_symbol)
    return {
        **stats,
        "trials": count,
        "trial_sharpe_variance_per_bar": variance,
        "expected_max_sharpe_per_bar": expected_max_sharpe(count, variance),
        "periods_per_year": ppy,
        "annualised_sharpe": stats["sharpe"] * float(np.sqrt(ppy)),
        "deflated_sharpe": deflated_sharpe(
            stats["sharpe"], stats["observations"], stats["skew"], stats["kurtosis"], count,
            variance,
        ),
    }  # fmt: skip


def benchmark(bars: pd.DataFrame, index: pd.DatetimeIndex, costs) -> BacktestResult:
    part = bars.loc[index]
    return run_backtest(part, pd.Series(1.0, index=part.index), costs)


def run_walk_forward(
    ws: Workspace, criteria: Criteria, card, pool: Pool | None = None
) -> tuple[WalkForward, dict]:
    universe = dev_universe(ws, criteria, card.timeframe)
    costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
    wf = walk_forward(
        card.spec,
        card.optimise,
        universe,
        costs,
        dev_end(criteria),
        criteria.get("validation.in_sample_years"),
        criteria.get("validation.out_of_sample_months"),
        pool=pool,
    )
    return wf, universe


def validate(
    ws: Workspace,
    criteria: Criteria,
    version: VersionState,
    commit: bool = True,
    report: bool = True,
    seed: int = 0,
    workers: int | None = None,
) -> results.StageResult:
    require_ready(ws, version, "validation", after="feasibility")
    card = read_card(ws.root / version.card_path)
    costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
    with Pool(workers) as pool:
        wf, universe = run_walk_forward(ws, criteria, card, pool)
        reports, sections = (
            render_reports([(r, universe[s], costs) for s, r in wf.results.items()], seed, pool)
            if report and wf.results
            else ([], [])
        )
    v = "validation."
    checks = [
        at_least("walk-forward efficiency (OOS / IS annualised return)", wf.efficiency,
                 criteria.get(v + "min_wf_efficiency"), "{:.0%}"),
        at_least("out-of-sample median Sharpe across symbols", wf.oos_median_sharpe,
                 criteria.get(v + "oos_min_median_sharpe")),
        at_least("share of out-of-sample windows profitable", wf.profitable_windows,
                 criteria.get(v + "oos_min_profitable_windows"), "{:.0%}"),
    ]  # fmt: skip
    failures = []
    if not wf.results:
        failures.append("no walk-forward window had enough in-sample data")
        dsr = {"deflated_sharpe": 0.0, "trials": TrialLedger(ws.trials_path).count()}
    else:
        dsr = deflation(ws, wf.results)
    checks.append(
        at_least(
            "deflated Sharpe ratio (probability the edge is real)",
            dsr["deflated_sharpe"],
            criteria.get(v + "min_deflated_sharpe"),
            "{:.2f}",
        )
    )

    returns = closed_trade_returns(wf.results) if wf.results else np.array([])
    sizing = None
    davey = limits(criteria)
    if len(returns) < MIN_MC_TRADES:
        failures.append(
            f"only {len(returns)} out-of-sample trades; Monte Carlo needs {MIN_MC_TRADES}"
        )
    else:
        sizing = choose_stake(
            returns, trades_per_year(wf.results), criteria.get(v + "stake_multipliers"),
            criteria.get(v + "monte_carlo_runs"), criteria.get(v + "ruin"), davey, seed,
        )  # fmt: skip
        size = sizing.stake if sizing.stake is not None else 1.0
        row = sizing.table.loc[size]
        checks += [
            below(f"Monte Carlo risk of ruin at {size:g}x", float(row["risk_of_ruin"]),
                  davey["risk_of_ruin"], "{:.0%}"),
            below(f"Monte Carlo median max drawdown at {size:g}x", float(row["median_max_dd"]),
                  davey["median_max_dd"], "{:.0%}"),
            above(f"Monte Carlo return / drawdown at {size:g}x", float(row["return_dd"]),
                  davey["return_dd"]),
        ]  # fmt: skip

    folder = ws.stage_dir(version.idea, version.slug, version.version, "validation")
    folder.mkdir(parents=True, exist_ok=True)
    wf.windows.to_csv(folder / "windows.csv", index=False)
    pd.DataFrame(
        {
            symbol: {
                "sharpe": float(wf.sharpes[symbol]),
                "total_return": float(r.equity.iloc[-1] - 1.0),
                "max_drawdown": float((r.equity / r.equity.cummax() - 1.0).min()),
                "closed_trades": int((~r.trades["open"].astype(bool)).sum()),
            }
            for symbol, r in wf.results.items()
        }
    ).T.to_csv(folder / "oos_by_symbol.csv")
    if sizing is not None:
        sizing.table.to_csv(folder / "monte_carlo.csv")
    pd.DataFrame({"return": returns}).to_csv(folder / "oos_trade_returns.csv", index=False)
    if wf.results:
        bar_returns = pd.DataFrame({s: r.returns for s, r in wf.results.items()}).astype("float32")
        bar_returns.index.name = "time"
        bar_returns.to_parquet(folder / "oos_returns.parquet", compression="zstd")

    report_path = None
    if report and wf.results:
        settings = {
            "Idea": f"{version.idea} v{version.version}: {version.title}",
            "Walk-forward": f"{criteria.get(v + 'in_sample_years')} years in-sample, "
            f"{criteria.get(v + 'out_of_sample_months')} months out-of-sample, rolling; "
            "each window re-optimised on the pre-registered grid",
            "Base spec": json.dumps(card.spec),
            "Data": f"{criteria.get('data.exchange')} {card.timeframe}, out-of-sample pieces of "
            f"the development period (before {dev_end(criteria):%Y-%m-%d})",
        }
        page = write_report(
            reports, ws.report_dir(version.idea, version.version, "validation"),
            f"{version.idea} v{version.version} walk-forward out-of-sample", settings, sections,
        )  # fmt: skip
        report_path = ws.relative(page)

    windows = [window(bars, symbol, card.timeframe) for symbol, bars in universe.items()]
    result = results.StageResult(
        version.idea,
        version.version,
        "validation",
        checks,
        failures=failures,
        metrics={
            "walk-forward windows": len(wf.windows),
            "out-of-sample annualised Sharpe of the median symbol": dsr.get(
                "annualised_sharpe", 0.0
            ),
            "trials counted (distinct configurations)": dsr["trials"],
            "stake (largest size meeting Davey's goals)": sizing.stake
            if sizing and sizing.stake
            else 0.0,
        },
        trials=0,
        notes=[
            "Walk-forward windows:\n\n" + _windows_markdown(wf.windows),
            *(["Monte Carlo by size:\n\n" + _mc_markdown(sizing.table)] if sizing else []),
        ],
        provenance=results.provenance(ws, criteria, version, spec_hash(card.spec), windows),
        report=report_path,
        extra={
            "stake": sizing.stake if sizing else None,
            "deflated_sharpe": dsr,
            "windows": json.loads(wf.windows.to_json(orient="records", date_format="iso")),
        },
    )
    results.record(ws, version, result, commit=commit)
    return result


def _windows_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no windows)"
    shown = frame.copy()
    for column in ("is_start", "oos_start", "oos_end"):
        shown[column] = pd.to_datetime(shown[column]).dt.strftime("%Y-%m-%d")
    shown = shown.round(3)
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "|" + "---|" * len(shown.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _mc_markdown(table: pd.DataFrame) -> str:
    shown = table.round(3).reset_index()
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "---|" * len(shown.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, rule, *rows])
