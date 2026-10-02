"""Backtest one strategy across symbols, net of fees, slippage and funding.

Examples:
    python -m scripts.run_backtest --strategy donchian_breakout --param entry=55 --param exit=20
    python -m scripts.run_backtest --strategy ewmac --vol-target 0.25 --symbols BTC,ETH,SOL
    python -m scripts.run_backtest --spec specs/carver_trend.json --timeframe 1d
    python -m scripts.run_backtest --spec specs/breakout_bracket.json --timeframe 4h --report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.engine import run_backtest
from algotrade.backtest.metrics import summarize
from algotrade.backtest.report import build_report
from algotrade.backtest.report_html import write_report
from algotrade.backtest.runner import backtest, cost_model, load_bars
from algotrade.strategies import (
    STRATEGIES,
    BracketStrategy,
    Strategy,
    VolTarget,
    from_spec,
    to_spec,
)

COLUMNS = {
    "cagr": "{:+.1%}",
    "sharpe": "{:.2f}",
    "max_drawdown": "{:.1%}",
    "ann_vol": "{:.0%}",
    "trades": "{:.0f}",
    "win_rate": "{:.0%}",
    "profit_factor": "{:.2f}",
    "exposure": "{:.0%}",
    "cost_drag_per_year": "{:.1%}",
    "funding_drag_per_year": "{:+.1%}",
}


def parse_value(raw: str) -> object:
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return {"true": True, "false": False}.get(raw.lower(), raw)


def build_strategy(args: argparse.Namespace) -> Strategy:
    if args.spec:
        strategy = from_spec(args.spec)
    else:
        params = {}
        for pair in args.param:
            key, _, raw = pair.partition("=")
            params[key] = parse_value(raw)
        strategy = STRATEGIES[args.strategy](**params)
    if args.vol_target and isinstance(strategy, BracketStrategy):
        raise SystemExit(f"{strategy.name} sizes its own trades; drop --vol-target")
    if args.vol_target and not isinstance(strategy, VolTarget):
        strategy = VolTarget(strategy, annual_vol=args.vol_target, max_leverage=args.max_leverage)
    return strategy


def format_row(stats: dict) -> dict:
    return {key: fmt.format(stats[key]) for key, fmt in COLUMNS.items()}


def plot_equity(curves: dict[str, pd.Series], output: Path, title: str) -> None:
    from matplotlib.figure import Figure

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()
    for label, equity in curves.items():
        ax.plot(equity.index, equity.values, label=label, linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log, start = 1)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main(args: argparse.Namespace) -> None:
    strategy = build_strategy(args)
    costs = cost_model(args.exchange, args.fee_bps, args.slippage_bps, not args.no_funding)
    rows, curves, notes, reports = {}, {}, {}, []
    for base in args.symbols.split(","):
        bars = load_bars(args.exchange, base, args.timeframe, args.start, args.end)
        result = backtest(strategy, bars, costs, args.max_leverage)
        name = bars.attrs["symbol"]
        rows[name] = format_row(summarize(result))
        curves[name] = result.equity
        if "exit_reason" in result.trades:
            reasons = result.trades["exit_reason"].value_counts().to_dict()
            killed = result.meta.get("killed_at")
            notes[name] = f"exits {reasons}" + (
                f" | kill switch at {killed:%Y-%m-%d}" if killed else ""
            )
        hold = None
        if not args.no_benchmark:
            hold = run_backtest(bars, pd.Series(1.0, index=bars.index), costs)
            rows[f"  {name} buy&hold"] = format_row(summarize(hold))
        if args.report:
            report = build_report(result, args.capital, hold, args.mc_runs, args.mc_ruin)
            reports.append(report)
            if report.monte_carlo is not None:
                checks = report.monte_carlo.checks()
                verdict = ", ".join(
                    f"{label.lower()} {value:.2f}"
                    if "Return" in label
                    else f"{label.lower()} {value:.0%}"
                    for label, value, _, _ in checks
                )
                passed = sum(ok for *_, ok in checks)
                mc_note = f"Monte Carlo: {verdict} -> {passed}/3 Davey checks"
                notes[name] = f"{notes[name]} | {mc_note}" if name in notes else mc_note

    print(json.dumps(to_spec(strategy)))
    print(
        f"{args.exchange} {args.timeframe} | fee {costs.fee_bps}bps + slippage "
        f"{costs.slippage_bps}bps | funding {'on' if costs.include_funding else 'off'}"
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).T.to_string())
    for name, note in notes.items():
        print(f"{name}: {note}")
    if args.plot:
        plot_equity(curves, args.plot, f"{strategy} ({args.timeframe}, net of costs)")
        print(f"Saved chart to {args.plot}")
    if args.report:
        spec = to_spec(strategy)
        folder = (
            Path("reports") / f"{spec['type']}_{args.timeframe}"
            if args.report == "auto"
            else Path(args.report)
        )
        settings = {
            "Strategy spec": json.dumps(spec),
            "Market": f"{args.exchange} perpetuals, {args.timeframe} bars",
            "Costs": f"fee {costs.fee_bps} bps + slippage {costs.slippage_bps} bps per fill, "
            f"funding {'on' if costs.include_funding else 'off'}",
            "Sizing": f"{args.max_leverage:g}x max leverage, starting capital ${args.capital:,.0f}",
            "Monte Carlo": f"{args.mc_runs:,} runs of one year, ruin = {args.mc_ruin:.0%} loss",
        }
        page = write_report(reports, folder, str(strategy), settings)
        print(f"Saved performance report to {page}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="ma_crossover")
    parser.add_argument("--param", action="append", default=[], help="Strategy parameter key=value")
    parser.add_argument("--spec", help="Strategy spec: JSON string or path to a .json file")
    parser.add_argument("--vol-target", type=float, help="Wrap in VolTarget, e.g. 0.25 = 25%%/yr")
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--exchange", choices=sorted(EXCHANGE_COSTS), default="binanceusdm")
    parser.add_argument("--symbols", default="BTC,ETH")
    parser.add_argument("--timeframe", choices=["4h", "1d"], default="1d")
    parser.add_argument("--start", help="Inclusive start date, e.g. 2021-01-01")
    parser.add_argument("--end", help="Exclusive end date")
    parser.add_argument("--fee-bps", type=float)
    parser.add_argument("--slippage-bps", type=float)
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip buy-and-hold rows")
    parser.add_argument("--plot", type=Path, help="Save an equity chart, e.g. reports/ma.png")
    parser.add_argument(
        "--report",
        nargs="?",
        const="auto",
        help="Write an HTML performance report (default folder reports/<strategy>_<timeframe>/)",
    )
    parser.add_argument("--capital", type=float, default=10_000.0, help="Report starting capital")
    parser.add_argument("--mc-runs", type=int, default=2500, help="Monte Carlo runs")
    parser.add_argument(
        "--mc-ruin", type=float, default=0.5, help="Monte Carlo ruin = this loss from the start"
    )
    main(parser.parse_args())
