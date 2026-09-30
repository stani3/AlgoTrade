"""Run every catalogued rule (default parameters) plus any spec files across a symbol universe.

Ranks strategies by how consistently they work across symbols, not by their best symbol.

Examples:
    python -m scripts.scan --timeframe 1d
    python -m scripts.scan --timeframe 4h --vol-target 0.25 --specs specs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.metrics import summarize
from algotrade.backtest.runner import backtest, cost_model, load_bars
from algotrade.strategies import RULES, Strategy, VolTarget, from_spec
from scripts.download_data import DEFAULT_UNIVERSE


def candidates(spec_dir: Path | None) -> dict[str, Strategy]:
    found = {cls.name: cls() for cls in RULES}
    if spec_dir:
        for path in sorted(spec_dir.glob("*.json")):
            found[f"spec:{path.stem}"] = from_spec(path)
    return found


def aggregate(per_symbol: pd.DataFrame) -> dict[str, float]:
    return {
        "median_sharpe": per_symbol["sharpe"].median(),
        "min_sharpe": per_symbol["sharpe"].min(),
        "positive": (per_symbol["sharpe"] > 0).mean(),
        "median_cagr": per_symbol["cagr"].median(),
        "worst_dd": per_symbol["max_drawdown"].min(),
        "trades_per_year": per_symbol["trades_per_year"].median(),
        "cost_drag": per_symbol["cost_drag_per_year"].median(),
    }


def main(args: argparse.Namespace) -> None:
    costs = cost_model(args.exchange, args.fee_bps, args.slippage_bps)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    universe = {
        s: load_bars(args.exchange, s, args.timeframe, args.start, args.end) for s in symbols
    }

    summary, detail = {}, []
    for label, strategy in candidates(args.specs).items():
        if args.vol_target and not isinstance(strategy, VolTarget):
            strategy = VolTarget(strategy, annual_vol=args.vol_target)
        stats = {}
        for symbol, bars in universe.items():
            stats[symbol] = summarize(backtest(strategy, bars, costs))
        per_symbol = pd.DataFrame(stats).T
        summary[label] = aggregate(per_symbol)
        detail.append(per_symbol.assign(strategy=label))

    board = pd.DataFrame(summary).T.sort_values("median_sharpe", ascending=False)
    formats = {
        "median_sharpe": "{:.2f}",
        "min_sharpe": "{:.2f}",
        "positive": "{:.0%}",
        "median_cagr": "{:+.1%}",
        "worst_dd": "{:.0%}",
        "trades_per_year": "{:.0f}",
        "cost_drag": "{:.1%}",
    }
    shown = board.apply(lambda col: col.map(formats[col.name].format))
    sizing = f"vol target {args.vol_target:.0%}" if args.vol_target else "fixed 1x exposure"
    print(
        f"{len(board)} strategies x {len(universe)} symbols | {args.exchange} {args.timeframe} "
        f"| {sizing} | positive = share of symbols with Sharpe > 0"
    )
    with pd.option_context("display.width", 200):
        print(shown.to_string())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(detail).to_csv(args.output)
    print(f"Per-symbol results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--exchange", choices=sorted(EXCHANGE_COSTS), default="binanceusdm")
    parser.add_argument("--symbols", default=DEFAULT_UNIVERSE)
    parser.add_argument("--timeframe", choices=["4h", "1d"], default="1d")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--vol-target", type=float)
    parser.add_argument("--fee-bps", type=float)
    parser.add_argument("--slippage-bps", type=float)
    parser.add_argument("--specs", type=Path, help="Folder of .json specs to include")
    parser.add_argument("--output", type=Path, default=Path("reports/scan.csv"))
    main(parser.parse_args())
