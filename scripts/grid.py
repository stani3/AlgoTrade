"""Parameter grid for any strategy, scored across the whole symbol universe.

Every combination is one trial; the total is printed so you know how many you ran. Look for a
broad plateau of good cells in the heatmap, not the single best one.

Example:
    python -m scripts.grid --strategy breakout_bracket --grid stop_atr=1,1.5,2,3,4
        --grid target_atr=2,3,4,6,8 --grid atr_length=14,30 --timeframe 4h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.backtest.runner import cost_model, load_bars
from algotrade.strategies import STRATEGIES, from_spec, to_spec
from algotrade.validation.optimize import run_grid
from scripts.download_data import DEFAULT_UNIVERSE
from scripts.run_backtest import parse_value

FORMATS = {
    "median_sharpe": "{:.2f}",
    "min_sharpe": "{:.2f}",
    "positive": "{:.0%}",
    "median_cagr": "{:+.1%}",
    "worst_dd": "{:.0%}",
    "trades_per_year": "{:.1f}",
    "cost_drag": "{:.1%}",
    "killed": "{:.0%}",
}


def parse_grid(items: list[str]) -> dict[str, list]:
    grid = {}
    for item in items:
        key, _, raw = item.partition("=")
        grid[key] = [parse_value(v) for v in raw.split(",") if v]
    return grid


def base_spec(args: argparse.Namespace) -> dict:
    if args.spec:
        return to_spec(from_spec(args.spec))
    params = {}
    for pair in args.param:
        key, _, raw = pair.partition("=")
        params[key] = parse_value(raw)
    return to_spec(STRATEGIES[args.strategy](**params))


def plot_heatmaps(board: pd.DataFrame, keys: list[str], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    row_key, col_key = keys[0], keys[1]
    panel_key = keys[2] if len(keys) > 2 else None
    panels = sorted(board[panel_key].unique()) if panel_key else [None]
    values = board["median_sharpe"]
    limit = max(abs(values.min()), abs(values.max()), 0.1)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
    for ax, panel in zip(axes[0], panels, strict=True):
        part = board if panel is None else board[board[panel_key] == panel]
        table = part.pivot_table(index=row_key, columns=col_key, values="median_sharpe")
        image = ax.imshow(table.values, cmap="RdYlGn", vmin=-limit, vmax=limit, origin="lower")
        ax.set_xticks(range(len(table.columns)), [f"{c:g}" for c in table.columns])
        ax.set_yticks(range(len(table.index)), [f"{r:g}" for r in table.index])
        ax.set_xlabel(col_key)
        ax.set_ylabel(row_key)
        for (r, c), value in np.ndenumerate(table.values):
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{panel_key} = {panel:g}" if panel_key else "median Sharpe", fontsize=10)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="median Sharpe across symbols")
    fig.suptitle(title, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    base = base_spec(args)
    grid = parse_grid(args.grid)
    if not grid:
        raise SystemExit("give at least one --grid key=v1,v2,...")
    costs = cost_model(args.exchange, args.fee_bps, args.slippage_bps)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    universe = {
        s: load_bars(args.exchange, s, args.timeframe, args.start, args.end) for s in symbols
    }

    board = run_grid(base, grid, universe, costs, args.max_leverage)
    combos = len(board)

    keys = list(grid)
    board = board.drop(columns=["spec", "spec_hash", "median_trades"])
    board = board.sort_values("median_sharpe", ascending=False)
    trials = combos * len(universe)
    print(json.dumps(base))
    print(
        f"{combos} combinations x {len(universe)} symbols = {trials} backtests | "
        f"{args.exchange} {args.timeframe} | killed = share of symbols where the kill switch fired"
    )
    shown = board.copy()
    for column, fmt in FORMATS.items():
        shown[column] = shown[column].map(fmt.format)
    with pd.option_context("display.width", 220, "display.max_rows", args.top):
        print(shown.head(args.top).to_string(index=False))

    stem = f"grid_{base['type']}_{args.timeframe}"
    args.output.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.output / f"{stem}.csv", index=False)
    print(f"Saved {args.output / f'{stem}.csv'}")
    if len(keys) >= 2:
        chart = args.output / f"{stem}.png"
        plot_heatmaps(board, keys, chart, f"{base['type']} | {args.timeframe} | {trials} backtests")
        print(f"Saved {chart}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="breakout_bracket")
    parser.add_argument("--spec", help="Base strategy spec (JSON string or file)")
    parser.add_argument("--param", action="append", default=[], help="Fixed parameter key=value")
    parser.add_argument(
        "--grid", action="append", default=[], help="key=v1,v2,... (2-3 keys for heatmaps)"
    )
    parser.add_argument("--exchange", choices=sorted(EXCHANGE_COSTS), default="binanceusdm")
    parser.add_argument("--symbols", default=DEFAULT_UNIVERSE)
    parser.add_argument("--timeframe", choices=["4h", "1d"], default="4h")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--fee-bps", type=float)
    parser.add_argument("--slippage-bps", type=float)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=15, help="Rows to print")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    main(parser.parse_args())
