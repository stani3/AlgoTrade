"""Paper trade a frozen strategy on an exchange TESTNET with NautilusTrader (you start it).

Before anything runs it checks that the strategy is frozen and that incubation was started
(`research incubate`, after you approved it). Testnet API keys are read from the environment;
put them in `.env` yourself:

    Binance USDT-M testnet: BINANCE_FUTURES_TESTNET_API_KEY, BINANCE_FUTURES_TESTNET_API_SECRET
    Bybit testnet:          BYBIT_TESTNET_API_KEY, BYBIT_TESTNET_API_SECRET

Every fill is logged to research/ideas/<id>-<slug>/v<n>/incubation/fills.csv for
`research incubation-report`. There is no mainnet option: this script cannot trade real money.

Examples:
    python -m scripts.paper_trade i002 --dry-run     # show what would run, connect nothing
    python -m scripts.paper_trade i002 --parity      # adapter vs our engine on recent real bars
    python -m scripts.paper_trade i002               # run on Binance testnet until stopped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from algotrade.research.criteria import load_criteria
from algotrade.research.incubation import frozen_strategy, incubation_dir
from algotrade.research.registry import Refused, find_version
from algotrade.research.split import load_full_bars
from algotrade.research.workspace import Workspace

PARITY_BARS = 1500


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from ``.env`` into the environment without printing anything."""

    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parity(ws: Workspace, frozen: dict, bars_per_symbol: int = PARITY_BARS) -> bool:
    """Adapter against our engine on the latest real bars of every symbol (zero costs)."""

    from algotrade.live.parity import run

    criteria = load_criteria(ws.criteria_path)
    worst = 0.0
    for symbol in frozen["symbols"]:
        bars = load_full_bars(ws, criteria, symbol, frozen["timeframe"]).iloc[-bars_per_symbol:]
        result = run(frozen["spec"], bars, frozen["timeframe"])
        gap = result.equity_gap()
        worst = max(worst, gap)
        print(f"{symbol}: final equity ours {result.ours.equity.iloc[-1]:.4f}, "
              f"NautilusTrader {result.equity.iloc[-1]:.4f}, largest gap {gap:.2e}")  # fmt: skip
    ok = worst < 0.01
    print(f"PARITY {'OK' if ok else 'FAILED'}: largest equity gap {worst:.2e} (allowed 1e-2 on real "
          "bars, whose gaps between one close and the next open make bracket fills differ slightly)")  # fmt: skip
    return ok


def run_node(node) -> None:
    """Connect and trade until stopped (Ctrl+C); only ever started by the user."""

    try:
        node.build()
        node.run()
    finally:
        node.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("idea")
    parser.add_argument("--version", type=int)
    parser.add_argument("--venue", choices=["binance", "bybit"], default="binance")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true", help="Print the set-up, connect nothing")
    parser.add_argument("--parity", action="store_true", help="Check the adapter on real bars")
    args = parser.parse_args(argv)

    ws = Workspace(args.root.resolve())
    try:
        version = find_version(ws, args.idea, args.version)
        frozen = frozen_strategy(ws, version)
    except Refused as error:
        print(f"REFUSED: {error}")
        return 1
    if args.parity:
        return 0 if parity(ws, frozen) else 1

    from algotrade.live.node import build_node, missing_keys

    fills = incubation_dir(ws, version) / "fills.csv"
    print(f"{version.idea} v{version.version} '{version.title}': {frozen['spec']}")
    print(f"{args.venue} TESTNET, {frozen['timeframe']} bars, {', '.join(frozen['symbols'])}; "
          f"stake {frozen.get('stake')}, equity split equally; fills -> {ws.relative(fills)}")  # fmt: skip
    if version.incubation_start is None:
        print("REFUSED: incubation has not been started; run `research incubate` after approval")
        return 1
    load_dotenv(ws.root / ".env")
    missing = missing_keys(args.venue)
    if missing:
        print(f"Set these in .env first (testnet keys only): {', '.join(missing)}")
        if not args.dry_run:
            return 1
    if args.dry_run:
        print("Dry run: nothing was started.")
        return 0
    run_node(build_node(frozen, args.venue, str(fills)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
