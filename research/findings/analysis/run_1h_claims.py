"""Core claims of the 1h ideas, with honest statistics (dev data only, registered card specs).

Trades on different coins at the same time are correlated (i008 trades all ten coins at 10:00
New York), so confidence intervals come from a bootstrap that resamples whole DAYS of trades,
not single trades. "Gross" is the price move only (zero fees, slippage and funding).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS, ZERO_COSTS
from algotrade.backtest.runner import backtest
from algotrade.research.cards import read_card
from algotrade.research.criteria import load_criteria
from algotrade.research.split import dev_universe
from algotrade.research.workspace import Workspace
from algotrade.strategies import from_spec

ws = Workspace(Path.cwd())
criteria = load_criteria(ws.criteria_path)
costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
universe = dev_universe(ws, criteria, "1h")
rng = np.random.default_rng(0)


def trades(spec, cost_model) -> pd.DataFrame:
    parts = []
    for symbol, bars in universe.items():
        t = backtest(from_spec(spec), bars, cost_model).trades
        t = t[~t["open"].astype(bool)].assign(symbol=symbol)
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t["day"] = pd.to_datetime(t["entry"]).dt.floor("D")
    t["year"] = t["day"].dt.year
    return t


def day_bootstrap(values: pd.Series, days: pd.Series, runs: int = 2000) -> tuple[float, float, float]:
    """Mean per trade with a 95% interval from resampling whole days."""

    by_day = pd.DataFrame({"v": values.to_numpy(), "d": days.to_numpy()}).groupby("d")["v"]
    sums, counts = by_day.sum().to_numpy(), by_day.count().to_numpy()
    picks = rng.integers(0, len(sums), size=(runs, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(sums.sum() / counts.sum()), *np.quantile(means, [0.025, 0.975])


def bp(x: float) -> str:
    return f"{x * 1e4:+.1f}"


ideas = sys.argv[1:] or ["i005-funding-settlement-rebound", "i006-bollinger-squeeze-breakout",
                         "i007-dual-thrust-intraday-breakout", "i008-us-open-session-momentum"]
for folder in ideas:
    card = read_card(ws.research / "ideas" / folder / "v1/idea.md")
    gross, net = trades(card.spec, ZERO_COSTS), trades(card.spec, costs)
    g, glo, ghi = day_bootstrap(gross["return"], gross["day"])
    n, nlo, nhi = day_bootstrap(net["return"], net["day"])
    print(f"\n=== {folder} (card spec, {len(gross)} trades on {gross['day'].nunique()} days) ===")
    print(f"gross per trade {bp(g)} bp  95% CI [{bp(glo)}, {bp(ghi)}]")
    print(f"net per trade   {bp(n)} bp  95% CI [{bp(nlo)}, {bp(nhi)}]")
    print(f"break-even cost per side ~ {g * 1e4 / 2:.1f} bp (we charge 8; funding adds "
          f"{(g - n) * 1e4 - 16:+.1f} bp a trade on top of 16 bp fees+slippage)")
    by_year = gross.groupby("year")["return"].agg(["count", "mean"])
    by_year["mean"] = (by_year["mean"] * 1e4).round(1)
    print("gross bp per trade by year:", by_year["mean"].to_dict(), "| trades:", by_year["count"].to_dict())
    ex = gross[~gross["year"].isin([2020, 2021])]
    e, elo, ehi = day_bootstrap(ex["return"], ex["day"])
    print(f"gross per trade 2022-2025 only {bp(e)} bp  95% CI [{bp(elo)}, {bp(ehi)}]")
    sym = gross.groupby("symbol")["return"].mean() * 1e4
    print(f"coins with positive gross: {(sym > 0).sum()}/10; best {sym.idxmax()} {sym.max():+.1f} bp, "
          f"worst {sym.idxmin()} {sym.min():+.1f} bp")
    side = gross.groupby("direction")["return"].mean() * 1e4
    print("gross bp per trade by side:", side.round(1).to_dict())
    if "exit_reason" in gross:
        decided = gross[gross["exit_reason"].isin(["stop", "target"])]
        hit = (decided["exit_reason"] == "target").astype(float)
        h, hlo, hhi = day_bootstrap(hit, decided["day"])
        ratio = card.spec["stop_atr"] / (card.spec["stop_atr"] + card.spec["target_atr"])
        print(f"target hit rate {h:.3f}  95% CI [{hlo:.3f}, {hhi:.3f}] vs {ratio:.3f} for a random "
              f"direction (stop {card.spec['stop_atr']} / target {card.spec['target_atr']} ATR); "
              f"{len(decided)} of {len(gross)} trades ended at stop or target")
