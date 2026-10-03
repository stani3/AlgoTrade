"""Patterns in i003 v1's short trades (dev data only, the gate's own trades; no new backtests).

Every label is taken from the close of the day BEFORE the trade's first day, so nothing a
trade could not have known at entry is used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from algotrade.research.criteria import load_criteria
from algotrade.research.split import dev_universe
from algotrade.research.workspace import Workspace

ws = Workspace(Path.cwd())
criteria = load_criteria(ws.criteria_path)
trades = pd.read_csv(
    ws.research / "ideas/i003-weekly-horizon-time-series-momentum/v1/feasibility/diagnostics_trades.csv",
    parse_dates=["entry", "exit"],
)
trades = trades[~trades["open"]]
universe = dev_universe(ws, criteria, "1d")


def labels(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)
    for n in (50, 100, 200):
        out[f"above_sma{n}"] = close > close.rolling(n).mean()
    out["mom28"] = close / close.shift(28) - 1
    out["mom90"] = close / close.shift(90) - 1
    vol = np.log(close).diff().rolling(25).std() * np.sqrt(365)
    out["vol"] = vol
    if "funding_rate" in bars:
        out["funding7"] = bars["funding_rate"].rolling(7).sum()
    out = out.shift(1)  # known at the previous close
    for n in (50, 100, 200):
        out[f"above_sma{n}"] = out[f"above_sma{n}"].astype("boolean").fillna(False).astype(bool)
    return out


btc = labels(universe["BTC"]).add_prefix("btc_")
rows = []
for symbol, group in trades.groupby("symbol"):
    own = labels(universe[symbol]).join(btc)
    rows.append(group.join(own.reindex(group["entry"]).reset_index(drop=True).set_index(group.index)))
trades = pd.concat(rows)
trades["year"] = trades["entry"].dt.year


def table(frame: pd.DataFrame, by) -> pd.DataFrame:
    g = frame.groupby(by, observed=True)["return"]
    return pd.DataFrame({"trades": g.size(), "win": g.apply(lambda r: (r > 0).mean()),
                         "avg": g.mean(), "sum": g.sum(), "worst": g.min(), "best": g.max()}).round(3)


shorts, longs = trades[trades["direction"] < 0], trades[trades["direction"] > 0]
print("ALL SHORTS:", table(shorts, lambda _: "all").to_string(header=False))
for name in ["above_sma50", "above_sma100", "above_sma200", "btc_above_sma100", "btc_above_sma200"]:
    print(f"\nShorts by {name}:\n{table(shorts, name).to_string()}")
both = shorts.assign(regime=np.where(~shorts["above_sma200"] & ~shorts["btc_above_sma200"], "coin & BTC below 200d",
                     np.where(~shorts["above_sma200"], "coin below, BTC above", "coin above 200d")))
print("\nShorts by combined regime:\n", table(both, "regime").to_string())
print("\nShorts below own 200d SMA, by year:\n", table(shorts[~shorts["above_sma200"]], "year").to_string())
print("\nShorts above own 200d SMA, by year:\n", table(shorts[shorts["above_sma200"]], "year").to_string())
print("\nShorts by symbol (below own 200d):\n", table(shorts[~shorts["above_sma200"]], "symbol").to_string())
shorts = shorts.assign(mom90_down=shorts["mom90"] < 0)
print("\nShorts by 90d momentum also negative:\n", table(shorts, "mom90_down").to_string())
shorts = shorts.assign(vol_t=pd.qcut(shorts["vol"], 3, labels=["low", "mid", "high"]))
print("\nShorts by own volatility tercile:\n", table(shorts, "vol_t").to_string())
if "funding7" in shorts:
    shorts = shorts.assign(fund=pd.qcut(shorts["funding7"], 3, labels=["low", "mid", "high"]))
    print("\nShorts by 7d funding tercile (high = longs paying):\n", table(shorts, "fund").to_string())
print("\nShorts by bars held:\n", table(shorts, pd.cut(shorts["bars"], [0, 3, 10, 30, 999])).to_string())
print("\nShort excursions: losers' median MFE (ATR) %.2f; share of losers with MFE >= 1 ATR %.2f" % (
    shorts.loc[shorts["return"] < 0, "mfe_atr"].median(), (shorts.loc[shorts["return"] < 0, "mfe_atr"] >= 1).mean()))
print("Short MAE (ATR) percentiles of winners: 50%% %.2f 75%% %.2f 90%% %.2f" % tuple(
    shorts.loc[shorts["return"] > 0, "mae_atr"].quantile([0.5, 0.75, 0.9])))
print("Shorts losing more than 20%%: %d trades, sum %.2f" % ((shorts["return"] < -0.2).sum(),
      shorts.loc[shorts["return"] < -0.2, "return"].sum()))
print("\nLONGS by own 200d SMA (for comparison):\n", table(longs, "above_sma200").to_string())

# Whipsaw check: how strong was the 28-day move (in volatility units) when the signal flipped?
trades["strength"] = (trades["mom28"].abs() / (trades["vol"] * np.sqrt(28 / 365))).round(3)
bins = [0, 0.25, 0.5, 1.0, 99]
for side, frame in [("SHORTS", trades[trades["direction"] < 0]), ("LONGS", trades[trades["direction"] > 0])]:
    print(f"\n{side} by signal strength at entry (|28d return| / 28d vol):\n",
          table(frame, pd.cut(frame["strength"], bins)).to_string())
    print(f"{side} median bars held by strength:", frame.groupby(pd.cut(frame["strength"], bins), observed=True)["bars"].median().to_dict())
