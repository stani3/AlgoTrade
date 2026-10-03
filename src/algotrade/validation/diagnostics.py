"""Evidence for a human (or the feasibility skill) to read; nothing here decides anything.

* sides: long against short trades;
* symbols and years: where and when it made or lost money;
* regimes: returns while the market trended (ADX >= 25), ranged (ADX <= 20) or neither, and in
  low/middle/high volatility terciles (terciles of each symbol's own ATR%, measured over the
  period: fine for a diagnostic, it would be lookahead in a trading rule);
* holding: how long winners and losers are held;
* excursions: maximum adverse (MAE) and favourable (MFE) excursion of each trade, as a share of
  the entry price and in ATRs at entry - where stops and targets actually sit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algotrade.backtest.engine import BacktestResult
from algotrade.backtest.metrics import periods_per_year, sharpe_ratio
from algotrade.indicators import adx, atr


def trade_excursions(
    result: BacktestResult, bars: pd.DataFrame, atr_length: int = 14
) -> pd.DataFrame:
    """Closed trades with entry/exit price, MAE and MFE (fractions of entry, and in ATRs)."""

    trades = result.trades
    if trades.empty:
        return pd.DataFrame(
            columns=["direction", "bars", "return", "mae", "mfe", "mae_atr", "mfe_atr"]
        )
    trades = trades[~trades["open"].astype(bool)].copy()
    index = bars.index
    high, low, close = (bars[c].to_numpy(dtype="float64") for c in ("high", "low", "close"))
    volatility = atr(bars["high"], bars["low"], bars["close"], atr_length).to_numpy()
    first = index.get_indexer(pd.DatetimeIndex(trades["entry"]))
    if "entry_price" in trades:
        # Bracket trades: filled at the entry bar's open, exit bar is the last one touched.
        last = index.get_indexer(pd.DatetimeIndex(trades["exit"]))
        entry_price = trades["entry_price"].to_numpy(dtype="float64")
        reference = np.maximum(first - 1, 0)
    else:
        # Engine trades: entered at the previous close, held through ``bars`` bars.
        last = first + trades["bars"].to_numpy() - 1
        reference = np.maximum(first - 1, 0)
        entry_price = close[reference]
    direction = trades["direction"].to_numpy()
    mae, mfe = np.zeros(len(trades)), np.zeros(len(trades))
    for row, (start, stop) in enumerate(zip(first, last, strict=True)):
        top, bottom = high[start : stop + 1].max(), low[start : stop + 1].min()
        up, down = (
            (top - entry_price[row]) / entry_price[row],
            (entry_price[row] - bottom) / entry_price[row],
        )
        mfe[row], mae[row] = (up, down) if direction[row] > 0 else (down, up)
    atr_share = volatility[reference] / entry_price
    trades["mae"], trades["mfe"] = np.maximum(mae, 0.0), np.maximum(mfe, 0.0)
    trades["mae_atr"], trades["mfe_atr"] = trades["mae"] / atr_share, trades["mfe"] / atr_share
    return trades


def _side_row(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sum_return": 0.0,
            "median_bars": 0.0,
        }
    return {
        "trades": len(trades),
        "win_rate": float((trades["return"] > 0).mean()),
        "avg_return": float(trades["return"].mean()),
        "sum_return": float(trades["return"].sum()),
        "median_bars": float(trades["bars"].median()),
    }


def regime_labels(bars: pd.DataFrame) -> pd.DataFrame:
    strength = adx(bars["high"], bars["low"], bars["close"], 14)["adx"]
    trend = pd.Series("neither", index=bars.index)
    trend[strength >= 25] = "trending (ADX>=25)"
    trend[strength <= 20] = "ranging (ADX<=20)"
    trend[strength.isna()] = "warm-up"
    width = atr(bars["high"], bars["low"], bars["close"], 14) / bars["close"]
    terciles = pd.qcut(width.rank(method="first"), 3, labels=["low vol", "mid vol", "high vol"])
    volatility = terciles.astype(object).where(terciles.notna(), "warm-up")
    return pd.DataFrame({"trend": trend, "volatility": volatility})


def diagnostics(
    results: dict[str, BacktestResult], universe: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    trades = []
    regime_rows = []
    yearly = {}
    symbols = {}
    for symbol, result in results.items():
        bars = universe[symbol]
        table = trade_excursions(result, bars).assign(symbol=symbol)
        trades.append(table)
        ppy = periods_per_year(bars.index)
        equity = result.equity
        symbols[symbol] = {
            "sharpe": sharpe_ratio(result.returns, ppy),
            "total_return": float(equity.iloc[-1] - 1.0),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "trades": len(table),
            "exposure": float((result.ledger["position"] != 0).mean()),
        }
        year_end = equity.groupby(equity.index.year).last()
        yearly[symbol] = year_end / year_end.shift(1, fill_value=1.0) - 1.0
        labels = regime_labels(bars)
        for kind in ("trend", "volatility"):
            frame = pd.DataFrame({"regime": labels[kind], "net": result.returns})
            for regime, part in frame.groupby("regime"):
                regime_rows.append(
                    {"kind": kind, "regime": regime, "symbol": symbol, "bars": len(part),
                     "mean_net": float(part["net"].mean()), "sharpe": sharpe_ratio(part["net"], ppy)}
                )  # fmt: skip
    trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    regimes = pd.DataFrame(regime_rows)
    regime_table = (
        regimes.groupby(["kind", "regime"])
        .agg(
            share_of_time=("bars", "sum"),
            median_sharpe=("sharpe", "median"),
            mean_net=("mean_net", "mean"),
        )
        .reset_index()
    )
    regime_table["share_of_time"] = regime_table.groupby("kind")["share_of_time"].transform(
        lambda s: s / s.sum()
    )
    years = pd.DataFrame(yearly)
    years["median"] = years.median(axis=1)

    sides = pd.DataFrame(
        {
            "all": _side_row(trades),
            "long": _side_row(trades[trades["direction"] > 0] if len(trades) else trades),
            "short": _side_row(trades[trades["direction"] < 0] if len(trades) else trades),
        }
    ).T
    holding = pd.DataFrame(
        {
            name: part["bars"].describe(percentiles=[0.5, 0.75, 0.9])[
                ["count", "50%", "75%", "90%", "max"]
            ]
            for name, part in (
                ("winners", trades[trades["return"] > 0] if len(trades) else trades),
                ("losers", trades[trades["return"] <= 0] if len(trades) else trades),
            )
            if len(part)
        }
    ).T
    excursions = pd.DataFrame(
        {
            name: {
                "median_mae": part["mae"].median(),
                "p75_mae": part["mae"].quantile(0.75),
                "median_mfe": part["mfe"].median(),
                "p75_mfe": part["mfe"].quantile(0.75),
                "median_mae_atr": part["mae_atr"].median(),
                "median_mfe_atr": part["mfe_atr"].median(),
            }
            for name, part in (
                ("winners", trades[trades["return"] > 0] if len(trades) else trades),
                ("losers", trades[trades["return"] <= 0] if len(trades) else trades),
            )
            if len(part)
        }
    ).T
    return {
        "symbols": pd.DataFrame(symbols).T,
        "sides": sides,
        "years": years,
        "regimes": regime_table,
        "holding": holding,
        "excursions": excursions,
        "trades": trades,
    }


def by_class(
    per_symbol: pd.DataFrame, classes: dict[str, str], entry: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-symbol statistics summarised by asset class (in order of first appearance).

    ``per_symbol`` has one row per symbol with ``sharpe``, ``cagr`` and ``trades`` (or
    ``closed_trades``); ``entry`` is the entry test's table, whose profitable share per class
    is added when given.
    """

    frame = per_symbol.assign(asset_class=[classes[symbol] for symbol in per_symbol.index])
    grouped = frame.groupby("asset_class", sort=False)
    trades = "trades" if "trades" in frame else "closed_trades"
    table = pd.DataFrame(
        {
            "symbols": grouped.size(),
            "median_sharpe": grouped["sharpe"].median(),
            "positive_share": grouped["sharpe"].agg(lambda s: float((s > 0).mean())),
            "median_trades": grouped[trades].median(),
        }
    )
    if "cagr" in frame:
        table["median_cagr"] = grouped["cagr"].median()
    if entry is not None and len(entry):
        cells = entry.assign(asset_class=[classes[s] for s in entry["symbol"]])
        table["entry_profitable_share"] = cells.groupby("asset_class", sort=False)[
            "profitable"
        ].mean()
    table.index.name = "asset_class"
    return table
