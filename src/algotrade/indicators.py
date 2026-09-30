"""Causal technical indicators: each value uses only data up to and including that bar."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# --- averages -------------------------------------------------------------------------------


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def wilder(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR, ADX): an EMA with alpha = 1 / length."""
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


# --- volatility -----------------------------------------------------------------------------


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift()
    ranges = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1)
    return ranges.max(axis=1, skipna=False).fillna(high - low)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), length)


def ewm_vol(returns: pd.Series, span: float) -> pd.Series:
    """Exponentially weighted standard deviation of per-bar returns (not annualized)."""
    return returns.ewm(span=span, adjust=False, min_periods=max(int(span // 2), 2)).std()


# --- bands and channels ---------------------------------------------------------------------


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = sma(close, length)
    width = mult * close.rolling(length, min_periods=length).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + width, "lower": mid - width})


def donchian(high: pd.Series, low: pd.Series, length: int = 20) -> pd.DataFrame:
    """Highest high / lowest low of the last ``length`` bars, including the current one."""
    upper = high.rolling(length, min_periods=length).max()
    lower = low.rolling(length, min_periods=length).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2})


def keltner(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, mult: float = 2.0
) -> pd.DataFrame:
    mid = ema(close, length)
    width = mult * atr(high, low, close, length)
    return pd.DataFrame({"mid": mid, "upper": mid + width, "lower": mid - width})


# --- oscillators ----------------------------------------------------------------------------


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), length)
    loss = wilder(-delta.clip(upper=0), length)
    value = 100 - 100 / (1 + gain / loss)
    return value.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    trigger = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": trigger, "hist": line - trigger})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = wilder(true_range(high, low, close), length)
    plus_di = 100 * wilder(plus_dm, length) / tr
    minus_di = 100 * wilder(minus_dm, length) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": wilder(dx, length)})


def zscore(series: pd.Series, length: int) -> pd.Series:
    mean = series.rolling(length, min_periods=length).mean()
    std = series.rolling(length, min_periods=length).std(ddof=0)
    return (series - mean) / std


# --- trailing stops -------------------------------------------------------------------------


def parabolic_sar(high: pd.Series, low: pd.Series, step: float, max_step: float) -> pd.Series:
    values = _psar(high.to_numpy(dtype="float64"), low.to_numpy(dtype="float64"), step, max_step)
    return pd.Series(values, index=high.index, name="psar")


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10, mult: float = 3.0
) -> pd.DataFrame:
    """SuperTrend line and direction (+1 up, -1 down, 0 during warm-up)."""
    band = mult * atr(high, low, close, length)
    hl2 = (high + low) / 2
    line, direction = _supertrend(
        (hl2 + band).to_numpy(dtype="float64"),
        (hl2 - band).to_numpy(dtype="float64"),
        close.to_numpy(dtype="float64"),
    )
    return pd.DataFrame({"line": line, "direction": direction}, index=close.index)


@njit(cache=True)
def _psar(high: np.ndarray, low: np.ndarray, step: float, max_step: float) -> np.ndarray:
    n = high.shape[0]
    psar = np.empty(n)
    if n == 0:
        return psar
    bull = True
    af = step
    ep = high[0]
    psar[0] = low[0]
    for i in range(1, n):
        prev = psar[i - 1]
        value = prev + af * (ep - prev)
        if bull:
            value = min(value, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < value:
                bull = False
                value = ep
                ep = low[i]
                af = step
        else:
            value = max(value, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > value:
                bull = True
                value = ep
                ep = high[i]
                af = step
        psar[i] = value
        if bull:
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
        elif low[i] < ep:
            ep = low[i]
            af = min(af + step, max_step)
    return psar


@njit(cache=True)
def _supertrend(
    basic_upper: np.ndarray, basic_lower: np.ndarray, close: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    line = np.full(n, np.nan)
    direction = np.zeros(n)
    upper = np.nan
    lower = np.nan
    trend = 0.0
    for i in range(n):
        if np.isnan(basic_upper[i]):
            continue
        if np.isnan(upper):
            upper, lower, trend = basic_upper[i], basic_lower[i], 1.0
        else:
            # Bands only tighten while price stays on their side.
            upper = min(basic_upper[i], upper) if close[i - 1] <= upper else basic_upper[i]
            lower = max(basic_lower[i], lower) if close[i - 1] >= lower else basic_lower[i]
            if trend > 0 and close[i] < lower:
                trend = -1.0
            elif trend < 0 and close[i] > upper:
                trend = 1.0
        direction[i] = trend
        line[i] = lower if trend > 0 else upper
    return line, direction


@njit(cache=True)
def entry_exit_state(
    enter_long: np.ndarray, exit_long: np.ndarray, enter_short: np.ndarray, exit_short: np.ndarray
) -> np.ndarray:
    """Turn entry/exit conditions into a +1/0/-1 position that persists between signals.

    Exits are processed before entries, and an opposite entry also closes the open position,
    so a single bar can reverse. An entry is ignored on a bar where its own exit is also true.
    """
    n = enter_long.shape[0]
    out = np.zeros(n)
    pos = 0.0
    for i in range(n):
        closing_long = pos > 0 and (exit_long[i] or enter_short[i])
        closing_short = pos < 0 and (exit_short[i] or enter_long[i])
        if closing_long or closing_short:
            pos = 0.0
        if pos == 0:
            if enter_long[i] and not exit_long[i] and not enter_short[i]:
                pos = 1.0
            elif enter_short[i] and not exit_short[i] and not enter_long[i]:
                pos = -1.0
        out[i] = pos
    return out
