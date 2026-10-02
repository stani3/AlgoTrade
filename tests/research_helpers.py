"""A throwaway research workspace: real criteria, synthetic market data, its own git repo."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from algotrade.data.exchange import MarketId, funding_path, ohlcv_path
from algotrade.research.cards import Card, parse_card
from algotrade.research.criteria import Criteria, load_criteria
from algotrade.research.workspace import Workspace

REPO = Path(__file__).resolve().parent.parent
SYMBOLS = ("BTC", "ETH", "SOL")
FREQ = {"4h": "4h", "1d": "1D"}


def synthetic_bars(
    seed: int, start: str, end: str, freq: str, edge: float = 0.0, waves: bool = True
) -> pd.DataFrame:
    """Random walk, by default with slow waves; ``edge`` > 0 plants persistent trends (regimes
    of a few weeks with a drift of ``edge`` per bar). ``waves=False`` with no edge is a pure
    random walk that no rule should beat."""

    index = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    rng = np.random.default_rng(seed)
    n = len(index)
    trend = np.sin(np.arange(n) / (n / 9)) * 0.004 * waves
    if edge:
        regime = np.repeat(rng.choice([-1.0, 1.0], n // 120 + 1), 120)[:n]
        trend = trend + edge * regime
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n) + trend))
    open_ = np.concatenate(([close[0]], close[:-1]))
    spread = np.abs(rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": 1.0,
        }
    )


def write_market(
    root: Path,
    symbols: tuple[str, ...] = SYMBOLS,
    start: str = "2023-01-01",
    end: str = "2026-01-01",
    exchange: str = "binanceusdm",
    edge: float = 0.0,
    waves: bool = True,
) -> None:
    for i, symbol in enumerate(symbols):
        market = MarketId(exchange=exchange, base=symbol)
        for timeframe, freq in FREQ.items():
            path = ohlcv_path(root, market, timeframe)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Same seed for both timeframes keeps the symbols distinct but deterministic.
            synthetic_bars(100 + i, start, end, freq, edge, waves).to_parquet(path, index=False)
        stamps = pd.date_range(start, end, freq="8h", tz="UTC", inclusive="left")
        rng = np.random.default_rng(200 + i)
        pd.DataFrame(
            {"timestamp": stamps, "funding_rate": rng.normal(1e-4, 1e-4, len(stamps))}
        ).to_parquet(funding_path(root, market), index=False)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def init_repo(root: Path) -> None:
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Research Test")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "config", "core.autocrlf", "false")


def make_workspace(
    root: Path,
    overrides: dict | None = None,
    repo: bool = True,
    market: bool = True,
    edge: float = 0.0,
    waves: bool = True,
) -> tuple[Workspace, Criteria]:
    """Workspace with the real criteria.yaml (optionally overridden) and synthetic data."""

    research = root / "research"
    research.mkdir(parents=True)
    criteria = yaml.safe_load((REPO / "research" / "criteria.yaml").read_text(encoding="utf-8"))
    criteria["data"]["symbols"] = list(SYMBOLS)
    for dotted, value in (overrides or {}).items():
        node = criteria
        *parents, last = dotted.split(".")
        for key in parents:
            node = node[key]
        node[last] = value
    (research / "criteria.yaml").write_text(yaml.safe_dump(criteria), encoding="utf-8")
    shutil.copytree(REPO / "specs", root / "specs")
    if market:
        write_market(root / "data" / "raw", edge=edge, waves=waves)
    if repo:
        init_repo(root)
        (root / ".gitignore").write_text("data/\nreports/\n", encoding="utf-8")
        git(root, "add", ".")
        git(root, "commit", "-q", "-m", "workspace")
    ws = Workspace(root=root)
    return ws, load_criteria(ws.criteria_path)


CARD = """
---
title: {title}
source: "Carver, Systematic Trading"
source_id: {source_id}
taxonomy:
  family: {family}
  inputs: {inputs}
  horizon: {horizon}
timeframe: {timeframe}
spec: {spec}
optimise: {optimise}
expected_trades_per_year: 12
differs_from: {differs_from}
---
## Hypothesis
Prices that trend keep trending for a while.

## Why it should work
Investors under-react to news, then herd.

## Rules
Long when the fast average is above the slow one, short below.

## Falsified if
The median Sharpe ratio across symbols is below 0.3.
"""


def draft(
    title: str = "Fast trend",
    spec: dict | None = None,
    optimise: dict | None = None,
    timeframe: str = "4h",
    family: str = "trend",
    inputs: list[str] | None = None,
    horizon: str = "days",
    differs_from: dict | None = None,
    source_id: str = "S01",
) -> Card:
    text = textwrap.dedent(CARD).strip() + "\n"
    filled = text.format(
        title=title,
        source_id=source_id,
        family=family,
        inputs=inputs or ["price"],
        horizon=horizon,
        timeframe=timeframe,
        spec=yaml.safe_dump(spec or {"type": "ma_crossover", "fast": 20, "slow": 100},
                            default_flow_style=True).strip(),
        optimise=yaml.safe_dump(optimise or {}, default_flow_style=True).strip(),
        differs_from=yaml.safe_dump(differs_from or {}, default_flow_style=True).strip(),
    )  # fmt: skip
    return parse_card(filled)


def write_draft(root: Path, card: Card, name: str = "draft.md") -> Path:
    path = root / name
    path.write_text(card.render(), encoding="utf-8")
    return path


def commits(root: Path) -> list[str]:
    return git(root, "log", "--format=%s").splitlines()


FAST_GATES = {
    "feasibility.monkey_runs": 100,
    "validation.in_sample_years": 1,
    "validation.out_of_sample_months": 3,
    "validation.monte_carlo_runs": 400,
}


def frozen_workspace(root: Path, overrides: dict | None = None) -> tuple[Workspace, Criteria]:
    """A planted-edge workspace with idea i001 (EWMAC) taken through every gate and frozen."""

    from algotrade.research.buildcheck import build_check
    from algotrade.research.feasibility import feasibility
    from algotrade.research.freeze import freeze
    from algotrade.research.holdout import holdout
    from algotrade.research.registry import find_version, register
    from algotrade.research.validate import validate

    ws, criteria = make_workspace(root, overrides={**FAST_GATES, **(overrides or {})}, edge=0.002)
    card = draft(spec={"type": "ewmac"}, optimise={"fast": [8, 16], "slow": [64, 128]})
    build_check(ws, criteria, register(ws, criteria, card))
    for stage in (feasibility, validate, holdout):
        result = stage(ws, criteria, find_version(ws, "i001"), report=False)
        assert result.verdict == "PASS", (result.stage, result.reasons)
    freeze(ws, criteria, find_version(ws, "i001"))
    return ws, criteria
