"""Crypto results must not change: recompute committed feasibility records on the real data.

Slow (about a minute per record) and needs the downloaded market data, so it only runs with
``pytest -m slow``. Point ``ALGOTRADE_DATA_ROOT`` at another checkout's ``data/raw`` to run it
from a worktree.

* Full records: every feasibility result recorded under the criteria frozen in
  ``tests/fixtures/criteria_2026-10-03.yaml`` is recomputed under that file and under the
  current ``research/criteria.yaml``; checks, metrics, notes, the chosen cell and every CSV must
  be exactly equal to what was committed.
* Headlines: every feasibility result's chosen spec is re-run and its median Sharpe compared,
  as ``research report`` does.

Both fail, rather than skip, when they could not check anything.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
from legacy_serial import legacy_feasibility

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.config import data_root_override
from algotrade.parallel import ENV
from algotrade.research.cards import read_card
from algotrade.research.criteria import Criteria, load_criteria
from algotrade.research.feasibility import run_feasibility
from algotrade.research.split import FUNDING_ALIGNMENT, dev_universe, recorded_alignment, window
from algotrade.research.workspace import Workspace
from algotrade.validation.optimize import evaluate

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parent.parent
DATA = data_root_override() or REPO / "data" / "raw"
WORKERS = int(os.environ.get(ENV) or 4)  # read at import: tests run inline otherwise
FROZEN = Path(__file__).parent / "fixtures" / "criteria_2026-10-03.yaml"
HEADLINE = "median Sharpe at the chosen parameters"


def _records() -> list[Path]:
    return sorted((REPO / "research" / "ideas").glob("*/v*/feasibility/result.json"))


def _criteria_files() -> list[Criteria]:
    """The frozen criteria and the current ones (once, if they are the same file)."""

    found = {}
    for path in (FROZEN, REPO / "research" / "criteria.yaml"):
        criteria = load_criteria(path)
        found.setdefault(criteria.hash, criteria)
    return list(found.values())


def _universe(criteria: Criteria, record: dict, timeframe: str) -> dict | None:
    """The development bars as the record saw them (its funding alignment, whatever
    ``data.funding_alignment`` says now), or None when they no longer hash as recorded."""

    ws = Workspace(REPO, data_root=DATA)
    criteria = criteria.with_value(FUNDING_ALIGNMENT, recorded_alignment(record))
    try:
        universe = dev_universe(ws, criteria, timeframe)
    except FileNotFoundError:
        return None
    now = {w.symbol: w.sha for w in (window(b, s, timeframe) for s, b in universe.items())}
    recorded = {w["symbol"]: w["sha"] for w in record["provenance"]["data"]}
    return universe if now == recorded else None


def _costs(criteria: Criteria):
    exchange = criteria.get("data.exchange", None)
    if exchange is None:  # multi-asset criteria: the crypto universe's exchange
        exchange = criteria.get("data.universes.crypto.source")
    return EXCHANGE_COSTS[exchange]


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def _parallel(card, universe: dict, costs, criteria: Criteria) -> dict:
    """The production code with worker processes (ALGOTRADE_WORKERS, default 4 here)."""

    found = run_feasibility(card.spec, card.optimise, universe, costs, criteria, report=False,
                            workers=WORKERS)  # fmt: skip
    return {
        "checks": [c.to_dict() for c in found.checks],
        "metrics": found.metrics,
        "notes": found.notes,
        "trials": len(found.limited.board),
        "chosen": found.chosen_params,
        "chosen_spec": found.spec,
        "csvs": found.csvs,
    }


def test_feasibility_records_reproduce_exactly() -> None:
    frozen = load_criteria(FROZEN)
    checked, unverifiable = [], []
    for path in _records():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["provenance"]["criteria_hash"] != frozen.hash:
            continue
        card = read_card(path.parent.parent / "idea.md")
        for criteria in _criteria_files():
            universe = _universe(criteria, record, card.timeframe)
            if universe is None:
                unverifiable.append(f"{path.parent.parent.parent.name} ({criteria.hash})")
                continue
            costs = _costs(criteria)
            for how, again in (
                ("serial reference", legacy_feasibility(card.spec, card.optimise, universe, costs,
                                                        frozen)),
                ("parallel", _parallel(card, universe, costs, frozen)),
            ):  # fmt: skip
                name = f"{path.parent.parent.parent.name} under criteria {criteria.hash}, {how}"
                assert again["checks"] == record["checks"], name
                assert again["metrics"] == record["metrics"], name
                assert again["notes"] == record["notes"], name
                assert again["trials"] == record["trials"], name
                assert again["chosen"] == record["chosen"], name
                assert again["chosen_spec"] == record["chosen_spec"], name
                for file, text in again["csvs"].items():
                    stored = (path.parent / file).read_text(encoding="utf-8")
                    assert _lf(text) == _lf(stored), f"{name}: {file}"
                checked.append(name)
    assert checked, f"no feasibility record could be checked; unverifiable: {unverifiable}"
    assert not unverifiable, f"data changed or missing for {unverifiable}"


def test_feasibility_headlines_reproduce() -> None:
    criteria = load_criteria(REPO / "research" / "criteria.yaml")
    checked = []
    for path in _records():
        record = json.loads(path.read_text(encoding="utf-8"))
        card = read_card(path.parent.parent / "idea.md")
        universe = _universe(criteria, record, card.timeframe)
        if universe is None:
            continue
        _, per_symbol = evaluate(record["chosen_spec"], universe, _costs(criteria))
        actual = float(per_symbol["sharpe"].median())
        assert math.isclose(record["metrics"][HEADLINE], actual, rel_tol=1e-9, abs_tol=1e-12), (
            path.parent.parent.parent.name
        )
        checked.append(path.parent.parent.parent.name)
    assert len(checked) >= 3, f"only {checked} could be checked"
