"""Checks every strategy in ``strategies/ideas`` gets automatically.

Position strategies also join the catalogue-wide checks in ``tests/test_strategies.py``
(no lookahead, aligned and bounded targets, JSON round trip, trades on synthetic bars).
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.strategies import IDEAS, BracketStrategy, from_spec, to_spec

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "src" / "algotrade" / "strategies" / "ideas"
BRACKET_IDEAS = [cls for cls in IDEAS if issubclass(cls, BracketStrategy)]


@pytest.mark.parametrize("cls", IDEAS, ids=lambda cls: cls.name)
def test_idea_names_carry_their_idea_id_and_module(cls) -> None:
    module = cls.__module__.rsplit(".", 1)[-1]
    assert re.match(r"^i\d{3}_", cls.name), f"{cls.name} must start with its idea id"
    assert cls.name.startswith(module[:5]), f"{cls.name} lives in the wrong module {module}"


def test_every_idea_module_has_its_own_tests() -> None:
    for module in CODE.glob("i[0-9][0-9][0-9]_*.py"):
        assert (ROOT / "tests" / "ideas" / f"test_{module.stem}.py").exists(), module.name


@pytest.mark.parametrize("cls", IDEAS, ids=lambda cls: cls.name)
def test_idea_spec_round_trip(cls) -> None:
    strategy = cls()
    assert from_spec(to_spec(strategy)) == strategy


@pytest.mark.parametrize("cls", BRACKET_IDEAS, ids=lambda cls: cls.name)
def test_bracket_ideas_have_no_lookahead(cls, bars) -> None:
    strategy = cls()
    costs = EXCHANGE_COSTS["binanceusdm"]
    result = strategy.simulate(bars, costs)
    full = result.ledger
    # Without trades on the fixture the cuts below would compare flat ledgers and test nothing.
    assert len(result.trades) > 0
    # Fixed cuts, plus cuts ending on each signal bar and on each entry bar.
    entries = bars.index.get_indexer(pd.DatetimeIndex(result.trades["entry"]))
    for cut in sorted({120, 260, 410, *entries.tolist(), *(entries + 1).tolist()}):
        partial = strategy.simulate(bars.iloc[:cut], costs).ledger
        pd.testing.assert_frame_equal(partial, full.iloc[:cut])
