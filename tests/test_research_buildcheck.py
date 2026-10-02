import json
import subprocess
import sys
import textwrap

import pytest
from research_helpers import commits, draft, make_workspace

import algotrade.strategies.ideas as idea_package
from algotrade.research.buildcheck import (
    blocking,
    build_check,
    code_files,
    measure,
    ruff_problems,
)
from algotrade.research.cards import CardError
from algotrade.research.fingerprint import Similarity
from algotrade.research.journal import Journal
from algotrade.research.registry import Refused, find_version, register
from algotrade.strategies import STRATEGIES

WHY = "Blends the same signal through a combiner, which a careless idea could think is new."

DUMMY = """
def sign(x):
    if x > 0:
        return 1
    return -1
"""
ONE_BRANCH = """
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("dummy", Path(__file__).parent / "dummy.py")
dummy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dummy)


def test_positive():
    assert dummy.sign(2) == 1
"""
OTHER_BRANCH = """

def test_negative():
    assert dummy.sign(-2) == -1
"""


def ruff_format(root, *paths) -> None:
    """Tidy generated files the way a careful author would, with the gate's working directory."""

    files = list(map(str, paths))
    ruff = [sys.executable, "-m", "ruff"]
    subprocess.run([*ruff, "check", "--fix", "-q", *files], cwd=root, check=False)
    subprocess.run([*ruff, "format", "-q", *files], cwd=root, check=True)


def test_coverage_gate_fails_on_a_missing_branch_then_passes(tmp_path) -> None:
    module, tests = tmp_path / "dummy.py", tmp_path / "test_dummy.py"
    module.write_text(textwrap.dedent(DUMMY).lstrip())
    tests.write_text(textwrap.dedent(ONE_BRANCH).lstrip())
    partial = measure(tmp_path, module, tests)
    assert partial.tests_passed
    assert partial.branch == pytest.approx(0.5) and partial.line < 1.0
    assert partial.missing_lines == [4] and partial.missing_branches == [[2, 4]]

    tests.write_text(tests.read_text() + textwrap.dedent(OTHER_BRANCH))
    full = measure(tmp_path, module, tests)
    assert full.tests_passed and full.line == 1.0 and full.branch == 1.0


def test_coverage_gate_reports_failing_tests_and_unmeasured_modules(tmp_path) -> None:
    module, tests = tmp_path / "dummy.py", tmp_path / "test_dummy.py"
    module.write_text(textwrap.dedent(DUMMY).lstrip())
    tests.write_text("def test_broken():\n    assert False\n")
    result = measure(tmp_path, module, tests)
    assert not result.tests_passed and result.line == 0.0 and "1 failed" in result.output


def test_ruff_problems(tmp_path) -> None:
    clean = tmp_path / "clean.py"
    clean.write_text("x = 1\n")
    assert ruff_problems(tmp_path, [clean]) == []
    messy = tmp_path / "messy.py"
    messy.write_text("import os\nx=1\n")
    problems = ruff_problems(tmp_path, [messy])
    assert any(p.startswith("ruff check") for p in problems)
    assert any(p.startswith("ruff format") for p in problems)


# --- the full gate ----------------------------------------------------------------------------


@pytest.fixture
def setup(tmp_path):
    return make_workspace(tmp_path)


def test_catalogue_spec_passes_and_is_committed_with_its_fingerprint(setup) -> None:
    ws, criteria = setup
    version = register(ws, criteria, draft(spec={"type": "ewmac"}, optimise={"fast": [8, 16]}))
    assert code_files(ws, version) == []
    outcome = build_check(ws, criteria, version)
    assert outcome.recorded and outcome.result.verdict == "PASS"
    data = json.loads((ws.stage_dir("i001", version.slug, 1, "build") / "result.json").read_text())
    assert data["verdict"] == "PASS" and data["provenance"]["criteria_hash"] == criteria.hash
    assert [w["symbol"] for w in data["provenance"]["data"]] == ["BTC", "ETH", "SOL"]
    assert all(w["end"] < "2025-05-01" for w in data["provenance"]["data"])
    assert len(list(ws.fingerprints_dir.glob("*.parquet"))) == 1
    assert commits(ws.root)[0] == "research(i001-fast-trend v1): build PASS"
    assert Journal(ws.journal_path).ideas()["i001"].status == "passed:build"
    with pytest.raises(Refused, match="already built"):
        build_check(ws, criteria, find_version(ws, "i001"))


def test_edited_card_is_refused(setup) -> None:
    ws, criteria = setup
    version = register(ws, criteria, draft(), commit=False)
    path = ws.root / version.card_path
    path.write_text(path.read_text(encoding="utf-8") + "\nsneaky edit\n", encoding="utf-8")
    with pytest.raises(Refused, match="edited after registration"):
        build_check(ws, criteria, version, commit=False)


def test_closed_ideas_cannot_be_built(setup) -> None:
    ws, criteria = setup
    register(ws, criteria, draft(), commit=False)
    Journal(ws.journal_path).append("abandoned", idea="i001", version=1, reason="x")
    with pytest.raises(Refused, match="closed"):
        build_check(ws, criteria, find_version(ws, "i001"), commit=False)


def test_grid_on_an_unknown_parameter_is_refused_at_registration(setup) -> None:
    ws, criteria = setup
    card = draft(spec={"type": "ewmac"}, optimise={"speed": [1, 2]})
    with pytest.raises(CardError, match="does not build"):
        register(ws, criteria, card, commit=False)
    bad_value = draft(spec={"type": "breakout_bracket"}, optimise={"lookback": [1, 48]})
    with pytest.raises(CardError, match="lookback must be at least 2"):
        register(ws, criteria, bad_value, commit=False)
    assert not ws.journal_path.exists()


def test_behavioural_duplicate_fails_and_closes_the_idea(setup) -> None:
    ws, criteria = setup
    first = register(ws, criteria, draft(spec={"type": "ewmac"}), commit=False)
    assert build_check(ws, criteria, first, commit=False).result.verdict == "PASS"
    blended = {"type": "combine", "strategies": [{"type": "ewmac"}], "weights": [1]}
    card = draft(title="Blended trend", spec=blended, differs_from={"i001": WHY})
    second = register(ws, criteria, card, commit=False)
    outcome = build_check(ws, criteria, second)
    assert outcome.recorded and outcome.result.verdict == "FAIL"
    assert "behavioural duplicate of i001 v1 Fast trend" in outcome.result.reasons[0]
    state = Journal(ws.journal_path).ideas()["i002"]
    assert state.status == "failed:build"
    assert commits(ws.root)[0].startswith("research(i002-blended-trend v1): build FAIL")
    assert len(list(ws.fingerprints_dir.glob("*.parquet"))) == 1  # no fingerprint for a copy


# --- new strategy code ------------------------------------------------------------------------

MODULE = '''
"""Test idea: long when the close is above its 50-bar mean."""

from dataclasses import dataclass

import pandas as pd

from algotrade.strategies.base import Strategy


@dataclass(frozen=True)
class AboveMean(Strategy):
    length: int = 50

    name = "{name}"

    def __post_init__(self) -> None:
        if self.length < 2:
            raise ValueError("length must be at least 2")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        mean = bars["close"].rolling(self.length).mean()
        return (bars["close"] > mean).astype(float)
'''
TESTS = """
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "src/algotrade/strategies/ideas/{name}.py"
spec = importlib.util.spec_from_file_location("{name}", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_long_above_the_mean():
    bars = pd.DataFrame({{"close": [1.0, 2.0, 3.0, 1.0]}})
    target = module.AboveMean(length=2).target_position(bars)
    assert target.tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0])
"""
VALIDATION_TEST = """

def test_rejects_short_length():
    with pytest.raises(ValueError):
        module.AboveMean(length=1)
"""


@pytest.fixture
def idea_code(setup, monkeypatch):
    """Lets ideas written into the temporary workspace be imported as idea modules."""

    ws, _ = setup
    ws.strategy_code_dir.mkdir(parents=True)
    ws.strategy_tests_dir.mkdir(parents=True)
    monkeypatch.setattr(
        idea_package, "__path__", [*idea_package.__path__, str(ws.strategy_code_dir)]
    )
    before = set(STRATEGIES)
    yield ws
    for name in set(STRATEGIES) - before:
        del STRATEGIES[name]
        sys.modules.pop(f"algotrade.strategies.ideas.{name}", None)


def test_new_strategy_code_must_be_fully_covered(setup, idea_code) -> None:
    ws, criteria = setup
    version = register(ws, criteria, draft(spec={"type": "above_mean", "length": 50}))
    name = "i001_above_mean"
    assert version.spec["type"] == name
    [(kind, module, tests)] = code_files(ws, version)
    assert kind == name and module.name == f"{name}.py" and tests.name == f"test_{name}.py"

    missing = build_check(ws, criteria, version)
    assert not missing.recorded and "expected src/algotrade/strategies/ideas" in missing.problems[0]

    module.write_text(textwrap.dedent(MODULE).lstrip().replace("{name}", name))
    tests.write_text(textwrap.dedent(TESTS).lstrip().format(name=name))
    ruff_format(ws.root, module, tests)
    partial = build_check(ws, criteria, version)
    assert not partial.recorded
    assert any("uncovered lines" in p for p in partial.problems)
    assert [c.name for c in partial.result.checks if not c.passed] == [
        f"line coverage ({name})",
        f"branch coverage ({name})",
    ]

    tests.write_text(tests.read_text() + textwrap.dedent(VALIDATION_TEST))
    ruff_format(ws.root, tests)
    outcome = build_check(ws, criteria, version)
    assert outcome.recorded and outcome.result.verdict == "PASS", outcome.problems
    shown = subprocess.run(
        ["git", "show", "--name-only", "--format=%s", "HEAD"],
        cwd=ws.root, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert "build PASS" in shown
    assert f"src/algotrade/strategies/ideas/{name}.py" in shown
    assert f"tests/ideas/test_{name}.py" in shown
    coverage = json.loads(
        (ws.stage_dir("i001", version.slug, 1, "build") / "result.json").read_text()
    )["coverage"]
    assert coverage[0]["line"] == 1.0 and coverage[0]["branch"] == 1.0


def test_failing_tests_and_lint_are_reported(setup, idea_code) -> None:
    ws, criteria = setup
    version = register(ws, criteria, draft(spec={"type": "above_mean"}), commit=False)
    [(name, module, tests)] = code_files(ws, version)
    module.write_text(textwrap.dedent(MODULE).lstrip().replace("{name}", name) + "import os\n")
    tests.write_text("def test_broken():\n    assert False\n")
    outcome = build_check(ws, criteria, version, commit=False)
    assert not outcome.recorded
    assert any(p.startswith("ruff check") for p in outcome.problems)
    assert any(p.startswith(f"tests failing for {name}") for p in outcome.problems)


def test_grid_on_a_parameter_the_new_code_lacks_does_not_build(setup, idea_code) -> None:
    ws, criteria = setup
    card = draft(spec={"type": "above_mean"}, optimise={"speed": [1, 2]})
    version = register(ws, criteria, card, commit=False)
    [(name, module, tests)] = code_files(ws, version)
    module.write_text(textwrap.dedent(MODULE).lstrip().replace("{name}", name))
    tests.write_text(
        textwrap.dedent(TESTS).lstrip().format(name=name) + textwrap.dedent(VALIDATION_TEST)
    )
    ruff_format(ws.root, module, tests)
    outcome = build_check(ws, criteria, version, commit=False)
    assert not outcome.recorded
    assert "does not build" in outcome.problems[0]


def test_screened_rules_block_only_re_implementations() -> None:
    ewmac = {"type": "vol_target", "strategy": {"type": "ewmac"}}
    screened = Similarity("k", "screened ewmac", 0.99, 0.5, 900, True, "screen", ewmac)
    idea = Similarity("k2", "i004 ewmac", 0.99, 0.5, 900, True, "idea", ewmac)
    hold = Similarity("k3", "buy & hold", float("nan"), 0.0, 900, True, "baseline",
                      {"type": "buy_and_hold"})  # fmt: skip
    unrelated = Similarity("k4", "far away", 0.1, 0.9, 900, False, "screen", ewmac)
    nearest = [screened, idea, hold, unrelated]
    assert blocking(nearest, {"type": "ewmac", "fast": 8}) == [idea, hold]
    assert blocking(nearest, {"type": "i007_my_ewmac"}) == [screened, idea, hold]
    assert blocking(nearest, {"type": "buy_and_hold"}) == [idea]
    # A different catalogue trend rule that happens to trade like the screened one is allowed.
    blend = {"type": "combine", "strategies": [{"type": "carver_breakout"}, {"type": "ewmac"}]}
    assert blocking([screened], blend) == []
    wrapped_new_code = {"type": "vol_target", "strategy": {"type": "i007_my_ewmac"}}
    assert blocking([screened], wrapped_new_code) == [screened]


def test_retest_also_covers_the_build_duplicate_check(setup) -> None:
    ws, criteria = setup
    first = register(ws, criteria, draft(spec={"type": "ewmac"}), commit=False)
    build_check(ws, criteria, first, commit=False)
    Journal(ws.journal_path).append(
        "stage_result", idea="i001", version=1, stage="feasibility", verdict="FAIL"
    )
    again = draft(title="Fast trend again", spec={"type": "ewmac"}, differs_from={"i001": WHY})
    retested = register(ws, criteria, again, retest="user: a year of new data", commit=False)
    outcome = build_check(ws, criteria, retested, commit=False)
    # Same configuration as the failed i001: allowed by the user's retest, and the fingerprint
    # of the very same spec is skipped as well.
    assert outcome.result.verdict == "PASS", outcome.result.reasons
