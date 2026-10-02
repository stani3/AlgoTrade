import json

import pandas as pd
import pytest
import yaml
from research_helpers import commits, draft, make_workspace

from algotrade.research.buildcheck import build_check
from algotrade.research.criteria import load_criteria
from algotrade.research.feasibility import feasibility
from algotrade.research.journal import Journal, TrialLedger
from algotrade.research.registry import Refused, find_version, register
from algotrade.research.vcs import GitError
from scripts import research as cli

FAST = {"feasibility.monkey_runs": 200}
GRID = {"fast": [8, 16], "slow": [64, 128]}


def built(ws, criteria, card) -> None:
    version = register(ws, criteria, card)
    assert build_check(ws, criteria, version).result.verdict == "PASS"


@pytest.fixture
def planted(tmp_path):
    ws, criteria = make_workspace(tmp_path, overrides=FAST, edge=0.002)
    built(ws, criteria, draft(spec={"type": "ewmac"}, optimise=GRID))
    return ws, criteria


def test_planted_edge_passes_feasibility_with_report(planted) -> None:
    ws, criteria = planted
    result = feasibility(ws, criteria, find_version(ws, "i001"))
    assert result.verdict == "PASS", result.reasons
    assert [c.name.split(":")[0] for c in result.checks] == [
        "entry test", "core system", "core system", "core system", "monkey test",
        "limited optimisation",
    ]  # fmt: skip
    assert set(result.extra["chosen"]) == {"fast", "slow"}
    assert result.extra["chosen_spec"]["type"] == "ewmac"

    folder = ws.stage_dir("i001", "fast-trend", 1, "feasibility")
    for name in (
        "result.json",
        "summary.md",
        "entry_test.csv",
        "grid.csv",
        "core_by_symbol.csv",
        "diagnostics_sides.csv",
        "diagnostics_regimes.csv",
        "diagnostics_trades.csv",
    ):
        assert (folder / name).exists(), name  # fmt: skip
    saved = json.loads((folder / "result.json").read_text())
    assert saved["verdict"] == "PASS" and saved["trials"] == 4
    assert all(w["end"] < "2025-05-01" for w in saved["provenance"]["data"])
    summary = (folder / "summary.md").read_text(encoding="utf-8")
    assert "Long against short" in summary and "| all |" in summary
    assert (ws.root / saved["report"]).exists()
    assert saved["report"] == "reports/research/i001/v1/feasibility/report.html"

    trials = TrialLedger(ws.trials_path).frame()
    assert len(trials) == 4 and set(trials["kind"]) == {"grid"}
    assert Journal(ws.journal_path).ideas()["i001"].status == "passed:feasibility"
    assert commits(ws.root)[0] == "research(i001-fast-trend v1): feasibility PASS"


def test_random_walk_fails_feasibility(tmp_path) -> None:
    ws, criteria = make_workspace(tmp_path, overrides=FAST, waves=False)
    built(ws, criteria, draft(spec={"type": "ewmac"}, optimise=GRID))
    result = feasibility(ws, criteria, find_version(ws, "i001"), report=False)
    assert result.verdict == "FAIL"
    failed = {c.name.split(":")[0] for c in result.checks if not c.passed}
    assert {"entry test", "monkey test", "limited optimisation"} <= failed
    assert Journal(ws.journal_path).ideas()["i001"].status == "failed:feasibility"
    assert commits(ws.root)[0].startswith("research(i001-fast-trend v1): feasibility FAIL - ")
    with pytest.raises(Refused, match="closed"):
        feasibility(ws, criteria, find_version(ws, "i001"), report=False)


@pytest.mark.parametrize("scoring", [None, "compounded", "fixed"])
def test_card_without_grid_skips_the_optimisation_check(tmp_path, scoring) -> None:
    overrides = FAST if scoring is None else {**FAST, "feasibility.entry_scoring": scoring}
    ws, criteria = make_workspace(tmp_path, overrides=overrides, edge=0.002)
    if scoring is None:  # an older criteria file, written before the key existed
        values = yaml.safe_load(ws.criteria_path.read_text(encoding="utf-8"))
        values["feasibility"].pop("entry_scoring", None)
        ws.criteria_path.write_text(yaml.safe_dump(values), encoding="utf-8")
        criteria = load_criteria(ws.criteria_path)
    built(ws, criteria, draft(spec={"type": "ewmac"}))
    result = feasibility(ws, criteria, find_version(ws, "i001"), report=False, commit=False)
    assert not any(c.name.startswith("limited optimisation") for c in result.checks)
    assert result.extra["chosen"] == {} and result.trials == 1
    assert result.report is None
    # Criteria files without the key keep the original (compounded) scoring.
    expected = "fixed size per trade" if scoring == "fixed" else "whole equity per trade"
    assert expected in result.notes[0]
    table = pd.read_csv(ws.stage_dir("i001", "fast-trend", 1, "feasibility") / "entry_test.csv")
    column = "fixed_return" if scoring == "fixed" else "net_return"
    share = ((table[column] > 0) & (table["trades"] > 0)).mean()
    assert result.checks[0].value == pytest.approx(share)


def test_base_spec_outside_the_grid_is_counted_too(tmp_path) -> None:
    ws, criteria = make_workspace(tmp_path, overrides=FAST, edge=0.002)
    built(ws, criteria, draft(spec={"type": "ewmac", "fast": 12}, optimise=GRID))
    feasibility(ws, criteria, find_version(ws, "i001"), report=False, commit=False)
    trials = TrialLedger(ws.trials_path).frame()
    assert len(trials) == 5 and (trials["kind"] == "base").sum() == 1


def test_feasibility_preconditions(planted) -> None:
    ws, criteria = planted
    register(ws, criteria, draft(title="Slow", spec={"type": "momentum"}, horizon="weeks"))
    with pytest.raises(Refused, match="needs a passing build"):
        feasibility(ws, criteria, find_version(ws, "i002"), report=False)

    card = ws.root / find_version(ws, "i001").card_path
    original = card.read_bytes()
    card.write_bytes(original.replace(b"\n", b"\r\n"))  # same text, but not what was committed
    with pytest.raises(GitError, match="uncommitted changes"):
        feasibility(ws, criteria, find_version(ws, "i001"), report=False)
    card.write_bytes(original + b"\nedited\n")
    with pytest.raises(Refused, match="edited after registration"):
        feasibility(ws, criteria, find_version(ws, "i001"), report=False)
    card.write_bytes(original)

    feasibility(ws, criteria, find_version(ws, "i001"), report=False)
    with pytest.raises(Refused, match="already has a feasibility result"):
        feasibility(ws, criteria, find_version(ws, "i001"), report=False)


def test_cli_feasibility(planted, capsys) -> None:
    ws, _ = planted
    code = cli.main(["--root", str(ws.root), "feasibility", "i001", "--no-report"])
    out = capsys.readouterr().out
    assert code == 0 and "FEASIBILITY PASS" in out and "chosen fast = " in out
    assert "PASS  monkey test" in out


def test_cli_reports_a_failure(tmp_path, capsys) -> None:
    ws, criteria = make_workspace(tmp_path, overrides=FAST, waves=False)
    built(ws, criteria, draft(spec={"type": "ewmac"}))
    code = cli.main(["--root", str(ws.root), "feasibility", "i001", "--no-report"])
    out = capsys.readouterr().out
    assert code == 1 and "FEASIBILITY FAIL" in out and "  - " in out
    assert pd.read_csv(ws.trials_path)["kind"].tolist() == ["base"]
