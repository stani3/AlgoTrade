import json
import shutil

import pandas as pd
import pytest
import yaml
from research_helpers import commits, frozen_workspace, make_workspace

from algotrade.research.criteria import load_criteria
from algotrade.research.incubation import (
    FILL_COLUMNS,
    incubation_report,
    read_fills,
    slippage_bps,
    start,
)
from algotrade.research.journal import Journal
from algotrade.research.registry import Refused, find_version
from algotrade.research.workspace import Workspace
from scripts import research as cli


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    return frozen_workspace(tmp_path_factory.mktemp("frozen") / "repo")


@pytest.fixture
def ws(frozen, tmp_path):
    original, _ = frozen
    shutil.copytree(original.root, tmp_path / "repo")
    copy = Workspace(tmp_path / "repo")
    return copy, load_criteria(copy.criteria_path)


def test_start_records_the_frozen_strategy(ws) -> None:
    ws, _ = ws
    record = start(ws, find_version(ws, "i001"), when="2025-08-01")
    assert record["start"] == "2025-08-01T00:00:00+00:00"
    folder = ws.stage_dir("i001", "fast-trend", 1, "incubation")
    stored = json.loads((folder / "started.json").read_text())
    frozen = json.loads((ws.version_dir("i001", "fast-trend", 1) / "frozen.json").read_text())
    assert stored["spec"] == frozen["spec"] and stored["stake"] == frozen["stake"]
    state = Journal(ws.journal_path).ideas()["i001"].latest
    assert state.incubation_start == record["start"] and state.status == "incubating"
    assert commits(ws.root)[0] == "research(i001-fast-trend v1): incubation started"
    with pytest.raises(Refused, match="has been incubating since"):
        start(ws, find_version(ws, "i001"))


def test_start_needs_a_frozen_strategy(tmp_path) -> None:
    from research_helpers import draft

    from algotrade.research.registry import register

    plain, criteria = make_workspace(tmp_path, market=False)
    register(plain, criteria, draft(), commit=False)
    with pytest.raises(Refused, match="is not frozen"):
        start(plain, find_version(plain, "i001"))
    with pytest.raises(Refused, match="is not incubating"):
        incubation_report(plain, criteria, find_version(plain, "i001"))


def test_short_incubation_says_continue_and_logs_a_check_in(ws) -> None:
    ws, criteria = ws
    start(ws, find_version(ws, "i001"), when="2025-12-01")
    report = incubation_report(ws, criteria, find_version(ws, "i001"), report=False)
    assert report.status == "continue" and report.days < 90
    assert any(note.startswith("Continue:") for note in report.result.notes)
    assert "incubation" not in Journal(ws.journal_path).ideas()["i001"].latest.stages
    assert commits(ws.root)[0].startswith("research(i001-fast-trend v1): incubation check-in")
    saved = json.loads(
        (ws.stage_dir("i001", "fast-trend", 1, "incubation") / "result.json").read_text()
    )
    assert saved["status"] == "continue"


def test_long_enough_incubation_passes_with_a_report(ws) -> None:
    ws, _ = ws
    # This slow EWMAC trades rarely; in this private workspace two trades are enough.
    values = yaml.safe_load(ws.criteria_path.read_text(encoding="utf-8"))
    values["incubation"]["min_trades"] = 2
    ws.criteria_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    criteria = load_criteria(ws.criteria_path)
    start(ws, find_version(ws, "i001"), when="2025-05-15")
    report = incubation_report(ws, criteria, find_version(ws, "i001"))
    assert report.days >= 90 and report.trades >= 2
    assert report.status == "pass", report.result.reasons
    assert Journal(ws.journal_path).ideas()["i001"].status == "passed:incubation"
    assert (ws.root / report.result.report).exists()
    assert "No testnet fills logged yet" in " ".join(report.result.notes)
    with pytest.raises(Refused, match="already has an incubation verdict"):
        incubation_report(ws, criteria, find_version(ws, "i001"))


def test_slippage_from_testnet_fills_is_scored(ws) -> None:
    ws, criteria = ws
    start(ws, find_version(ws, "i001"), when="2025-12-01")
    folder = ws.stage_dir("i001", "fast-trend", 1, "incubation")
    fills = pd.DataFrame(
        [["2025-12-02T00:00:00Z", "BTC", 1, 0.1, 101.0, 100.0],
         ["2025-12-03T00:00:00Z", "BTC", -1, 0.1, 99.0, 100.0]],
        columns=FILL_COLUMNS,
    )  # fmt: skip
    fills.to_csv(folder / "fills.csv", index=False)
    assert slippage_bps(read_fills(folder)).tolist() == pytest.approx([100.0, 100.0])
    report = incubation_report(ws, criteria, find_version(ws, "i001"), report=False, commit=False)
    slip = [c for c in report.result.checks if c.name.startswith("median testnet slippage")]
    assert slip and not slip[0].passed and report.status == "fail"


def test_no_bars_after_the_start_yet(ws) -> None:
    ws, criteria = ws
    start(ws, find_version(ws, "i001"), when="2027-01-01")
    report = incubation_report(ws, criteria, find_version(ws, "i001"), report=True, commit=False)
    assert report.status == "continue" and report.days == 0 and report.result.report is None
    assert any("No bars after the start yet" in note for note in report.result.notes)
    assert read_fills(ws.root).empty


def test_cli_incubate_and_report(ws, capsys) -> None:
    ws, _ = ws
    assert cli.main(["--root", str(ws.root), "incubate", "i001", "--start", "2025-12-01"]) == 0
    assert "Incubation of i001 v1 started" in capsys.readouterr().out
    assert cli.main(["--root", str(ws.root), "incubation-report", "i001", "--no-report"]) == 0
    out = capsys.readouterr().out
    assert "INCUBATION CONTINUE" in out and "days" in out


def test_start_without_commit_and_cli_report_path(ws, capsys) -> None:
    ws, _ = ws
    before = commits(ws.root)
    start(ws, find_version(ws, "i001"), when="2025-12-01", commit=False)
    assert commits(ws.root) == before
    assert cli.main(["--root", str(ws.root), "--no-commit", "incubation-report", "i001"]) == 0
    assert "report: reports/research/i001/v1/incubation/report.html" in capsys.readouterr().out
