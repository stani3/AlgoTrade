import io
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
from research_helpers import make_workspace
from test_research_journal import register as journal_register

from algotrade.research import fingerprint
from algotrade.research.buildcheck import BuildOutcome
from algotrade.research.criteria import at_least
from algotrade.research.dedup import Region, region, structure
from algotrade.research.index import write_index
from algotrade.research.journal import Journal
from algotrade.research.results import StageResult, summary_markdown
from algotrade.research.split import bars_hash
from algotrade.research.workspace import Workspace
from scripts import research as cli


def test_workspace_paths(tmp_path) -> None:
    ws = Workspace(tmp_path)
    assert ws.stage_dir("i001", "x", 2, "build") == tmp_path / "research/ideas/i001-x/v2/build"
    assert ws.report_dir("i001", 2, "validation") == (
        tmp_path / "reports/research/i001/v2/validation"
    )
    assert ws.raw_data == tmp_path / "data" / "raw"
    assert Workspace(tmp_path, data_root=tmp_path / "d").raw_data == tmp_path / "d"
    with pytest.raises(ValueError, match="unknown stage"):
        ws.stage_dir("i001", "x", 1, "deploy")


def test_structure_of_blends_and_regions_with_different_paths() -> None:
    blend = {"type": "combine", "strategies": [{"type": "ewmac"}, {"type": "carver_breakout"}]}
    assert "carver_breakout" in structure(blend) and "ewmac" in structure(blend)
    a = Region("s", "4h", {"x": (1.0, 2.0)}, {"flag": frozenset({"true"})})
    b = Region("s", "4h", {"y": (5.0, 6.0)}, {"other": frozenset({"x"})})
    assert a.overlaps(b)  # nothing shared to tell them apart
    weights = region(blend, {}, "4h", 0.2)
    assert "weights.0" in weights.numeric


def test_nearest_skips_fingerprints_that_barely_overlap(tmp_path) -> None:
    _, criteria = make_workspace(tmp_path / "ws", repo=False, market=False)
    store = fingerprint.FingerprintStore(tmp_path / "fp")
    days = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    store.save("short_4h", pd.DataFrame({"BTC": 1.0}, index=days), {"label": "short"})
    long_days = pd.date_range("2023-01-01", periods=400, freq="D", tz="UTC")
    assert store.nearest(pd.DataFrame({"BTC": 1.0}, index=long_days), criteria) == []


def test_index_without_ideas(tmp_path) -> None:
    ws = Workspace(tmp_path)
    text = write_index(ws).read_text(encoding="utf-8")
    assert "| - | none yet |" in text and "- none yet" in text


def test_journal_ignores_unknown_events_for_known_ideas(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    journal_register(journal)
    journal.append("note", idea="i001", version=1, text="hello")
    assert journal.ideas()["i001"].status == "registered"


def test_summary_markdown_shows_missing_values_and_report() -> None:
    result = StageResult(
        "i001",
        1,
        "feasibility",
        [at_least("Sharpe", float("nan"), 0.3)],
        metrics={"monkey percentile": float("nan"), "trades": 12},
        report="reports/research/i001/v1/feasibility/report.html",
    )
    text = summary_markdown(result, "Fast trend")
    assert "| monkey percentile | n/a |" in text and "| trades | 12 |" in text
    assert "Report: `reports/research/i001/v1/feasibility/report.html`" in text
    assert result.verdict == "FAIL" and result.reasons == ["Sharpe n/a (needs >= 0.3)"]


def test_bars_hash_without_funding() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0, 3.0]}, index=index
    )
    assert bars_hash(bars) != bars_hash(bars.assign(funding_rate=0.0))


def test_cli_prints_checks_and_survives_plain_stdout(tmp_path, monkeypatch, capsys) -> None:
    ws, _ = make_workspace(tmp_path, market=False)
    result = StageResult("i001", 1, "build", [at_least("line coverage (x)", 0.5, 1.0)])
    monkeypatch.setattr(cli, "find_version", lambda *args: None)
    monkeypatch.setattr(
        cli, "build_check", lambda *args, **kw: BuildOutcome(result, False, ["uncovered"])
    )
    assert cli.main(["--root", str(ws.root), "build-check", "i001"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  line coverage (x): 0.5 (needs >= 1)" in out and "BUILD NOT READY" in out
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert cli.main(["--root", str(ws.root), "status"]) == 0
    assert "configurations evaluated" in sys.stdout.getvalue()


def test_cli_runs_as_a_module(tmp_path, monkeypatch) -> None:
    ws, _ = make_workspace(tmp_path, market=False)
    monkeypatch.setattr(sys, "argv", ["research", "--root", str(ws.root), "status"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(Path(cli.__file__)), run_name="__main__")
    assert exit_info.value.code == 0


def test_print_result_shows_chosen_parameters_and_report(capsys) -> None:
    result = StageResult(
        "i001",
        1,
        "feasibility",
        [at_least("monkey", 0.95, 0.9)],
        report="reports/research/i001/v1/feasibility/report.html",
        extra={"chosen": {"fast": 16}},
    )
    assert cli.print_result(result) == 0
    out = capsys.readouterr().out
    assert "chosen fast = 16" in out and "report: reports/research" in out
    assert "FEASIBILITY PASS" in out
