import json

import pandas as pd
import pytest
from research_helpers import commits, draft, make_workspace, write_draft

from algotrade.research.journal import Journal, TrialLedger
from algotrade.research.registry import Refused, register
from algotrade.research.seed import seed
from scripts import research as cli

WHY = "A moving-average crossover, not a channel breakout with RSI and an ATR bracket."
GRID = ["stop_atr", "target_atr", "atr_length", "median_sharpe", "killed"]


def fake_reports(root) -> None:
    reports = root / "reports"
    (reports / "no_kill").mkdir(parents=True)
    grid = pd.DataFrame(
        [[1.5, 4, 14, 0.25, 0.7], [2.0, 4, 14, 0.10, 0.9], [2.0, 6, 30, -0.05, 1.0]],
        columns=GRID,
    )
    grid.to_csv(reports / "grid_breakout_bracket_4h.csv", index=False)
    grid.assign(median_sharpe=grid["median_sharpe"] - 0.1).to_csv(
        reports / "no_kill" / "grid_breakout_bracket_4h.csv", index=False
    )
    grid.iloc[:2].to_csv(reports / "grid_breakout_bracket_1d.csv", index=False)
    rows = []
    for label, sharpe in (("ewmac", 0.4), ("rsi_reversion", -0.2), ("spec:turtle_55_20", 0.1)):
        for symbol, shift in (("BTC", 0.1), ("ETH", 0.0), ("SOL", -0.1)):
            rows.append({"symbol": symbol, "sharpe": sharpe + shift, "strategy": label})
    scan = pd.DataFrame(rows).set_index("symbol")
    scan.to_csv(reports / "scan_4h_vt.csv")
    # carver_trend is already vol-targeted, so the 1d vol-target scan repeats its configuration.
    carver = pd.DataFrame(
        {"sharpe": [0.3, 0.2, 0.1], "strategy": "spec:carver_trend"},
        index=pd.Index(["BTC", "ETH", "SOL"], name="symbol"),
    )
    pd.concat([scan, carver]).to_csv(reports / "scan_1d.csv")
    carver.to_csv(reports / "scan_1d_vt.csv")


@pytest.fixture
def seeded(tmp_path):
    ws, criteria = make_workspace(tmp_path)
    fake_reports(tmp_path)
    summary = seed(ws, criteria)
    return ws, criteria, summary


def test_seed_records_the_book_strategy_as_failed(seeded) -> None:
    ws, _, summary = seeded
    book = Journal(ws.journal_path).ideas()["i001"]
    assert book.status == "failed:feasibility" and book.latest.historical
    v1, v2 = book.versions[1], book.versions[2]
    assert v1.timeframe == "4h" and v2.timeframe == "1d" and v2.parent == 1
    assert v1.grid == {
        "stop_atr": [1.5, 2.0],
        "target_atr": [4, 6],
        "atr_length": [14, 30],
        "kill_drawdown": [0.5, 1.0],
    }
    assert "kill_drawdown" not in v2.grid
    result = json.loads(
        (ws.stage_dir("i001", v1.slug, 1, "feasibility") / "result.json").read_text()
    )
    assert result["verdict"] == "FAIL" and result["trials"] == 6
    assert "best median Sharpe in the grid 0.25" in result["reasons"][0]
    # Screened: 4 rules on 1d, carver_trend again in the 1d vol-target scan (same config), 3 on 4h.
    # Fingerprints: i001 on 4h and 1d, the 7 distinct screened configurations, buy & hold x 2.
    assert summary == {"book_trials": 8, "screened": 8, "fingerprints": 11, "trials": 15}
    assert commits(ws.root)[0] == "research: seed the journal with pre-journal experiments"


def test_seed_counts_trials_and_writes_the_index(seeded) -> None:
    ws, _, _ = seeded
    trials = TrialLedger(ws.trials_path).frame()
    screened = trials[trials["kind"] == "screen"]
    assert set(screened["label"]) >= {"ewmac (1d)", "ewmac (4h, vol target 25%)"}
    assert screened.loc[screened["label"] == "ewmac (1d)", "metric"].item() == pytest.approx(0.4)
    assert (trials["window"].str.contains("holdout included")).all()
    index = ws.index_path.read_text(encoding="utf-8")
    assert "Book breakout + RSI + ATR bracket (pre-journal)" in index
    assert "## Screened before the journal" in index and "| ewmac (1d) | 1d |" in index
    assert "Configurations evaluated so far: **15**" in index


def test_seeded_book_strategy_cannot_be_retested_but_screened_rules_can(seeded) -> None:
    ws, criteria, _ = seeded
    book = draft(title="Breakout again", spec={"type": "breakout_bracket"}, family="breakout")
    with pytest.raises(Refused, match="Failed ideas stay failed"):
        register(ws, criteria, book, commit=False)
    near = draft(title="Breakout 50", spec={"type": "breakout_bracket", "lookback": 50},
                 family="breakout")  # fmt: skip
    with pytest.raises(Refused, match="overlaps the parameter region of i001"):
        register(ws, criteria, near, commit=False)
    ewmac = draft(title="Carver trend", spec={"type": "ewmac"})
    assert register(ws, criteria, ewmac, commit=False).idea == "i002"


def test_seed_runs_once_and_needs_the_grid_files(seeded, tmp_path_factory) -> None:
    ws, criteria, _ = seeded
    with pytest.raises(Refused, match="already seeded"):
        seed(ws, criteria)
    fresh, fresh_criteria = make_workspace(tmp_path_factory.mktemp("bare"))
    with pytest.raises(Refused, match="grid_breakout_bracket_4h.csv is missing"):
        seed(fresh, fresh_criteria, commit=False)
    quiet, quiet_criteria = make_workspace(tmp_path_factory.mktemp("quiet"))
    fake_reports(quiet.root)
    (quiet.root / "reports" / "scan_1d_vt.csv").unlink()  # scans that were never run are skipped
    before = commits(quiet.root)
    assert seed(quiet, quiet_criteria, commit=False)["book_trials"] == 8
    assert commits(quiet.root) == before


# --- command line -----------------------------------------------------------------------------


def run(ws, *args) -> int:
    return cli.main(["--root", str(ws.root), *args])


def test_cli_full_phase_one_flow(tmp_path, capsys) -> None:
    ws, _ = make_workspace(tmp_path)
    fake_reports(tmp_path)
    assert run(ws, "seed") == 0
    assert "idea i001, failed" in capsys.readouterr().out

    path = write_draft(tmp_path, draft(spec={"type": "ewmac"}, differs_from={"i001": WHY}))
    assert run(ws, "new", str(path)) == 0
    assert "Registered i002 v1" in capsys.readouterr().out

    assert run(ws, "build-check", "i002") == 0
    out = capsys.readouterr().out
    assert "BUILD PASS" in out and "nearest earlier strategy" in out

    assert run(ws, "status") == 0
    out = capsys.readouterr().out
    assert "i001 v2 [failed:feasibility]" in out and "i002 v1 [passed:build]" in out

    revised = write_draft(tmp_path, draft(spec={"type": "ewmac", "allow_short": False}), "v2.md")
    assert run(ws, "revise", "i002", str(revised), "--reason", "test long only") == 0
    assert "Registered i002 v2" in capsys.readouterr().out

    assert run(ws, "abandon", "i002", "--reason", "enough for a test") == 0
    assert "abandoned" in capsys.readouterr().out

    assert run(ws, "new", str(path)) == 1
    assert capsys.readouterr().out.startswith("REFUSED:")


def test_cli_build_check_reports_problems_and_duplicates(tmp_path, capsys) -> None:
    ws, _ = make_workspace(tmp_path)
    path = write_draft(tmp_path, draft(spec={"type": "new_rule"}))
    assert run(ws, "--no-commit", "new", str(path)) == 0
    assert run(ws, "build-check", "i001") == 1
    assert "BUILD NOT READY" in capsys.readouterr().out

    first = write_draft(tmp_path, draft(title="Carver", spec={"type": "ewmac"},
                                        differs_from={"i001": WHY}), "a.md")  # fmt: skip
    assert run(ws, "new", str(first)) == 0 and run(ws, "build-check", "i002") == 0
    copy = {"type": "combine", "strategies": [{"type": "ewmac"}]}
    second = write_draft(tmp_path, draft(title="Copy", spec=copy,
                                         differs_from={"i001": WHY, "i002": WHY}), "b.md")  # fmt: skip
    assert run(ws, "new", str(second)) == 0
    capsys.readouterr()
    assert run(ws, "build-check", "i003") == 1
    out = capsys.readouterr().out
    assert "BUILD FAIL" in out and "behavioural duplicate" in out
