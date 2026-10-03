import json
import shutil

import numpy as np
import pandas as pd
import pytest
from research_helpers import commits, draft, git, make_workspace

from algotrade.data.exchange import MarketId, funding_path, ohlcv_path
from algotrade.data.market import FUNDING_ALIGNMENTS
from algotrade.research import holdout as holdout_module
from algotrade.research import validate as validate_module
from algotrade.research.buildcheck import build_check
from algotrade.research.criteria import load_criteria
from algotrade.research.feasibility import feasibility
from algotrade.research.fingerprint import FingerprintStore
from algotrade.research.freeze import freeze, weaknesses
from algotrade.research.holdout import holdout, load_result
from algotrade.research.journal import Journal
from algotrade.research.registry import Refused, find_version, register
from algotrade.research.reproduce import reproduce
from algotrade.research.split import FUNDING_ALIGNMENT, HoldoutViolation, recorded_alignment
from algotrade.research.validate import deflation, validate
from algotrade.research.workspace import Workspace
from scripts import research as cli

FAST = {
    "feasibility.monkey_runs": 100,
    "validation.in_sample_years": 1,
    "validation.out_of_sample_months": 3,
    "validation.monte_carlo_runs": 400,
}
LENIENT = {
    **FAST,
    "feasibility.entry_min_profitable_share": 0.0,
    "feasibility.core_min_median_sharpe": -10.0,
    "feasibility.core_min_positive_share": 0.0,
    "feasibility.min_trades_per_symbol": 0,
    "feasibility.monkey_min_percentile": 0.0,
    "feasibility.optimise_min_profitable_share": 0.0,
}
GRID = {"fast": [8, 16], "slow": [64, 128]}


def through_feasibility(root, overrides=FAST, **market):
    ws, criteria = make_workspace(root, overrides=overrides, **market)
    version = register(ws, criteria, draft(spec={"type": "ewmac"}, optimise=GRID))
    build_check(ws, criteria, version)
    assert feasibility(ws, criteria, find_version(ws, "i001"), report=False).verdict == "PASS"
    return ws, criteria


@pytest.fixture(scope="module")
def validated(tmp_path_factory):
    root = tmp_path_factory.mktemp("validated") / "repo"
    ws, criteria = through_feasibility(root, edge=0.002)
    result = validate(ws, criteria, find_version(ws, "i001"))
    return ws, criteria, result


@pytest.fixture
def copy(validated, tmp_path):
    """A private copy of the validated workspace (git history included)."""

    ws, _, _ = validated
    root = tmp_path / "repo"
    shutil.copytree(ws.root, root)
    copied = Workspace(root)
    return copied, load_criteria(copied.criteria_path)


# --- validation ---------------------------------------------------------------------------------


def test_validation_passes_on_a_planted_edge(validated) -> None:
    ws, criteria, result = validated
    assert result.verdict == "PASS", result.reasons
    names = [c.name for c in result.checks]
    assert names[:4] == [
        "walk-forward efficiency (OOS / IS annualised return)",
        "out-of-sample median Sharpe across symbols",
        "share of out-of-sample windows profitable",
        "deflated Sharpe ratio (probability the edge is real)",
    ]
    assert any(name.startswith("Monte Carlo risk of ruin at") for name in names)
    folder = ws.stage_dir("i001", "fast-trend", 1, "validation")
    for name in (
        "windows.csv",
        "oos_by_symbol.csv",
        "monte_carlo.csv",
        "oos_trade_returns.csv",
        "oos_returns.parquet",
        "summary.md",
    ):
        assert (folder / name).exists(), name  # fmt: skip
    saved = json.loads((folder / "result.json").read_text())
    assert saved["stake"] in criteria.get("validation.stake_multipliers")
    assert saved["deflated_sharpe"]["trials"] == 4 and saved["trials"] == 0
    assert all(w["end"] < "2025-05-01" for w in saved["provenance"]["data"])
    assert (ws.root / saved["report"]).exists()
    assert "Walk-forward windows" in (folder / "summary.md").read_text(encoding="utf-8")
    assert commits(ws.root)[0] == "research(i001-fast-trend v1): validation PASS"


def test_deflation_uses_the_whole_trial_ledger(validated) -> None:
    ws, _, _ = validated
    returns = pd.read_parquet(
        ws.stage_dir("i001", "fast-trend", 1, "validation") / "oos_returns.parquet"
    )
    results = {}
    for symbol in returns.columns:
        series = returns[symbol].dropna().astype(float)
        ledger = pd.DataFrame({"net": series, "equity": (1 + series).cumprod()})
        results[symbol] = validate_module.BacktestResult(ledger=ledger, trades=None, costs=None)
    found = deflation(ws, results)
    trials = pd.read_csv(ws.trials_path)
    expected_variance = trials["metric"].var(ddof=1) / found["periods_per_year"]
    assert found["trials"] == 4
    assert found["trial_sharpe_variance_per_bar"] == pytest.approx(expected_variance)
    assert 0 <= found["deflated_sharpe"] <= 1


def test_validation_needs_a_passing_feasibility(tmp_path) -> None:
    ws, criteria = make_workspace(tmp_path, overrides=FAST)
    version = register(ws, criteria, draft(spec={"type": "ewmac"}))
    build_check(ws, criteria, version)
    with pytest.raises(Refused, match="needs a passing feasibility"):
        validate(ws, criteria, find_version(ws, "i001"))


def test_random_walk_fails_validation(tmp_path) -> None:
    ws, criteria = through_feasibility(tmp_path, overrides=LENIENT, waves=False)
    result = validate(ws, criteria, find_version(ws, "i001"), report=False)
    assert result.verdict == "FAIL"
    failed = {c.name for c in result.checks if not c.passed}
    assert "deflated Sharpe ratio (probability the edge is real)" in failed
    assert "out-of-sample median Sharpe across symbols" in failed
    assert Journal(ws.journal_path).ideas()["i001"].status == "failed:validation"


def test_validation_without_any_window(tmp_path) -> None:
    ws, criteria = through_feasibility(
        tmp_path, overrides={**FAST, "validation.in_sample_years": 10}, edge=0.002
    )
    result = validate(ws, criteria, find_version(ws, "i001"), report=True, commit=False)
    assert result.verdict == "FAIL" and result.report is None
    assert "no walk-forward window had enough in-sample data" in result.reasons
    assert any("Monte Carlo needs 10" in reason for reason in result.reasons)


def test_validation_with_too_few_trades_for_monte_carlo(tmp_path, monkeypatch) -> None:
    ws, criteria = through_feasibility(tmp_path, edge=0.002)
    monkeypatch.setattr(
        validate_module, "closed_trade_returns", lambda results: np.array([0.1] * 3)
    )
    result = validate(ws, criteria, find_version(ws, "i001"), report=False, commit=False)
    assert "only 3 out-of-sample trades; Monte Carlo needs 10" in result.reasons
    assert not any(c.name.startswith("Monte Carlo") for c in result.checks)
    assert result.extra["stake"] is None


# --- holdout and freeze -----------------------------------------------------------------------------


def test_holdout_once_then_freeze(copy) -> None:
    ws, criteria = copy
    with pytest.raises(Refused, match="needs a passing holdout"):
        freeze(ws, criteria, find_version(ws, "i001"))
    result = holdout(ws, criteria, find_version(ws, "i001"))
    assert result.verdict == "PASS", result.reasons
    saved = json.loads(
        (ws.stage_dir("i001", "fast-trend", 1, "holdout") / "result.json").read_text()
    )
    assert all(w["start"] >= "2025-05-01" for w in saved["provenance"]["data"])
    assert (
        saved["spec"]
        == json.loads(
            (ws.stage_dir("i001", "fast-trend", 1, "feasibility") / "result.json").read_text()
        )["chosen_spec"]
    )
    assert (ws.root / saved["report"]).exists()
    looks = [e for e in Journal(ws.journal_path).entries() if e["event"] == "holdout_look"]
    assert len(looks) == 1 and looks[0]["forced"] is False
    with pytest.raises(Refused, match="already has a holdout result"):
        holdout(ws, criteria, find_version(ws, "i001"))

    frozen = freeze(ws, criteria, find_version(ws, "i001"))
    folder = ws.version_dir("i001", "fast-trend", 1)
    stored = json.loads((folder / "frozen.json").read_text())
    assert (
        stored == frozen and frozen["spec"] == saved["spec"] and frozen["tag"] == "strategy/i001-v1"
    )
    assert "strategy/i001-v1" in git(ws.root, "tag")
    decision = (folder / "decision.md").read_text(encoding="utf-8")
    assert "| holdout | holdout median Sharpe across symbols |" in decision
    assert "Deflated Sharpe ratio" in decision and "## Known weaknesses" in decision
    assert "- holdout: `reports/research/i001/v1/holdout/report.html`" in decision
    assert Journal(ws.journal_path).ideas()["i001"].status == "frozen"
    assert commits(ws.root)[0] == "research(i001-fast-trend v1): frozen"
    with pytest.raises(Refused, match="already frozen"):
        freeze(ws, criteria, find_version(ws, "i001"))


def test_forced_second_look_is_journaled(copy) -> None:
    ws, criteria = copy
    holdout(ws, criteria, find_version(ws, "i001"), report=False)
    again = holdout(ws, criteria, find_version(ws, "i001"), force=True, reason="user: bug fix",
                    report=False)  # fmt: skip
    looks = [e for e in Journal(ws.journal_path).entries() if e["event"] == "holdout_look"]
    assert [look["forced"] for look in looks] == [False, True]
    assert looks[1]["reason"] == "user: bug fix"
    assert "forced: user: bug fix" in again.notes[0]


def test_a_look_without_a_result_still_counts(copy) -> None:
    ws, criteria = copy
    Journal(ws.journal_path).append("holdout_look", idea="i001", version=1, reason="crashed run")
    with pytest.raises(HoldoutViolation, match="already looked"):
        holdout(ws, criteria, find_version(ws, "i001"), report=False)


def test_holdout_without_a_band(copy, monkeypatch) -> None:
    ws, criteria = copy

    def no_band(*args, **kwargs):
        raise ValueError("too short")

    monkeypatch.setattr(holdout_module, "block_bootstrap", no_band)
    result = holdout(ws, criteria, find_version(ws, "i001"), report=False, commit=False)
    assert result.verdict == "FAIL"
    assert (
        "not enough out-of-sample data or holdout bars to build the drawdown band" in result.reasons
    )


def test_weaknesses_read_the_diagnostics(copy) -> None:
    ws, _ = copy
    version = find_version(ws, "i001")
    folder = ws.stage_dir("i001", "fast-trend", 1, "feasibility")
    pd.DataFrame({"trades": [10, 4, 6], "sum_return": [0.2, 0.5, -0.3]},
                 index=["all", "long", "short"]).to_csv(folder / "diagnostics_sides.csv")  # fmt: skip
    pd.DataFrame(
        {"kind": ["trend", "trend", "trend"], "regime": ["ranging (ADX<=20)", "warm-up", "neither"],
         "share_of_time": [0.3, 0.2, 0.05], "median_sharpe": [-0.8, -2.0, -1.0]}
    ).to_csv(folder / "diagnostics_regimes.csv", index=False)  # fmt: skip
    oos = ws.stage_dir("i001", "fast-trend", 1, "validation") / "oos_by_symbol.csv"
    pd.DataFrame({"sharpe": [0.5, -0.2]}, index=["BTC", "ETH"]).to_csv(oos)
    found = weaknesses(ws, version)
    assert found == [
        "short trades lost money overall in development data",
        "loses in the 'ranging (ADX<=20)' regime (30% of the time, median Sharpe -0.80)",
        "negative out-of-sample Sharpe on ETH",
    ]


# --- reproducing reports --------------------------------------------------------------------------


def test_reports_are_reproduced_from_the_committed_results(copy) -> None:
    ws, criteria = copy
    holdout(ws, criteria, find_version(ws, "i001"), report=False)
    version = find_version(ws, "i001")
    for stage in ("feasibility", "validation", "holdout"):
        page = reproduce(ws, criteria, version, stage)
        assert page == f"reports/research/i001/v1/{stage}/report.html"
        assert (ws.root / page).exists()
    with pytest.raises(Refused, match="no report to reproduce"):
        reproduce(ws, criteria, version, "build")


def test_reproduction_refuses_changed_data_or_numbers(copy) -> None:
    ws, criteria = copy
    version = find_version(ws, "i001")
    path = ws.stage_dir("i001", "fast-trend", 1, "feasibility") / "result.json"
    saved = json.loads(path.read_text())
    saved["metrics"]["median Sharpe at the chosen parameters"] += 0.5
    path.write_text(json.dumps(saved))
    with pytest.raises(Refused, match="differs from the committed"):
        reproduce(ws, criteria, version, "feasibility")

    bars_file = ohlcv_path(ws.raw_data, MarketId("binanceusdm", "BTC"), "4h")
    bars = pd.read_parquet(bars_file)
    bars.loc[10, "close"] *= 1.01
    bars.to_parquet(bars_file, index=False)
    with pytest.raises(Refused, match="market data changed .* BTC"):
        reproduce(ws, criteria, version, "validation")


def test_a_tested_version_keeps_its_funding_alignment(copy) -> None:
    ws, criteria = copy
    tested = recorded_alignment(load_result(ws, find_version(ws, "i001"), "build"))
    assert tested == criteria.get("data.funding_alignment")
    # The user switches the alignment after i001 v1 was built, tested and validated.
    other = next(name for name in FUNDING_ALIGNMENTS if name != tested)
    switched = criteria.with_value(FUNDING_ALIGNMENT, other)
    assert holdout(ws, switched, find_version(ws, "i001"), report=False).verdict == "PASS"
    frozen = freeze(ws, switched, find_version(ws, "i001"), commit=False)
    version = find_version(ws, "i001")
    for stage in ("build", "feasibility", "validation", "holdout"):
        assert recorded_alignment(load_result(ws, version, stage)) == tested, stage
    assert frozen["funding_alignment"] == tested
    store = FingerprintStore(ws.fingerprints_dir)
    stored = store.keys()  # only i001 v1 was built
    assert [store.load(key)[1]["funding_alignment"] for key in stored] == [tested]


def test_reports_reproduce_with_the_alignment_they_recorded(copy) -> None:
    ws, criteria = copy
    version = find_version(ws, "i001")
    # Stamp every settlement 23 ms late, as Binance does: rounding to the second gives back the
    # data the results were computed on, the raw timestamps no longer do.
    for symbol in ("BTC", "ETH", "SOL"):
        path = funding_path(ws.raw_data, MarketId("binanceusdm", symbol))
        frame = pd.read_parquet(path)
        late = frame.assign(timestamp=frame["timestamp"] + pd.Timedelta("23ms"))
        late.to_parquet(path, index=False)
    path = ws.stage_dir("i001", "fast-trend", 1, "feasibility") / "result.json"
    saved = json.loads(path.read_text())

    def reproduced(recorded: str, now: str) -> str:
        saved["provenance"]["funding_alignment"] = recorded  # as if recorded with it
        path.write_text(json.dumps(saved))
        return reproduce(ws, criteria.with_value(FUNDING_ALIGNMENT, now), version, "feasibility")

    # Recorded with raw timestamps, so reproduced with them whatever the criteria say now.
    with pytest.raises(Refused, match="market data changed"):
        reproduced("raw_timestamp", now="nearest_second")
    page = reproduced("nearest_second", now="raw_timestamp")
    assert page == "reports/research/i001/v1/feasibility/report.html"


# --- command line ------------------------------------------------------------------------------------


def run(ws, *args) -> int:
    return cli.main(["--root", str(ws.root), *args])


def test_cli_holdout_freeze_and_report(copy, capsys) -> None:
    ws, _ = copy
    assert run(ws, "holdout", "i001", "--force") == 1
    assert "needs --reason" in capsys.readouterr().out
    assert run(ws, "holdout", "i001", "--no-report") == 0
    assert "HOLDOUT PASS" in capsys.readouterr().out
    assert run(ws, "freeze", "i001") == 0
    out = capsys.readouterr().out
    assert "Frozen i001 v1 as strategy/i001-v1" in out and "decision.md" in out
    assert run(ws, "report", "i001", "holdout") == 0
    assert "reports/research/i001/v1/holdout/report.html" in capsys.readouterr().out


def test_cli_validate(tmp_path, capsys) -> None:
    ws, _ = through_feasibility(tmp_path, edge=0.002)
    assert run(ws, "validate", "i001", "--no-report") == 0
    out = capsys.readouterr().out
    assert "VALIDATION PASS" in out and "stake " in out


def test_freeze_without_committing_and_weaknesses_without_files(copy) -> None:
    ws, criteria = copy
    holdout(ws, criteria, find_version(ws, "i001"), report=False)
    before = commits(ws.root)
    frozen = freeze(ws, criteria, find_version(ws, "i001"), commit=False)
    assert commits(ws.root) == before and "strategy/" not in git(ws.root, "tag")
    assert frozen["stake"] is not None
    register(ws, criteria, draft(title="Slow", spec={"type": "momentum"}, horizon="weeks"),
             commit=False)  # fmt: skip
    assert weaknesses(ws, find_version(ws, "i002")) == []
    with pytest.raises(Refused, match="has no feasibility result"):
        holdout_module.load_result(ws, find_version(ws, "i002"), "feasibility")


def test_holdout_skips_symbols_without_holdout_data(copy) -> None:
    ws, criteria = copy
    eth = ohlcv_path(ws.raw_data, MarketId("binanceusdm", "ETH"), "4h")
    bars = pd.read_parquet(eth)
    bars[bars["timestamp"] < pd.Timestamp("2025-04-01", tz="UTC")].to_parquet(eth, index=False)
    found = holdout_module.holdout_results(ws, criteria, {"type": "ewmac"}, "4h")
    assert set(found) == {"BTC", "SOL"}


def test_cli_validate_without_a_stake(tmp_path, monkeypatch, capsys) -> None:
    ws, _ = make_workspace(tmp_path, market=False)
    failed = validate_module.results.StageResult("i001", 1, "validation", [], failures=["x"])
    monkeypatch.setattr(cli, "find_version", lambda *args: None)
    monkeypatch.setattr(cli, "validate", lambda *args, **kwargs: failed)
    assert run(ws, "validate", "i001") == 1
    out = capsys.readouterr().out
    assert "stake" not in out and "VALIDATION FAIL" in out
