import json

import numpy as np
import pandas as pd
import pytest
from research_helpers import commits, git, init_repo, make_workspace
from test_research_journal import register

from algotrade.data.exchange import MarketId, funding_path
from algotrade.research import vcs
from algotrade.research.criteria import Criteria
from algotrade.research.journal import Journal
from algotrade.research.split import (
    FUNDING_ALIGNMENT,
    HoldoutViolation,
    bars_hash,
    dev_end,
    dev_universe,
    for_version,
    funding_alignment,
    load_dev_bars,
    load_full_bars,
    open_holdout,
    window,
)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_workspace(tmp_path_factory.mktemp("split"), repo=False)


# --- split ----------------------------------------------------------------------------------


def test_dev_bars_stop_before_the_cutoff(setup) -> None:
    ws, criteria = setup
    cutoff = dev_end(criteria)
    assert cutoff == pd.Timestamp("2025-05-01", tz="UTC")
    full = load_full_bars(ws, criteria, "BTC", "4h")
    dev = load_dev_bars(ws, criteria, "BTC", "4h")
    assert full.index[-1] > cutoff
    assert dev.index[-1] < cutoff and dev.index[-1] == full.index[full.index < cutoff][-1]
    # The last development bar closes exactly at the cutoff.
    assert dev.index[-1] + pd.Timedelta("4h") == cutoff


def test_dev_bars_window_and_refusal(setup) -> None:
    ws, criteria = setup
    part = load_dev_bars(ws, criteria, "ETH", "1d", start="2024-01-01", end="2024-02-01")
    assert part.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert part.index[-1] == pd.Timestamp("2024-01-31", tz="UTC")
    naive_end = load_dev_bars(ws, criteria, "ETH", "1d", end=pd.Timestamp("2025-05-01"))
    assert naive_end.index[-1] < dev_end(criteria)
    with pytest.raises(HoldoutViolation, match="development data ends at 2025-05-01"):
        load_dev_bars(ws, criteria, "ETH", "1d", end="2025-05-02")


def test_dev_universe_uses_criteria_symbols(setup) -> None:
    ws, criteria = setup
    universe = dev_universe(ws, criteria, "1d")
    assert list(universe) == ["BTC", "ETH", "SOL"]
    assert list(dev_universe(ws, criteria, "1d", ["SOL"])) == ["SOL"]


def test_bars_hash_and_window(setup) -> None:
    ws, criteria = setup
    bars = load_dev_bars(ws, criteria, "BTC", "1d")
    assert bars_hash(bars) == bars_hash(bars.copy())
    changed = bars.copy()
    changed.iloc[5, changed.columns.get_loc("close")] *= 1.0001
    assert bars_hash(changed) != bars_hash(bars)
    w = window(bars, "BTC", "1d")
    assert w.rows == len(bars) and w.sha == bars_hash(bars)
    assert w.to_dict()["symbol"] == "BTC"
    empty = window(bars.iloc[:0], "BTC", "1d")
    assert empty.start == "" and empty.rows == 0


def test_funding_alignment_is_legacy_when_criteria_do_not_name_one() -> None:
    assert funding_alignment(Criteria(values={"data": {}}, hash="x")) == "raw_timestamp"
    fixed = Criteria(values={"data": {"funding_alignment": "nearest_second"}}, hash="x")
    assert funding_alignment(fixed) == "nearest_second"


def test_loaders_align_funding_as_the_criteria_say(tmp_path) -> None:
    ws, criteria = make_workspace(tmp_path, repo=False)
    path = funding_path(ws.raw_data, MarketId("binanceusdm", "BTC"))
    on_the_hour = pd.read_parquet(path)
    expected = load_full_bars(ws, criteria, "BTC", "4h")["funding_rate"]
    # Stamp every settlement 23 ms late, as Binance does.
    late = on_the_hour.assign(timestamp=on_the_hour["timestamp"] + pd.Timedelta("23ms"))
    late.to_parquet(path, index=False)

    fixed = criteria.with_value(FUNDING_ALIGNMENT, "nearest_second")
    pd.testing.assert_series_equal(load_full_bars(ws, fixed, "BTC", "4h")["funding_rate"], expected)
    dev = load_dev_bars(ws, fixed, "BTC", "4h")["funding_rate"]
    pd.testing.assert_series_equal(dev, expected[expected.index < dev_end(criteria)])
    # The legacy alignment charges every one of them a bar late.
    raw = criteria.with_value(FUNDING_ALIGNMENT, "raw_timestamp")
    legacy = load_full_bars(ws, raw, "BTC", "4h")["funding_rate"].to_numpy()
    assert np.array_equal(legacy[1:], expected.to_numpy()[:-1]) and legacy[0] != 0


def test_a_version_keeps_the_alignment_of_its_recorded_results(tmp_path) -> None:
    ws, criteria = make_workspace(tmp_path, repo=False, market=False)
    journal = Journal(ws.journal_path)
    register(journal)
    version = journal.ideas()["i001"].latest
    fixed = criteria.with_value(FUNDING_ALIGNMENT, "nearest_second")
    # No result yet: the version follows criteria.yaml.
    assert for_version(ws, fixed, version) is fixed
    assert for_version(ws, criteria, version) is criteria

    # A result recorded before data.funding_alignment existed was made with the legacy one.
    build = ws.stage_dir("i001", "fast-trend", 1, "build")
    build.mkdir(parents=True)
    (build / "result.json").write_text(json.dumps({"provenance": {"criteria_hash": "x"}}))
    pinned = for_version(ws, fixed, version)
    assert funding_alignment(pinned) == "raw_timestamp" and pinned.hash == fixed.hash
    assert funding_alignment(fixed) == "nearest_second"  # the criteria themselves are untouched

    # Whatever a result recorded wins over criteria.yaml.
    provenance = {"provenance": {"funding_alignment": "nearest_second"}}
    (build / "result.json").write_text(json.dumps(provenance))
    assert funding_alignment(for_version(ws, criteria, version)) == "nearest_second"


def test_holdout_is_looked_at_once_unless_forced(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    register(journal)
    version = journal.ideas()["i001"].latest
    open_holdout(journal, version)
    version = journal.ideas()["i001"].latest
    assert version.holdout_looks == 1
    with pytest.raises(HoldoutViolation, match="already looked"):
        open_holdout(journal, version)
    open_holdout(journal, version, force=True, reason="user asked")
    looks = [e for e in journal.entries() if e["event"] == "holdout_look"]
    assert [e["forced"] for e in looks] == [False, True]
    assert looks[1]["reason"] == "user asked"


# --- vcs ------------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "base.txt").write_text("base\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_commit_stages_only_the_given_paths(repo) -> None:
    (repo / "unrelated.txt").write_text("work in progress\n")
    (repo / "base.txt").write_text("edited by the user\n")
    git(repo, "add", "base.txt")  # even something the user staged stays out
    idea = repo / "research" / "ideas" / "i001"
    idea.mkdir(parents=True)
    (idea / "idea.md").write_text("card\n")
    sha = vcs.commit(repo, [idea, repo / "missing.txt"], "research(i001): registered", "body")
    assert sha == vcs.head(repo)
    shown = git(repo, "show", "--name-only", "--format=%B", "HEAD")
    assert "research/ideas/i001/idea.md" in shown
    assert "unrelated.txt" not in shown and "base.txt" not in shown
    assert vcs.TRAILER in shown and "body" in shown
    status = git(repo, "status", "--porcelain")
    assert "?? unrelated.txt" in status and "M  base.txt" in status


def test_commit_without_changes_returns_none(repo) -> None:
    assert vcs.commit(repo, [repo / "base.txt"], "nothing") is None
    assert vcs.commit(repo, [repo / "nope"], "nothing") is None
    assert commits(repo) == ["base"]


def test_dirty_and_require_clean(repo) -> None:
    code = repo / "strategy.py"
    assert vcs.dirty(repo, [code]) == []
    code.write_text("x = 1\n")
    assert vcs.dirty(repo, [code]) == ["strategy.py"]
    with pytest.raises(vcs.GitError, match="uncommitted changes"):
        vcs.require_clean(repo, [code], "i001 code")
    vcs.commit(repo, [code], "add code")
    vcs.require_clean(repo, [code], "i001 code")


def test_tag_head_and_repo_detection(repo, tmp_path_factory) -> None:
    vcs.tag(repo, "strategy/i001-v1", "frozen")
    assert "strategy/i001-v1" in git(repo, "tag")
    assert vcs.is_repo(repo)
    outside = tmp_path_factory.mktemp("not_a_repo")
    assert not vcs.is_repo(outside)
    assert vcs.head(outside) is None
    with pytest.raises(vcs.GitError):
        vcs.git(outside, "log")


def test_failed_commit_raises(repo, monkeypatch) -> None:
    (repo / "x.txt").write_text("x\n")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(vcs.GitError, match="git commit failed"):
        vcs.commit(repo, [repo / "x.txt"], "blocked by hook")
