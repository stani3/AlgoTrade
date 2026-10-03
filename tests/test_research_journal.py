import json

from algotrade.research.journal import TRIAL_COLUMNS, Journal, TrialLedger


def register(journal: Journal, idea: str = "i001", version: int = 1, **extra) -> None:
    journal.append(
        "idea_registered",
        idea=idea,
        version=version,
        slug="fast-trend",
        title="Fast trend",
        timeframe="4h",
        spec={"type": "ma_crossover"},
        grid={},
        taxonomy={"family": "trend"},
        card_hash="abc",
        card_path="research/ideas/i001-fast-trend/v1/idea.md",
        **extra,
    )


def test_empty_journal(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    assert journal.entries() == []
    assert journal.ideas() == {}
    assert journal.next_id() == "i001"
    assert not journal.seeded()


def test_append_only_lines_and_state(tmp_path) -> None:
    journal = Journal(tmp_path / "research" / "journal.jsonl")
    register(journal)
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["event"] == "idea_registered"
    state = journal.ideas()["i001"]
    assert state.status == "registered" and not state.failed
    assert state.slug == "fast-trend" and state.title == "Fast trend"
    assert state.latest.last_stage is None and state.latest.failure == ""
    assert journal.next_id() == "i002"


def test_status_follows_stage_verdicts(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    register(journal)
    journal.append("stage_result", idea="i001", version=1, stage="build", verdict="PASS")
    assert journal.ideas()["i001"].status == "passed:build"
    journal.append(
        "stage_result",
        idea="i001",
        version=1,
        stage="feasibility",
        verdict="FAIL",
        failures=["monkey 0.62 (needs >= 0.9)"],
    )
    state = journal.ideas()["i001"]
    assert state.status == "failed:feasibility" and state.failed
    assert state.latest.failure == "monkey 0.62 (needs >= 0.9)"
    assert state.latest.last_stage == "feasibility"


def test_failure_without_reasons_names_the_stage(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    register(journal)
    journal.append("stage_result", idea="i001", version=1, stage="build", verdict="FAIL")
    assert journal.ideas()["i001"].latest.failure == "build failed"


def test_versions_holdout_freeze_and_abandon(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    register(journal)
    register(journal, version=2, parent=1, reason="long only")
    journal.append("holdout_look", idea="i001", version=2)
    journal.append("frozen", idea="i001", version=2, tag="strategy/i001-v2")
    journal.append("stage_result", idea="i001", version=2, stage="holdout", verdict="PASS")
    state = journal.ideas()["i001"]
    assert sorted(state.versions) == [1, 2]
    assert state.latest.version == 2 and state.latest.parent == 1
    assert state.latest.holdout_looks == 1
    assert state.status == "frozen"
    journal.append("abandoned", idea="i001", version=2, reason="testnet adapter impossible")
    state = journal.ideas()["i001"]
    assert state.status == "abandoned" and state.failed
    assert state.latest.failure == "testnet adapter impossible"


def test_events_for_unknown_ideas_or_versions_are_ignored(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    journal.append("stage_result", idea="i009", version=1, stage="build", verdict="PASS")
    register(journal)
    journal.append("stage_result", idea="i001", version=5, stage="build", verdict="PASS")
    journal.append("seeded", trials=3)
    journal.append("unrelated")
    assert journal.ideas()["i001"].status == "registered"
    assert journal.seeded()
    journal.path.write_text(journal.path.read_text() + "\n\n", encoding="utf-8")
    assert len(journal.entries()) == 5


def test_trial_ledger_counts_distinct_configurations(tmp_path) -> None:
    ledger = TrialLedger(tmp_path / "trials.csv")
    assert ledger.count() == 0 and list(ledger.frame().columns) == [*TRIAL_COLUMNS, "universe"]
    assert ledger.add([]) == 0
    rows = [
        {"idea": "i001", "spec_hash": "aaa", "timeframe": "4h", "metric": 0.1, "spec": {"a": 1}},
        {"idea": "i001", "spec_hash": "aaa", "timeframe": "1d", "metric": 0.2, "spec": "{}"},
    ]
    assert ledger.add(rows) == 2
    ledger.add([{"idea": "i002", "spec_hash": "aaa", "timeframe": "4h", "metric": 0.3}])
    frame = ledger.frame()
    assert len(frame) == 3 and frame["spec"].iloc[0] == '{"a": 1}'
    assert ledger.count() == 2
    distinct = ledger.distinct()
    assert distinct.loc[distinct["timeframe"] == "4h", "metric"].item() == 0.3
