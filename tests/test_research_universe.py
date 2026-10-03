"""Idea cards that name their asset classes: registration rules, the trial ledger, the stages."""

import json

import pandas as pd
import pytest
from research_helpers import FAST_GATES, UNIVERSES, draft, make_workspace, write_session_market

from algotrade.research.cards import validate
from algotrade.research.journal import (
    TRIAL_COLUMNS,
    Journal,
    LedgerFormatError,
    TrialLedger,
    migrate_trials,
)
from algotrade.research.registry import Refused, register, revise
from algotrade.research.split import version_symbols


def card(universe=None, **kwargs):
    made = draft(**kwargs)
    if universe is not None:
        made.front["universe"] = universe
    return made


@pytest.fixture
def multi(tmp_path):
    ws, criteria = make_workspace(tmp_path, {"data.universes": UNIVERSES, **FAST_GATES})
    write_session_market(ws.raw_data)
    return ws, criteria


def test_card_universe_rules(multi) -> None:
    _, criteria = multi
    assert validate(card(["bonds", "fx"]), criteria) == []
    assert validate(card("all"), criteria) == []
    assert validate(card(), criteria) == []
    assert any("unknown asset class" in p for p in validate(card(["metals"]), criteria))
    assert any("non-empty list" in p for p in validate(card([]), criteria))
    funding = card(["bonds"], inputs=["price", "funding"])
    assert any("funding is a crypto" in p for p in validate(funding, criteria))
    assert validate(card(["crypto"], inputs=["funding"]), criteria) == []
    volume = card(["fx"], inputs=["price", "volume"])
    assert any("forex volume" in p for p in validate(volume, criteria))


def test_registration_journals_the_universe_and_its_symbols(multi) -> None:
    ws, criteria = multi
    version = register(ws, criteria, card(["fx", "bonds"]), commit=False)
    assert version.universe == ("bonds", "fx")
    assert version.symbols == ("TLT", "IEF", "EURUSD")
    assert version_symbols(criteria, version) == ["TLT", "IEF", "EURUSD"]
    entry = next(e for e in Journal(ws.journal_path).entries() if e["event"] == "idea_registered")
    assert entry["universe"] == ["bonds", "fx"]
    slow = card(
        title="Slow trend",
        spec={"type": "ma_crossover", "fast": 40, "slow": 300},
        differs_from={"i001": "Much slower averages, on crypto rather than bonds and fx"},
    )
    crypto = register(ws, criteria, slow, commit=False)  # fmt: skip
    assert crypto.universe == ("crypto",)
    assert version_symbols(criteria, crypto) == ["BTC", "ETH", "SOL"]


def test_same_rules_on_another_universe_is_a_revision(multi) -> None:
    ws, criteria = multi
    first = register(ws, criteria, card(["bonds"]), commit=False)
    with pytest.raises(Refused, match="another timeframe or universe"):
        register(ws, criteria, card(["fx"]), commit=False)
    second = revise(ws, criteria, first.idea, card(["fx"]), reason="bonds barely trend",
                    commit=False)  # fmt: skip
    assert second.universe == ("fx",) and second.version == 2


def test_old_journal_entries_mean_crypto(multi) -> None:
    ws, _ = multi
    Journal(ws.journal_path).append(
        "idea_registered", idea="i001", version=1, slug="old", title="Old", timeframe="4h",
        spec={"type": "ma_crossover"}, card_hash="x", card_path="research/ideas/old.md",
    )  # fmt: skip
    old = Journal(ws.journal_path).ideas()["i001"].latest
    assert old.universe == ("crypto",) and old.symbols is None


def ledger_row(**fields) -> dict:
    return {"idea": "i001", "version": 1, "stage": "feasibility", "kind": "grid", "label": "x",
            "spec_hash": "abc", "timeframe": "1d", "metric": 0.1, "observations": 10,
            "window": "dev", "spec": "{}", **fields}  # fmt: skip


def test_trial_ledger_counts_each_universe(tmp_path) -> None:
    ledger = TrialLedger(tmp_path / "trials.csv")
    ledger.add([ledger_row(), ledger_row(universe="bonds+fx"), ledger_row(universe="crypto")])
    frame = ledger.frame()
    assert list(frame.columns) == [*TRIAL_COLUMNS, "universe"]
    assert frame["universe"].tolist() == ["crypto", "bonds+fx", "crypto"]
    assert ledger.count() == 2


def test_old_ledgers_take_crypto_rows_and_migrate_once(tmp_path) -> None:
    path = tmp_path / "trials.csv"
    old = pd.DataFrame([{"ts": "2026-01-01T00:00:00Z", **ledger_row()}])[TRIAL_COLUMNS]
    old.to_csv(path, index=False, lineterminator="\n")
    before = path.read_text(encoding="utf-8")
    ledger = TrialLedger(path)
    assert ledger.frame()["universe"].tolist() == ["crypto"]
    ledger.add([ledger_row(spec_hash="def", universe="crypto")])
    assert path.read_text(encoding="utf-8").startswith(before)
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(TRIAL_COLUMNS)
    with pytest.raises(LedgerFormatError, match="migrate-trials"):
        ledger.add([ledger_row(universe="fx")])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert migrate_trials(path) == 2
    migrated = path.read_text(encoding="utf-8").splitlines()
    assert migrated[0] == ",".join([*TRIAL_COLUMNS, "universe"])
    assert all(new == f"{line},crypto" for line, new in zip(lines[1:], migrated[1:], strict=True))
    ledger.add([ledger_row(universe="fx")])
    assert ledger.count() == 3
    with pytest.raises(LedgerFormatError, match="already"):
        migrate_trials(path)
    assert migrate_trials(tmp_path / "none.csv") == 0


def test_multi_asset_feasibility(multi) -> None:
    from algotrade.research.buildcheck import build_check
    from algotrade.research.feasibility import feasibility
    from algotrade.research.registry import find_version

    ws, criteria = multi
    made = card(["bonds", "fx"], timeframe="1d", spec={"type": "ma_crossover", "fast": 10,
                                                        "slow": 40},
                optimise={"fast": [5, 10], "slow": [40, 60]})  # fmt: skip
    version = register(ws, criteria, made)
    assert build_check(ws, criteria, version).result.verdict == "PASS"
    result = feasibility(ws, criteria, find_version(ws, version.idea), report=False)
    assert [w["symbol"] for w in result.provenance["data"]] == ["TLT", "IEF", "EURUSD"]
    folder = ws.stage_dir(version.idea, version.slug, 1, "feasibility")
    by_class = pd.read_csv(folder / "core_by_class.csv", index_col=0)
    assert list(by_class.index) == ["bonds", "fx"]
    assert any(note.startswith("By asset class") for note in result.notes)
    trials = TrialLedger(ws.trials_path).frame()
    assert set(trials["universe"]) == {"bonds+fx"}
    saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert saved["universe"] == ["bonds", "fx"]


def test_mixed_universe_through_every_gate_to_the_freeze(tmp_path) -> None:
    """Crypto, bond ETFs and forex (three calendars) from registration to the freeze."""

    from algotrade.research.buildcheck import build_check
    from algotrade.research.feasibility import feasibility
    from algotrade.research.freeze import freeze
    from algotrade.research.holdout import holdout
    from algotrade.research.incubation import start
    from algotrade.research.index import write_index
    from algotrade.research.registry import find_version
    from algotrade.research.validate import validate as validate_stage

    overrides = {"data.universes": UNIVERSES, **FAST_GATES, "feasibility.min_trades_per_symbol": 3}
    ws, criteria = make_workspace(tmp_path, overrides, edge=0.002)
    write_session_market(ws.raw_data, edge=0.002)
    made = card("all", spec={"type": "ewmac"}, optimise={"fast": [8, 16], "slow": [64, 128]},
                timeframe="1d")  # fmt: skip
    build_check(ws, criteria, register(ws, criteria, made))
    results = {}
    for stage in (feasibility, validate_stage, holdout):
        result = stage(ws, criteria, find_version(ws, "i001"), report=False)
        assert result.verdict == "PASS", (result.stage, result.reasons)
        results[result.stage] = result
    assert results["validation"].extra["deflated_sharpe"]["per_symbol"] is True
    folder = ws.stage_dir("i001", "fast-trend", 1, "validation")
    assert list(pd.read_csv(folder / "oos_by_class.csv", index_col=0).index) == [
        "crypto", "bonds", "fx"
    ]  # fmt: skip
    bands = [c.name for c in results["holdout"].checks if "bootstrap band" in c.name]
    assert [name.split("(")[-1] for name in bands] == ["24/7)", "us_equity)", "fx)"]

    frozen = freeze(ws, criteria, find_version(ws, "i001"))
    assert frozen["universe"] == ["crypto", "bonds", "fx"] and frozen["exchange"] is None
    assert frozen["symbols"] == ["BTC", "ETH", "SOL", "TLT", "IEF", "EURUSD"]
    decision = ws.version_dir("i001", "fast-trend", 1) / "decision.md"
    assert "Paper trading runs on the Binance and Bybit testnets only" in decision.read_text(
        encoding="utf-8"
    )
    with pytest.raises(Refused, match="stops at the freeze"):
        start(ws, find_version(ws, "i001"), commit=False)
    assert "| 1d (crypto+bonds+fx) |" in write_index(ws).read_text(encoding="utf-8")


def test_migrate_trials_from_the_command_line(tmp_path, capsys) -> None:
    from scripts import research as cli

    ws, _ = make_workspace(tmp_path)
    old = pd.DataFrame([{"ts": "2026-01-01T00:00:00Z", **ledger_row()}])[TRIAL_COLUMNS]
    old.to_csv(ws.trials_path, index=False, lineterminator="\n")
    assert cli.main(["--root", str(ws.root), "migrate-trials"]) == 0
    assert "to 1 trials" in capsys.readouterr().out
    assert TrialLedger(ws.trials_path).frame()["universe"].tolist() == ["crypto"]
    events = [e["event"] for e in Journal(ws.journal_path).entries()]
    assert events[-1] == "trials_migrated"
    assert cli.main(["--root", str(ws.root), "--no-commit", "migrate-trials"]) == 1
    assert "already has a universe column" in capsys.readouterr().out


def test_paper_trading_refuses_other_asset_classes(tmp_path, monkeypatch, capsys) -> None:
    from scripts import paper_trade

    version = type("V", (), {"idea": "i009", "version": 1})()
    monkeypatch.setattr(paper_trade, "find_version", lambda ws, idea, number: version)
    monkeypatch.setattr(paper_trade, "frozen_strategy",
                        lambda ws, found: {"universe": ["bonds", "fx"]})  # fmt: skip
    assert paper_trade.main(["i009", "--root", str(tmp_path), "--dry-run"]) == 1
    assert "trades bonds, fx; the testnets are crypto" in capsys.readouterr().out
