import pytest
from research_helpers import commits, draft, make_workspace

from algotrade.research.cards import CardError, card_hash, read_card
from algotrade.research.journal import Journal
from algotrade.research.registry import (
    Conflict,
    Refused,
    abandon,
    find_version,
    register,
    rename_new_types,
    revise,
    strategy_types,
)

WHY = "Uses a breakout channel instead of averages, so it enters later but on cleaner moves."


@pytest.fixture
def setup(tmp_path):
    return make_workspace(tmp_path, market=False)


def fail(ws, idea: str, version: int = 1, stage: str = "feasibility") -> None:
    Journal(ws.journal_path).append(
        "stage_result", idea=idea, version=version, stage=stage, verdict="FAIL",
        failures=["median Sharpe 0.1 (needs >= 0.3)"],
    )  # fmt: skip


def test_register_stores_journals_indexes_and_commits(setup) -> None:
    ws, criteria = setup
    version = register(ws, criteria, draft(optimise={"fast": [10, 20], "slow": [100, 200]}))
    assert version.idea == "i001" and version.version == 1
    path = ws.root / version.card_path
    assert path == ws.version_dir("i001", "fast-trend", 1) / "idea.md"
    card = read_card(path)
    assert card.id == "i001" and card.front["registered"]
    assert card.spec == {"type": "ma_crossover", "fast": 20, "slow": 100}
    assert version.card_hash == card_hash(path)
    assert version.grid == {"fast": [10, 20], "slow": [100, 200]}
    assert "| i001 | Fast trend |" in ws.index_path.read_text(encoding="utf-8")
    assert commits(ws.root)[0] == "research(i001-fast-trend v1): registered"


def test_invalid_card_is_refused_before_anything_is_written(setup) -> None:
    ws, criteria = setup
    card = draft()
    card.front["timeframe"] = "1m"
    with pytest.raises(CardError):
        register(ws, criteria, card)
    assert not ws.journal_path.exists()


def test_new_strategy_types_get_the_idea_prefix(setup) -> None:
    ws, criteria = setup
    spec = {"type": "vol_target", "strategy": {"type": "funding_fade", "z": 2}}
    version = register(ws, criteria, draft(spec=spec), commit=False)
    assert version.spec == {"type": "vol_target", "strategy": {"type": "i001_funding_fade", "z": 2}}
    assert strategy_types(version.spec) == ["vol_target", "i001_funding_fade"]
    assert rename_new_types([{"type": "x"}, 3], "i002") == [{"type": "i002_x"}, 3]


def test_another_ideas_strategy_type_is_refused(setup) -> None:
    ws, criteria = setup
    with pytest.raises(Refused, match="belongs to idea i004"):
        register(ws, criteria, draft(spec={"type": "i004_funding_fade"}), commit=False)


@pytest.mark.parametrize(
    ("spec", "optimise", "timeframe", "message"),
    [
        ({"type": "ma_crossover", "fast": 20, "slow": 100}, {}, "4h", "repeats a configuration"),
        ({"type": "ma_crossover", "fast": 20, "slow": 200}, {}, "4h", "repeats a configuration"),
        ({"type": "ma_crossover", "fast": 22, "slow": 150}, {}, "4h", "overlaps the parameter"),
        ({"type": "ma_crossover", "fast": 20, "slow": 100}, {}, "1d", "same rules on another"),
    ],
)
def test_duplicates_of_an_active_idea_are_refused(setup, spec, optimise, timeframe, message):
    ws, criteria = setup
    register(ws, criteria, draft(optimise={"slow": [100, 200]}), commit=False)
    card = draft(title="Again", spec=spec, optimise=optimise, timeframe=timeframe,
                 differs_from={"i001": WHY})  # fmt: skip
    with pytest.raises(Refused, match=message) as error:
        register(ws, criteria, card, commit=False)
    if timeframe == "1d":
        assert "research revise" in str(error.value)


def test_neighbours_must_be_answered(setup) -> None:
    ws, criteria = setup
    register(ws, criteria, draft(), commit=False)
    other = draft(title="Channel trend", spec={"type": "donchian_breakout"})
    with pytest.raises(Refused, match="differs_from: i001"):
        register(ws, criteria, other, commit=False)
    other.front["differs_from"] = {"i001": "too short"}
    with pytest.raises(Refused, match="differs_from: i001"):
        register(ws, criteria, other, commit=False)
    other.front["differs_from"] = {"i001": WHY}
    assert register(ws, criteria, other, commit=False).idea == "i002"
    # A different horizon, family or input set is not a neighbour.
    weekly = draft(title="Weekly", spec={"type": "momentum"}, horizon="weeks")
    assert register(ws, criteria, weekly, commit=False).idea == "i003"
    carry = draft(title="Carry", spec={"type": "funding_carry"}, family="carry",
                  inputs=["funding"])  # fmt: skip
    assert register(ws, criteria, carry, commit=False).idea == "i004"


def test_failed_ideas_stay_failed_unless_the_user_retests(setup) -> None:
    ws, criteria = setup
    register(ws, criteria, draft(), commit=False)
    fail(ws, "i001")
    again = draft(title="Fast trend again", differs_from={"i001": WHY})
    with pytest.raises(Refused, match="Failed ideas stay failed"):
        register(ws, criteria, again, commit=False)
    version = register(ws, criteria, again, retest="new year of data, user asked", commit=False)
    assert version.idea == "i002" and version.retest == "new year of data, user asked"
    # The same rules on another timeframe of a failed idea also need the override.
    daily = draft(title="Daily trend", timeframe="1d", differs_from={"i001": WHY, "i002": WHY})
    with pytest.raises(Refused, match="Failed ideas stay failed"):
        register(ws, criteria, daily, commit=False)


def test_revise_within_budget_with_a_reason(setup) -> None:
    ws, criteria = setup
    register(ws, criteria, draft(), commit=False)
    fail(ws, "i001")
    long_only = draft(spec={"type": "ma_crossover", "fast": 20, "slow": 100, "allow_short": False})
    with pytest.raises(Refused, match="needs a reason"):
        revise(ws, criteria, "i001", long_only, "  ", commit=False)
    v2 = revise(ws, criteria, "i001", long_only, "shorts lost money in every regime")
    assert v2.version == 2 and v2.parent == 1 and v2.reason == "shorts lost money in every regime"
    card = read_card(ws.root / v2.card_path)
    assert card.parent == 1 and card.front["revision_reason"] == v2.reason
    assert commits(ws.root)[0].startswith("research(i001-fast-trend v2): revised - shorts")
    assert Journal(ws.journal_path).ideas()["i001"].status == "registered"
    with pytest.raises(Refused, match="revision budget"):
        revise(ws, criteria, "i001", draft(spec={"type": "ma_crossover", "fast": 5}), "again")


def test_revision_must_differ_from_every_version(setup) -> None:
    ws, criteria = setup
    register(ws, criteria, draft(), commit=False)
    with pytest.raises(Refused, match="repeats a configuration already registered by i001"):
        revise(ws, criteria, "i001", draft(), "same thing", commit=False)
    # Overlap with its own parent is allowed (that is what a revision is) ...
    nearby = draft(spec={"type": "ma_crossover", "fast": 21, "slow": 100})
    assert revise(ws, criteria, "i001", nearby, "slightly faster", commit=False).version == 2


def test_revision_cannot_copy_another_idea(setup) -> None:
    ws, criteria = setup
    register(ws, criteria, draft(), commit=False)
    register(ws, criteria, draft(title="Slow", spec={"type": "momentum"}, horizon="weeks"),
             commit=False)  # fmt: skip
    copy = draft(spec={"type": "momentum"}, horizon="weeks")
    with pytest.raises(Refused, match="duplicate"):
        revise(ws, criteria, "i001", copy, "copy i002", commit=False)
    neighbour = draft(spec={"type": "momentum", "lookback": 400}, horizon="weeks")
    with pytest.raises(Refused, match="differs from: i002"):
        revise(ws, criteria, "i001", neighbour, "go slower", commit=False)
    with pytest.raises(Refused, match="belongs to idea i002"):
        revise(ws, criteria, "i001", draft(spec={"type": "i002_x"}), "steal", commit=False)
    ok = draft(spec={"type": "i001_rewrite"})
    assert revise(ws, criteria, "i001", ok, "own new type", commit=False).spec == {
        "type": "i001_rewrite"
    }


def test_no_revisions_after_validation(setup) -> None:
    ws, criteria = setup
    register(ws, criteria, draft(), commit=False)
    fail(ws, "i001", stage="validation")
    with pytest.raises(Refused, match="no revisions after validation"):
        revise(ws, criteria, "i001", draft(spec={"type": "ma_crossover", "fast": 5}), "x")


def test_revise_and_abandon_unknown_ideas(setup) -> None:
    ws, criteria = setup
    with pytest.raises(Refused, match="no idea i009"):
        revise(ws, criteria, "i009", draft(), "x", commit=False)
    with pytest.raises(Refused, match="no idea i009"):
        abandon(ws, "i009", "x", commit=False)
    with pytest.raises(Refused, match="no idea i009"):
        find_version(ws, "i009")


def test_abandon_closes_the_idea(setup) -> None:
    ws, criteria = setup
    register(ws, criteria, draft(), commit=False)
    state = abandon(ws, "i001", "needs open-interest data")
    assert state.status == "abandoned" and state.failed
    assert (
        commits(ws.root)[0] == "research(i001-fast-trend v1): abandoned - needs open-interest data"
    )
    assert find_version(ws, "i001").version == 1
    with pytest.raises(Refused, match="has no v3"):
        find_version(ws, "i001", 3)
    slow = draft(title="Slow", spec={"type": "momentum"}, horizon="weeks")
    register(ws, criteria, slow, commit=False)
    before = commits(ws.root)
    assert abandon(ws, "i002", "quiet", commit=False).status == "abandoned"
    assert commits(ws.root) == before


def test_conflict_descriptions() -> None:
    assert "failed" in Conflict("exact", "i001", 1, True, "T").describe()
    assert "active" in Conflict("overlap", "i001", 1, False, "T").describe()
