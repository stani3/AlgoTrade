import math

import pytest
from research_helpers import REPO, draft

from algotrade.research.cards import (
    CardError,
    Registration,
    card_hash,
    parse_card,
    read_card,
    require_valid,
    slugify,
    stamp,
    validate,
)
from algotrade.research.criteria import (
    Check,
    above,
    all_passed,
    at_least,
    at_most,
    below,
    load_criteria,
)

LF, CRLF = chr(10), chr(13) + chr(10)
CRITERIA = load_criteria(REPO / "research" / "criteria.yaml")


# --- criteria -------------------------------------------------------------------------------


def test_criteria_load_with_hash_and_dotted_access() -> None:
    assert CRITERIA.get("data.dev_end") == "2025-05-01"
    assert CRITERIA.get("budgets.revisions_per_idea") == 1
    assert len(CRITERIA.hash) == 16
    with pytest.raises(KeyError, match="no 'data.nope'"):
        CRITERIA.get("data.nope")
    with pytest.raises(KeyError):
        CRITERIA.get("data.dev_end.deeper")


def test_hash_follows_the_exact_bytes(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\n")
    first = load_criteria(path).hash
    path.write_text("a: 1  # same value, different bytes\n")
    assert load_criteria(path).hash != first
    path.write_text("")
    assert load_criteria(path).values == {}


@pytest.mark.parametrize(
    ("check", "value", "limit", "passed", "text"),
    [
        (at_least, 0.7, 0.7, True, ">= 0.7"),
        (at_least, 0.69, 0.7, False, ">= 0.7"),
        (at_most, 0.0, 0, True, "<= 0"),
        (at_most, 1.0, 0, False, "<= 0"),
        (above, 2.0, 2.0, False, "> 2"),
        (above, 2.01, 2.0, True, "> 2"),
        (below, 0.1, 0.1, False, "< 0.1"),
        (below, 0.05, 0.1, True, "< 0.1"),
        (at_least, math.nan, 0.0, False, ">= 0"),
    ],
)
def test_checks_compare_and_describe_the_limit(check, value, limit, passed, text) -> None:
    result = check("thing", value, limit)
    assert result.passed is passed
    assert result.limit == text
    assert result.line().startswith("PASS" if passed else "FAIL")
    assert result.to_dict()["passed"] is passed


def test_check_line_formats_and_all_passed() -> None:
    assert at_least("cov", 0.5, 1.0, "{:.0%}").limit == ">= 100%"
    assert "n/a" not in Check("x", 0.123456, ">= 0", True).line()
    assert Check("label", "text", "", True).line() == "PASS  label: text (needs )"
    assert all_passed([Check("a", 1, "", True)])
    assert not all_passed([Check("a", 1, "", True), Check("b", 0, "", False)])


# --- cards ----------------------------------------------------------------------------------


def test_valid_card_round_trips_through_render() -> None:
    card = draft(optimise={"fast": [10, 20], "slow": [50, 100, 200]})
    assert validate(card, CRITERIA) == []
    again = parse_card(card.render())
    assert again.front == card.front
    assert again.combinations()[0] == {"fast": 10, "slow": 50}
    assert len(again.combinations()) == 6
    assert card.slug == "fast-trend"
    assert card.version == 1 and card.id is None and card.parent is None


def test_card_without_grid_has_one_empty_combination() -> None:
    assert draft().combinations() == [{}]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"title": ""}, "missing 'title'"),
        ({"taxonomy": {"family": "astrology", "inputs": ["price"], "horizon": "days"}},
         "taxonomy.family"),
        ({"taxonomy": {"family": "trend", "inputs": ["tweets"], "horizon": "days"}},
         "taxonomy.inputs"),
        ({"taxonomy": {"family": "trend", "inputs": [], "horizon": "days"}}, "taxonomy.inputs"),
        ({"taxonomy": {"family": "trend", "inputs": ["price"], "horizon": "years"}},
         "taxonomy.horizon"),
        ({"timeframe": "15m"}, "timeframe must be one of"),
        ({"spec": {"fast": 2}}, "spec needs a 'type'"),
        ({"optimise": [1, 2]}, "optimise must map"),
        ({"optimise": {"fast": [1]}}, "at least 2 values"),
        ({"optimise": {"a": [1, 2], "b": [1, 2], "c": [1, 2], "d": [1, 2]}}, "at most 3"),
        ({"optimise": {"a": list(range(11)), "b": list(range(10))}}, "110 combinations"),
        ({"expected_trades_per_year": "lots"}, "must be a number"),
    ],
)  # fmt: skip
def test_invalid_cards_are_rejected(change, problem) -> None:
    card = draft()
    card.front.update(change)
    problems = validate(card, CRITERIA)
    assert any(problem in p for p in problems), problems
    with pytest.raises(CardError, match="card rejected"):
        require_valid(card, CRITERIA)


def test_body_needs_every_section() -> None:
    card = draft()
    card.body = card.body.replace("## Falsified if", "## Maybe")
    assert validate(card, CRITERIA) == ["body needs a '## Falsified if' section"]


def test_parse_errors() -> None:
    with pytest.raises(CardError, match="front matter"):
        parse_card("no front matter")
    with pytest.raises(CardError, match="mapping"):
        parse_card("---\n- a list\n---\nbody")
    assert parse_card("---\n\n---\n").front == {}


def test_read_card_and_hash(tmp_path) -> None:
    path = tmp_path / "idea.md"
    path.write_text(draft().render(), encoding="utf-8", newline=LF)
    assert read_card(path).path == path
    before = card_hash(path)
    path.write_text(draft().render(), encoding="utf-8", newline=CRLF)  # git checkout on Windows
    assert card_hash(path) == before
    path.write_text(draft().render() + " ", encoding="utf-8")
    assert card_hash(path) != before


def test_criteria_hash_ignores_line_endings(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("a: 1" + LF + "b: 2" + LF, encoding="utf-8", newline=LF)
    unix = load_criteria(path).hash
    path.write_text("a: 1" + LF + "b: 2" + LF, encoding="utf-8", newline=CRLF)
    assert load_criteria(path).hash == unix


def test_slugify() -> None:
    assert slugify("Funding extremes: fade the crowd!") == "funding-extremes-fade-the-crowd"
    assert slugify("***") == "idea"
    assert len(slugify("x" * 80)) == 40


def test_stamp_puts_registration_first_and_drops_stale_stamps() -> None:
    card = draft()
    card.front.update(id="i999", version=7, parent=3, revision_reason="old")
    reg = Registration("i004", 1, "2026-10-02T00:00:00Z", {"type": "i004_new_rule", "x": 1})
    stamped = stamp(card, reg)
    assert list(stamped.front)[:3] == ["id", "version", "registered"]
    assert stamped.id == "i004" and stamped.version == 1
    assert "parent" not in stamped.front
    assert stamped.spec == {"type": "i004_new_rule", "x": 1}

    revised = stamp(card, Registration("i004", 2, "t", {"type": "x"}, parent=1, reason="why"))
    assert revised.parent == 1 and revised.front["revision_reason"] == "why"
    extra = stamp(card, Registration("i004", 1, "t", {"type": "x"}, extra={"historical": True}))
    assert extra.front["historical"] is True
    with pytest.raises(CardError, match="bad idea id"):
        stamp(card, Registration("x1", 1, "t", {"type": "x"}))
