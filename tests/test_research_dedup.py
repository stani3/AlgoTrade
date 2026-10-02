import pytest

from algotrade.research.dedup import (
    canonical,
    config_hashes,
    core_rules,
    flatten,
    get_path,
    grid_specs,
    known_type,
    region,
    set_path,
    spec_hash,
    structure,
)

EWMAC = {"type": "ewmac", "fast": 16, "slow": 64}


def test_hash_ignores_key_order_defaults_and_number_spelling() -> None:
    explicit = {"slow": 64.0, "type": "ewmac", "fast": 16}
    assert spec_hash(EWMAC) == spec_hash(explicit)
    assert spec_hash({"type": "ewmac"}) == spec_hash(EWMAC)  # 16/64 are the defaults
    assert spec_hash({"type": "breakout_bracket", "lookback": 48}) == spec_hash(
        {"type": "breakout_bracket", "lookback": 48.0, "stop_atr": 2}
    )


def test_any_real_parameter_change_changes_the_hash() -> None:
    assert spec_hash(EWMAC) != spec_hash({**EWMAC, "fast": 8})
    assert spec_hash(EWMAC) != spec_hash({**EWMAC, "allow_short": False})
    wrapped = {"type": "vol_target", "strategy": EWMAC}
    assert spec_hash(wrapped) != spec_hash(EWMAC)
    assert spec_hash(wrapped) != spec_hash({**wrapped, "annual_vol": 0.3})


def test_combine_children_order_and_weight_scale_do_not_matter() -> None:
    a, b = {"type": "ewmac"}, {"type": "carver_breakout", "lookback": 80}
    one = {"type": "combine", "strategies": [a, b], "weights": [2, 1]}
    two = {"type": "combine", "strategies": [b, a], "weights": [1, 2]}
    three = {"type": "combine", "strategies": [b, a], "weights": [10, 20]}
    assert spec_hash(one) == spec_hash(two) == spec_hash(three)
    equal = {"type": "combine", "strategies": [a, b]}
    assert spec_hash(equal) == spec_hash({**equal, "weights": [3, 3]})
    assert spec_hash(one) != spec_hash(equal)


def test_unknown_types_are_normalised_without_defaults() -> None:
    new = {"type": "i007_funding_fade", "z": 2}
    assert not known_type(new)
    assert not known_type({"type": "vol_target", "strategy": new})
    assert not known_type({"type": "combine", "strategies": [new]})
    assert known_type({"type": "combine", "strategies": [EWMAC]})
    assert known_type([1, 2])
    assert canonical(new) == {"type": "i007_funding_fade", "z": 2.0}
    assert spec_hash(new) == spec_hash({"z": 2.0, "type": "i007_funding_fade"})


def test_canonical_rejects_unserialisable_values() -> None:
    with pytest.raises(TypeError, match="cannot canonicalise"):
        canonical({"type": "i007_x", "bad": object()})


def test_paths_and_grids() -> None:
    nested = {"type": "vol_target", "strategy": {"type": "ewmac", "fast": 16}}
    assert get_path(nested, "strategy.fast") == 16
    changed = set_path(nested, "strategy.fast", 8)
    assert changed["strategy"]["fast"] == 8 and nested["strategy"]["fast"] == 16
    combined = {"type": "combine", "strategies": [{"type": "ewmac", "fast": 16}]}
    assert get_path(set_path(combined, "strategies.0.fast", 4), "strategies.0.fast") == 4
    assert set_path({"weights": [1, 2]}, "weights.1", 5) == {"weights": [1, 5]}
    assert set_path({}, "a.b", 1) == {"a": {"b": 1}}
    specs = grid_specs(EWMAC, {"fast": [8, 16], "slow": [32, 64, 128]})
    assert len(specs) == 6 and specs[-1] == {"type": "ewmac", "fast": 16, "slow": 128}
    assert grid_specs(EWMAC, {}) == [EWMAC]
    assert len(config_hashes(EWMAC, {"fast": [8, 16]})) == 2  # 16 is the base itself


def test_flatten_and_structure() -> None:
    spec = {
        "type": "combine",
        "strategies": [{"type": "ewmac", "fast": 16}, {"type": "carver_breakout"}],
        "weights": [1, 2],
    }
    flat = flatten(spec)
    assert flat["strategies.0.fast"] == 16 and flat["weights.1"] == 2
    assert "type" not in flat
    assert structure(EWMAC) == structure({**EWMAC, "fast": 4})
    assert structure(EWMAC) != structure({"type": "vol_target", "strategy": EWMAC})


def test_regions_overlap_on_the_same_rules_and_timeframe() -> None:
    book = {"type": "breakout_bracket"}
    grid = {"stop_atr": [1, 1.5, 2, 3, 4], "target_atr": [2, 3, 4, 6, 8]}
    tested = region(book, grid, "4h", 0.2)
    near = region({**book, "lookback": 50}, {}, "4h", 0.2)  # 50 is within 48 +- 20%
    far = region({**book, "lookback": 100}, {}, "4h", 0.2)
    long_only = region({**book, "allow_short": False}, {}, "4h", 0.2)
    wide_stop = region({**book, "stop_atr": 6}, {}, "4h", 0.2)
    assert tested.overlaps(near) and near.overlaps(tested)
    assert not tested.overlaps(far)
    assert not tested.overlaps(long_only)
    assert not tested.overlaps(wide_stop)
    assert region(EWMAC, {}, "4h", 0.2).overlaps(region(EWMAC, {}, "1d", 0.2))  # tf is separate
    assert not tested.overlaps(region(EWMAC, {}, "4h", 0.2))


def test_zero_valued_parameters_have_no_tolerance_band() -> None:
    zero = region({"type": "i007_rule", "threshold": 0.0}, {}, "4h", 0.2)
    assert zero.numeric["threshold"] == (0.0, 0.0)


def test_core_rules_ignore_sizing_and_blending() -> None:
    assert core_rules({"type": "vol_target", "strategy": EWMAC}) == ("ewmac",)
    blend = {
        "type": "combine",
        "strategies": [{"type": "carver_breakout"}, {"type": "vol_target", "strategy": EWMAC}],
    }
    assert core_rules(blend) == ("carver_breakout", "ewmac")
    filtered = {"type": "trend_filter", "strategy": {"type": "rsi_reversion"}}
    assert core_rules(filtered) == ("trend_filter:rsi_reversion",)
