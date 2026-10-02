"""Duplicate detection on strategy specs: canonical hashes and parameter-region overlap.

Behavioural duplicates (different code, same trades) are handled in ``fingerprint.py``.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from numbers import Number

from algotrade.strategies import STRATEGIES, from_spec, to_spec


def _number(value: float) -> float:
    return float(f"{float(value):.12g}")


def _structure_key(spec: dict) -> str:
    """Types only, so sorting combine children is stable while their parameters change."""

    def types(node: object) -> object:
        if isinstance(node, dict):
            return {
                key: value if key == "type" else types(value)
                for key, value in sorted(node.items())
                if key == "type" or _nested(value)
            }
        return [types(item) for item in node if _nested(item)]  # only dicts and lists get here

    return json.dumps(types(spec), sort_keys=True)


def _nested(value: object) -> bool:
    return isinstance(value, dict) or (
        isinstance(value, list) and any(isinstance(item, dict) for item in value)
    )


def _normalise(node: object) -> object:
    if isinstance(node, bool) or node is None or isinstance(node, str):
        return node
    if isinstance(node, Number):
        return _number(node)
    if isinstance(node, list | tuple):
        return [_normalise(item) for item in node]
    if isinstance(node, dict):
        out = {key: _normalise(value) for key, value in sorted(node.items())}
        if out.get("type") == "combine" and out.get("strategies"):
            children = out["strategies"]
            weights = out.get("weights") or [1.0] * len(children)
            total = sum(weights)
            pairs = sorted(
                zip(children, [_number(w / total) for w in weights], strict=True),
                key=lambda pair: (_structure_key(pair[0]), json.dumps(pair[0], sort_keys=True)),
            )
            out["strategies"] = [child for child, _ in pairs]
            out["weights"] = [weight for _, weight in pairs]
        return out
    raise TypeError(f"cannot canonicalise {type(node).__name__}")


def known_type(spec: object) -> bool:
    """True when every ``type`` in the spec is a registered strategy."""

    if isinstance(spec, dict):
        if "type" in spec and spec["type"] not in STRATEGIES:
            return False
        children = list(spec.values())
    elif isinstance(spec, list):
        children = spec
    else:
        return True
    return all(known_type(child) for child in children)


def canonical(spec: dict) -> dict:
    """Defaults filled in, keys sorted, numbers as floats, combine children in a fixed order.

    Specs naming a strategy type that does not exist yet (an idea not built yet) are only
    normalised, without default filling.
    """

    full = to_spec(from_spec(spec)) if known_type(spec) else copy.deepcopy(spec)
    return _normalise(full)


def spec_hash(spec: dict) -> str:
    text = json.dumps(canonical(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def get_path(spec: dict, path: str) -> object:
    node: object = spec
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def set_path(spec: dict, path: str, value: object) -> dict:
    """Copy of ``spec`` with ``a.b.0.c``-style ``path`` set to ``value``."""

    out = copy.deepcopy(spec)
    parts = path.split(".")
    node: object = out
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return out


def grid_specs(base: dict, grid: dict[str, list]) -> list[dict]:
    """The base spec with every combination of the grid applied (the base alone if no grid)."""

    specs = []
    for combo in itertools.product(*grid.values()):
        spec = base
        for path, value in zip(grid, combo, strict=True):
            spec = set_path(spec, path, value)
        specs.append(spec)
    return specs


def flatten(spec: dict, prefix: str = "") -> dict[str, object]:
    """Leaf parameters by dotted path; ``type`` entries are part of the structure instead."""

    out: dict[str, object] = {}
    for key, value in spec.items():
        path = f"{prefix}{key}"
        if key == "type":
            continue
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
            for i, item in enumerate(value):
                out.update(flatten(item, f"{path}.{i}."))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                out[f"{path}.{i}"] = item
        else:
            out[path] = value
    return out


def structure(spec: dict) -> str:
    return _structure_key(canonical(spec))


@dataclass(frozen=True)
class Region:
    """The parameter box a registered idea version covered on one timeframe."""

    structure: str
    timeframe: str
    numeric: dict[str, tuple[float, float]]
    other: dict[str, frozenset]

    def overlaps(self, other: Region) -> bool:
        """Same rules, and every parameter's range or value set intersects."""

        if self.structure != other.structure:
            return False
        for path, (low, high) in self.numeric.items():
            if path in other.numeric:
                other_low, other_high = other.numeric[path]
                if high < other_low or other_high < low:
                    return False
        for path, values in self.other.items():
            if path in other.other and not values & other.other[path]:
                return False
        return True


def region(base: dict, grid: dict[str, list], timeframe: str, tolerance: float) -> Region:
    numeric: dict[str, list[float]] = {}
    other: dict[str, set] = {}
    for spec in grid_specs(base, grid):
        for path, value in flatten(canonical(spec)).items():
            if isinstance(value, bool) or not isinstance(value, Number):
                other.setdefault(path, set()).add(json.dumps(value))
            else:
                numeric.setdefault(path, []).append(float(value))
    boxes = {}
    for path, values in numeric.items():
        low, high = min(values), max(values)
        boxes[path] = (
            low - abs(low) * tolerance if low == high else low,
            high + abs(high) * tolerance if low == high else high,
        )
    return Region(
        structure=structure(base),
        timeframe=timeframe,
        numeric=boxes,
        other={path: frozenset(values) for path, values in other.items()},
    )


def core_rules(spec: dict) -> tuple[str, ...]:
    """The trading rules inside a spec, ignoring how they are sized or blended.

    ``vol_target(ewmac)`` and ``combine([ewmac])`` are both just ``ewmac``; a trend filter
    changes which trades are taken, so it stays part of the rule.
    """

    kind = spec.get("type")
    if kind == "vol_target":
        return core_rules(spec["strategy"])
    if kind == "combine":
        return tuple(sorted(rule for child in spec["strategies"] for rule in core_rules(child)))
    if kind == "trend_filter":
        return (f"trend_filter:{'+'.join(core_rules(spec['strategy']))}",)
    return (str(kind),)


def config_hashes(base: dict, grid: dict[str, list]) -> set[str]:
    return {spec_hash(spec) for spec in [base, *grid_specs(base, grid)]}
