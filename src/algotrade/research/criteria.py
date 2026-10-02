"""Load the pre-set thresholds and turn measurements into pass/fail checks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REQUIRED = object()


@dataclass(frozen=True)
class Criteria:
    """The parsed ``criteria.yaml`` plus the hash of its exact bytes."""

    values: dict
    hash: str

    def get(self, dotted: str, default: Any = _REQUIRED) -> Any:
        """The value at ``dotted``; a missing key is an error unless a ``default`` is given."""

        node: Any = self.values
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                if default is not _REQUIRED:
                    return default
                raise KeyError(f"criteria.yaml has no '{dotted}'")
            node = node[key]
        return node


def load_criteria(path: Path) -> Criteria:
    raw = path.read_bytes().replace(b"\r\n", b"\n")  # same hash whatever git's line endings
    return Criteria(values=yaml.safe_load(raw) or {}, hash=hashlib.sha256(raw).hexdigest()[:16])


@dataclass(frozen=True)
class Check:
    """One gate: the measured value, the limit as text, and whether it passed."""

    name: str
    value: float
    limit: str
    passed: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "limit": self.limit, "passed": self.passed}

    def line(self, fmt: str = "{:.3g}") -> str:
        shown = self.value if isinstance(self.value, str) else fmt.format(self.value)
        return f"{'PASS' if self.passed else 'FAIL'}  {self.name}: {shown} (needs {self.limit})"


def _make(symbol: str, test: Callable[[float, float], bool]):
    def check(name: str, value: float, limit: float, fmt: str = "{:g}") -> Check:
        passed = bool(not math.isnan(value) and test(value, limit))
        return Check(name, float(value), f"{symbol} {fmt.format(limit)}", passed)

    return check


at_least = _make(">=", lambda value, limit: value >= limit)
at_most = _make("<=", lambda value, limit: value <= limit)
above = _make(">", lambda value, limit: value > limit)
below = _make("<", lambda value, limit: value < limit)


def all_passed(checks: list[Check]) -> bool:
    return all(check.passed for check in checks)
