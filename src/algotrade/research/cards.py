"""Idea cards: the pre-registration of a strategy before any of it is tested.

A card is Markdown with YAML front matter::

    ---
    title: Funding extremes fade
    source: "Carver, Systematic Trading (carry)"
    source_id: S21
    taxonomy: {family: mean_reversion, inputs: [price, funding], horizon: days}
    timeframe: 4h
    spec: {type: funding_fade, lookback: 90, z_entry: 2.0}
    optimise: {z_entry: [1.5, 2.0, 2.5], lookback: [45, 90, 180]}
    expected_trades_per_year: 12
    differs_from: {i003: "..."}
    ---
    ## Hypothesis
    ...

``research new`` adds ``id``, ``version`` and ``registered``, prefixes a new strategy type with the
idea id, and stores the card; its hash is journaled, so later edits are detected.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .criteria import Criteria

FAMILIES = ("trend", "breakout", "mean_reversion", "carry", "volatility", "seasonality", "other")
INPUTS = ("price", "volume", "funding", "calendar")
HORIZONS = ("hours", "days", "weeks", "months")
SECTIONS = ("Hypothesis", "Why it should work", "Rules", "Falsified if")
IDEA_ID = re.compile(r"^i\d{3}$")
STAMP_KEYS = frozenset({"id", "version", "registered", "parent", "revision_reason"})


class CardError(ValueError):
    """The card is malformed or breaks a research rule."""


@dataclass
class Card:
    front: dict
    body: str
    path: Path | None = None

    @property
    def id(self) -> str | None:
        return self.front.get("id")

    @property
    def version(self) -> int:
        return int(self.front.get("version") or 1)

    @property
    def title(self) -> str:
        return str(self.front.get("title", ""))

    @property
    def slug(self) -> str:
        return slugify(self.title)

    @property
    def timeframe(self) -> str:
        return str(self.front.get("timeframe", ""))

    @property
    def spec(self) -> dict:
        return dict(self.front.get("spec") or {})

    @property
    def optimise(self) -> dict[str, list]:
        return {key: list(values) for key, values in (self.front.get("optimise") or {}).items()}

    @property
    def taxonomy(self) -> dict:
        return dict(self.front.get("taxonomy") or {})

    @property
    def differs_from(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in (self.front.get("differs_from") or {}).items()}

    @property
    def parent(self) -> int | None:
        parent = self.front.get("parent")
        return None if parent is None else int(parent)

    def combinations(self) -> list[dict]:
        """Every parameter combination of the pre-registered grid (one empty dict if none)."""

        grid = self.optimise
        return [dict(zip(grid, combo, strict=True)) for combo in itertools.product(*grid.values())]

    def render(self) -> str:
        front = yaml.safe_dump(self.front, sort_keys=False, allow_unicode=True, width=100)
        return f"---\n{front}---\n{self.body.lstrip(chr(10))}"


def slugify(title: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:limit].rstrip("-") or "idea"


def parse_card(text: str, path: Path | None = None) -> Card:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, flags=re.DOTALL)
    if not match:
        raise CardError("a card starts with YAML front matter between '---' lines")
    front = yaml.safe_load(match.group(1)) or {}
    if not isinstance(front, dict):
        raise CardError("the front matter must be a mapping")
    return Card(front=front, body=match.group(2), path=path)


def read_card(path: Path) -> Card:
    return parse_card(path.read_text(encoding="utf-8"), path)


def card_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def validate(card: Card, criteria: Criteria) -> list[str]:
    """Every rule a card must satisfy before it can be registered; returns the problems."""

    problems = []
    front = card.front
    for key in ("title", "source", "taxonomy", "timeframe", "spec", "expected_trades_per_year"):
        if front.get(key) in (None, "", {}, []):
            problems.append(f"missing '{key}'")
    taxonomy = card.taxonomy
    if taxonomy.get("family") not in FAMILIES:
        problems.append(f"taxonomy.family must be one of {', '.join(FAMILIES)}")
    inputs = taxonomy.get("inputs") or []
    if not inputs or not set(inputs) <= set(INPUTS):
        problems.append(f"taxonomy.inputs must be a non-empty list from {', '.join(INPUTS)}")
    if taxonomy.get("horizon") not in HORIZONS:
        problems.append(f"taxonomy.horizon must be one of {', '.join(HORIZONS)}")
    timeframes = criteria.get("data.timeframes")
    if card.timeframe not in timeframes:
        problems.append(f"timeframe must be one of {', '.join(timeframes)}")
    if "type" not in card.spec:
        problems.append("spec needs a 'type'")

    grid = front.get("optimise") or {}
    if not isinstance(grid, dict):
        problems.append("optimise must map parameter -> list of values")
    else:
        limit = criteria.get("budgets.max_optimise_params")
        if len(grid) > limit:
            problems.append(f"optimise declares {len(grid)} parameters; at most {limit}")
        bad = [k for k, values in grid.items() if not isinstance(values, list) or len(values) < 2]
        problems += [f"optimise.{key} needs a list of at least 2 values" for key in bad]
        combos = math.prod(len(values) for values in grid.values()) if not bad else 0
        most = criteria.get("budgets.max_combinations")
        if combos > most:
            problems.append(f"optimise grid has {combos} combinations; at most {most}")
    expected = front.get("expected_trades_per_year")
    if expected is not None and not isinstance(expected, int | float):
        problems.append("expected_trades_per_year must be a number")

    for section in SECTIONS:
        if not re.search(rf"^##\s+{re.escape(section)}\s*$", card.body, flags=re.MULTILINE):
            problems.append(f"body needs a '## {section}' section")
    return problems


def require_valid(card: Card, criteria: Criteria) -> None:
    problems = validate(card, criteria)
    if problems:
        raise CardError("card rejected:\n  - " + "\n  - ".join(problems))


@dataclass
class Registration:
    """What ``research new``/``revise`` stamps onto a card."""

    idea: str
    version: int
    registered: str
    spec: dict
    parent: int | None = None
    reason: str | None = None
    extra: dict = field(default_factory=dict)


def stamp(card: Card, reg: Registration) -> Card:
    """Return the card as it will be stored: id, version and dates first, then the author's."""

    if not IDEA_ID.match(reg.idea):
        raise CardError(f"bad idea id '{reg.idea}'")
    front = {"id": reg.idea, "version": reg.version, "registered": reg.registered}
    if reg.parent is not None:
        front.update(parent=reg.parent, revision_reason=reg.reason)
    front.update({k: v for k, v in card.front.items() if k not in STAMP_KEYS})
    front["spec"] = reg.spec
    front.update(reg.extra)
    return Card(front=front, body=card.body)
