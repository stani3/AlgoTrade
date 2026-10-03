"""Append-only research journal and trial ledger.

``journal.jsonl`` holds one event per line (ideas registered, stage verdicts, holdout looks,
freezes, abandons, seeding). ``trials.csv`` holds every strategy configuration ever evaluated,
which feeds duplicate detection and the deflated Sharpe ratio's trial count. Neither file is
ever rewritten, only appended to, with one exception: ``research migrate-trials`` adds the
``universe`` column to a ledger written before ideas could name their asset classes (every
earlier row is crypto).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .workspace import STAGES

TRIAL_COLUMNS = [
    "ts",
    "idea",
    "version",
    "stage",
    "kind",
    "label",
    "spec_hash",
    "timeframe",
    "metric",
    "observations",
    "window",
    "spec",
]
UNIVERSE_COLUMN = "universe"
CRYPTO_ONLY = ("crypto",)


class LedgerFormatError(RuntimeError):
    """The trial ledger's columns do not fit what is being added (or migrated)."""


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class VersionState:
    """Everything the journal knows about one version of an idea."""

    idea: str
    version: int
    slug: str
    title: str
    timeframe: str
    spec: dict
    grid: dict
    taxonomy: dict
    card_hash: str
    card_path: str
    parent: int | None = None
    reason: str | None = None
    retest: str | None = None
    historical: bool = False
    stages: dict[str, dict] = field(default_factory=dict)
    holdout_looks: int = 0
    frozen: str | None = None
    abandoned: str | None = None
    incubation_start: str | None = None
    universe: tuple[str, ...] = CRYPTO_ONLY  # the asset classes the card registered
    symbols: tuple[str, ...] | None = None  # their symbols at registration (None: older cards)

    @property
    def last_stage(self) -> str | None:
        done = [stage for stage in STAGES if stage in self.stages]
        return done[-1] if done else None

    @property
    def status(self) -> str:
        if self.abandoned is not None:
            return "abandoned"
        last = self.last_stage
        failed = [s for s in STAGES if self.stages.get(s, {}).get("verdict") == "FAIL"]
        if failed:
            return f"failed:{failed[0]}"
        if last == "incubation":
            return "passed:incubation"
        if self.incubation_start is not None:
            return "incubating"
        if self.frozen is not None:
            return "frozen"
        return "registered" if last is None else f"passed:{last}"

    @property
    def failed(self) -> bool:
        return self.status == "abandoned" or self.status.startswith("failed:")

    @property
    def failure(self) -> str:
        if self.abandoned is not None:
            return self.abandoned
        for stage in STAGES:
            entry = self.stages.get(stage, {})
            if entry.get("verdict") == "FAIL":
                return "; ".join(entry.get("failures") or []) or f"{stage} failed"
        return ""


@dataclass
class IdeaState:
    idea: str
    versions: dict[int, VersionState] = field(default_factory=dict)

    @property
    def latest(self) -> VersionState:
        return self.versions[max(self.versions)]

    @property
    def slug(self) -> str:
        return self.versions[min(self.versions)].slug

    @property
    def title(self) -> str:
        return self.latest.title

    @property
    def status(self) -> str:
        return self.latest.status

    @property
    def failed(self) -> bool:
        return self.latest.failed


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: str, **fields) -> dict:
        entry = {"ts": now(), "event": event, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def ideas(self) -> dict[str, IdeaState]:
        ideas: dict[str, IdeaState] = {}
        for entry in self.entries():
            event = entry["event"]
            if event == "idea_registered":
                state = VersionState(
                    idea=entry["idea"],
                    version=int(entry["version"]),
                    slug=entry["slug"],
                    title=entry["title"],
                    timeframe=entry["timeframe"],
                    spec=entry["spec"],
                    grid=entry.get("grid") or {},
                    taxonomy=entry.get("taxonomy") or {},
                    card_hash=entry["card_hash"],
                    card_path=entry["card_path"],
                    parent=entry.get("parent"),
                    reason=entry.get("reason"),
                    retest=entry.get("retest"),
                    historical=bool(entry.get("historical")),
                    universe=tuple(entry.get("universe") or CRYPTO_ONLY),
                    symbols=tuple(entry["symbols"]) if entry.get("symbols") else None,
                )
                ideas.setdefault(state.idea, IdeaState(state.idea)).versions[state.version] = state
                continue
            if "idea" not in entry or entry["idea"] not in ideas:
                continue
            versions = ideas[entry["idea"]].versions
            if int(entry.get("version", 0)) not in versions:
                continue
            version = versions[int(entry["version"])]
            if event == "stage_result":
                version.stages[entry["stage"]] = entry
            elif event == "holdout_look":
                version.holdout_looks += 1
            elif event == "frozen":
                version.frozen = entry.get("tag", "")
            elif event == "abandoned":
                version.abandoned = entry.get("reason", "")
            elif event == "incubation_started":
                version.incubation_start = entry.get("start")
        return ideas

    def next_id(self) -> str:
        numbers = [int(idea[1:]) for idea in self.ideas()]
        return f"i{max(numbers, default=0) + 1:03d}"

    def seeded(self) -> bool:
        return any(entry["event"] == "seeded" for entry in self.entries())


class TrialLedger:
    """Every configuration evaluated, one row each (``trials.csv``)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _header(self) -> list[str] | None:
        if not self.path.exists():
            return None
        with self.path.open(encoding="utf-8") as handle:
            return handle.readline().strip().split(",")

    def add(self, rows: list[dict]) -> int:
        """Append rows; a row without a ``universe`` is crypto.

        A ledger from before universes existed keeps its columns while only crypto rows are
        added; the first row for another universe needs ``research migrate-trials`` first.
        """

        if not rows:
            return 0
        rows = [{**row, UNIVERSE_COLUMN: row.get(UNIVERSE_COLUMN) or "crypto"} for row in rows]
        header = self._header()
        columns = [*TRIAL_COLUMNS, UNIVERSE_COLUMN]
        if header == TRIAL_COLUMNS:
            others = sorted({row[UNIVERSE_COLUMN] for row in rows} - {"crypto"})
            if others:
                raise LedgerFormatError(
                    f"{self.path.name} has no universe column yet; run `python -m "
                    f"scripts.research migrate-trials` before adding {others[0]} trials"
                )
            columns = TRIAL_COLUMNS
        frame = pd.DataFrame(
            [{"ts": now(), **row} for row in rows],
        ).reindex(columns=columns)
        frame["spec"] = frame["spec"].map(
            lambda spec: spec if isinstance(spec, str) else json.dumps(spec, sort_keys=True)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.path, mode="a", header=header is None, index=False, lineterminator="\n")
        return len(frame)

    def frame(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=[*TRIAL_COLUMNS, UNIVERSE_COLUMN])
        frame = pd.read_csv(
            self.path, dtype={"spec_hash": str, "idea": str, "label": str, UNIVERSE_COLUMN: str}
        )
        if UNIVERSE_COLUMN not in frame:
            frame[UNIVERSE_COLUMN] = "crypto"
        return frame

    def distinct(self) -> pd.DataFrame:
        """One row per (configuration, timeframe, universe): the most recent evaluation of each.
        The same rules tried on another universe are another trial."""

        frame = self.frame()
        return frame.drop_duplicates(["spec_hash", "timeframe", UNIVERSE_COLUMN], keep="last")

    def count(self) -> int:
        return len(self.distinct())


def migrate_trials(path: Path) -> int:
    """Add the ``universe`` column (``crypto`` on every row) to a ledger from before universes.

    Each line keeps its text and gains ``,crypto``; returns the number of rows (0 if there is
    no ledger). A ledger that already has the column is refused.
    """

    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[0].split(",") != TRIAL_COLUMNS:
        raise LedgerFormatError(f"{path.name} already has a universe column (or unknown columns)")
    migrated = [f"{lines[0]},{UNIVERSE_COLUMN}"] + [f"{line},crypto" for line in lines[1:]]
    path.write_text("\n".join(migrated) + "\n", encoding="utf-8", newline="\n")
    return len(lines) - 1
