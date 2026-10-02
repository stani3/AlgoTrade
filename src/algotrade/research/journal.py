"""Append-only research journal and trial ledger.

``journal.jsonl`` holds one event per line (ideas registered, stage verdicts, holdout looks,
freezes, abandons, seeding). ``trials.csv`` holds every strategy configuration ever evaluated,
which feeds duplicate detection and the deflated Sharpe ratio's trial count. Neither file is
ever rewritten, only appended to.
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

    def add(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        frame = pd.DataFrame(
            [{"ts": now(), **row} for row in rows],
        ).reindex(columns=TRIAL_COLUMNS)
        frame["spec"] = frame["spec"].map(
            lambda spec: spec if isinstance(spec, str) else json.dumps(spec, sort_keys=True)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists()
        frame.to_csv(self.path, mode="a", header=header, index=False, lineterminator="\n")
        return len(frame)

    def frame(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=TRIAL_COLUMNS)
        return pd.read_csv(self.path, dtype={"spec_hash": str, "idea": str, "label": str})

    def distinct(self) -> pd.DataFrame:
        """One row per (configuration, timeframe): the most recent evaluation of each."""

        frame = self.frame()
        return frame.drop_duplicates(["spec_hash", "timeframe"], keep="last")

    def count(self) -> int:
        return len(self.distinct())
