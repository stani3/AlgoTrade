"""Where the research pipeline keeps its files, relative to a repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STAGES = ("build", "feasibility", "validation", "holdout", "incubation")


@dataclass(frozen=True)
class Workspace:
    """A repository checkout seen by the research pipeline.

    Everything under ``research/`` is committed; HTML reports go to the gitignored
    ``reports/research/`` tree.
    """

    root: Path
    data_root: Path | None = None

    @property
    def research(self) -> Path:
        return self.root / "research"

    @property
    def criteria_path(self) -> Path:
        return self.research / "criteria.yaml"

    @property
    def journal_path(self) -> Path:
        return self.research / "journal.jsonl"

    @property
    def trials_path(self) -> Path:
        return self.research / "trials.csv"

    @property
    def index_path(self) -> Path:
        return self.research / "index.md"

    @property
    def ideas_dir(self) -> Path:
        return self.research / "ideas"

    @property
    def fingerprints_dir(self) -> Path:
        return self.research / "fingerprints"

    @property
    def raw_data(self) -> Path:
        return self.data_root or self.root / "data" / "raw"

    @property
    def strategy_code_dir(self) -> Path:
        return self.root / "src" / "algotrade" / "strategies" / "ideas"

    @property
    def strategy_tests_dir(self) -> Path:
        return self.root / "tests" / "ideas"

    def idea_dir(self, idea: str, slug: str) -> Path:
        return self.ideas_dir / f"{idea}-{slug}"

    def version_dir(self, idea: str, slug: str, version: int) -> Path:
        return self.idea_dir(idea, slug) / f"v{version}"

    def stage_dir(self, idea: str, slug: str, version: int, stage: str) -> Path:
        if stage not in STAGES:
            raise ValueError(f"unknown stage '{stage}'")
        return self.version_dir(idea, slug, version) / stage

    def report_dir(self, idea: str, version: int, stage: str) -> Path:
        return self.root / "reports" / "research" / idea / f"v{version}" / stage

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()
