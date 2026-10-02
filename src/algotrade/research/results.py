"""Stage results: ``result.json`` + ``summary.md`` with provenance, journaled and committed."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import vcs
from .criteria import Check, Criteria
from .journal import Journal, VersionState, now
from .split import DataWindow
from .workspace import Workspace

NEWLINE = chr(10)


@dataclass
class StageResult:
    idea: str
    version: int
    stage: str
    checks: list[Check]
    metrics: dict = field(default_factory=dict)
    trials: int = 0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    report: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        failed = self.failures or [c for c in self.checks if not c.passed]
        return "FAIL" if failed else "PASS"

    @property
    def reasons(self) -> list[str]:
        return self.failures + [
            f"{c.name} {_value(c.value)} (needs {c.limit})" for c in self.checks if not c.passed
        ]

    def to_dict(self) -> dict:
        return {
            "idea": self.idea,
            "version": self.version,
            "stage": self.stage,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "checks": [c.to_dict() for c in self.checks],
            "metrics": self.metrics,
            "trials": self.trials,
            "notes": self.notes,
            "report": self.report,
            "provenance": self.provenance,
            **self.extra,
        }


def _value(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def provenance(
    ws: Workspace,
    criteria: Criteria,
    version: VersionState,
    spec_hash: str,
    windows: list[DataWindow] = (),
) -> dict:
    return {
        "created": now(),
        "commit": vcs.head(ws.root),
        "criteria_hash": criteria.hash,
        "card_hash": version.card_hash,
        "spec_hash": spec_hash,
        "data": [w.to_dict() for w in windows],
    }


def summary_markdown(result: StageResult, title: str) -> str:
    lines = [
        f"# {result.idea} v{result.version} - {title}",
        "",
        f"**{result.stage.capitalize()}: {result.verdict}**",
        "",
    ]
    if result.reasons:
        lines += ["Why it failed:", ""] + [f"- {reason}" for reason in result.reasons] + [""]
    if result.checks:
        lines += ["| Check | Value | Needs | Result |", "|---|---|---|---|"]
        lines += [
            f"| {c.name} | {_value(c.value)} | {c.limit} | {'PASS' if c.passed else 'FAIL'} |"
            for c in result.checks
        ]
        lines.append("")
    if result.metrics:
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| {key} | {_value(value)} |" for key, value in result.metrics.items()]
        lines.append("")
    short = [note for note in result.notes if NEWLINE not in note]
    if short:
        lines += ["Notes:", ""] + [f"- {note}" for note in short] + [""]
    lines += [part for note in result.notes if NEWLINE in note for part in (note, "")]
    if result.report:
        lines += [f"Report: `{result.report}`", ""]
    lines += [
        f"Configurations evaluated in this stage: {result.trials}",
        "",
        (
            f"Commit at run time: `{result.provenance.get('commit')}` | criteria "
            f"`{result.provenance.get('criteria_hash')}` | card "
            f"`{result.provenance.get('card_hash')}`"
        ),
        "",
    ]
    return "\n".join(lines)


def write(ws: Workspace, version: VersionState, result: StageResult) -> list[Path]:
    folder = ws.stage_dir(version.idea, version.slug, version.version, result.stage)
    folder.mkdir(parents=True, exist_ok=True)
    data = folder / "result.json"
    data.write_text(
        json.dumps(result.to_dict(), indent=2, default=str, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    summary = folder / "summary.md"
    summary.write_text(summary_markdown(result, version.title), encoding="utf-8")
    return [folder]


def record(
    ws: Workspace,
    version: VersionState,
    result: StageResult,
    extra_paths: list[Path] = (),
    commit: bool = True,
) -> str | None:
    """Write the result, journal the verdict, refresh the index and commit; returns the SHA."""

    from .index import write_index  # local: index imports results' consumers

    paths = write(ws, version, result)
    Journal(ws.journal_path).append(
        "stage_result",
        idea=version.idea,
        version=version.version,
        stage=result.stage,
        verdict=result.verdict,
        failures=result.reasons,
        trials=result.trials,
        result=ws.relative(paths[0] / "result.json"),
    )
    paths += [ws.journal_path, ws.trials_path, write_index(ws), *extra_paths]
    if not commit:
        return None
    subject = f"research({version.idea}-{version.slug} v{version.version}): {result.stage} "
    subject += result.verdict
    if result.verdict == "FAIL" and result.reasons:
        subject += f" - {result.reasons[0]}"
    return vcs.commit(ws.root, paths, subject[:120])
