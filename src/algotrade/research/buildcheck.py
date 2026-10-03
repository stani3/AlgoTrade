"""``research build-check``: the gate between writing a strategy and testing it on data.

New strategy code must be lint-clean, pass its own tests with 100% line and branch coverage
(measured with Numba's JIT off so compiled loops count too), build every pre-registered grid
combination, and not duplicate an earlier idea - neither by configuration nor by behaviour.

Code-quality problems are reported and nothing is recorded: fix them and run again. A duplicate
is a verdict: it is journaled and committed, and the idea is closed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.instruments import universe_key
from algotrade.strategies.catalog import refresh_ideas

from . import fingerprint, results
from .cards import CardError, card_hash, read_card
from .criteria import Check, Criteria, at_least, at_most
from .dedup import core_rules, spec_hash
from .journal import Journal, VersionState
from .registry import IDEA_TYPE, Refused, check_builds, conflicts, strategy_types
from .split import dev_universe, for_version, funding_alignment, window
from .workspace import Workspace


@dataclass(frozen=True)
class Coverage:
    module: str
    tests_passed: bool
    line: float
    branch: float
    missing_lines: list[int] = field(default_factory=list)
    missing_branches: list[list[int]] = field(default_factory=list)
    output: str = ""


def measure(root: Path, module: Path, tests: Path, python: str = sys.executable) -> Coverage:
    """Run ``tests`` under branch coverage and report coverage of ``module`` alone."""

    with tempfile.TemporaryDirectory() as tmp:
        data, report = Path(tmp) / ".coverage", Path(tmp) / "coverage.json"
        env = {**os.environ, "NUMBA_DISABLE_JIT": "1", "COVERAGE_FILE": str(data)}
        run = subprocess.run(
            [python, "-m", "coverage", "run", "--branch", f"--include={module}", "-m", "pytest"]
            + ["-q", "-p", "no:cacheprovider", str(tests)],
            check=False,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        subprocess.run(
            [python, "-m", "coverage", "json", "-q", "-o", str(report), f"--include={module}"],
            check=False,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        files = json.loads(report.read_text())["files"] if report.exists() else {}
    entry = next(
        (v for k, v in files.items() if (root / k).resolve() == module.resolve()),
        None,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-15:])
    if entry is None:
        return Coverage(str(module), run.returncode == 0, 0.0, 0.0, output=tail)
    summary = entry["summary"]
    statements = summary["num_statements"]
    branches = summary["num_branches"]
    return Coverage(
        module=str(module),
        tests_passed=run.returncode == 0,
        line=summary["covered_lines"] / statements if statements else 1.0,
        branch=summary["covered_branches"] / branches if branches else 1.0,
        missing_lines=entry.get("missing_lines", []),
        missing_branches=entry.get("missing_branches", []),
        output=tail,
    )


def ruff_problems(root: Path, files: list[Path], python: str = sys.executable) -> list[str]:
    problems = []
    for args in (["check", "--quiet"], ["format", "--check", "--quiet"]):
        process = subprocess.run(
            [python, "-m", "ruff", *args, *map(str, files)],
            check=False,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.returncode != 0:
            problems.append(f"ruff {args[0]}: {(process.stdout + process.stderr).strip()}")
    return problems


def blocking(nearest: list[fingerprint.Similarity], spec: dict) -> list[fingerprint.Similarity]:
    """Behavioural duplicates that close the idea.

    * An earlier idea: always - testing the same bet twice is the duplicate this prevents.
    * A screened catalogue rule: only for new idea code. Catalogue rules were looked at by the
      scan but never tested through the gates, so registering one (or a blend or sizing of
      them) is how it gets tested properly; new code that re-creates one is a copy and should
      use the catalogue instead. Trend rules in crypto correlate highly with each other, so
      blocking every catalogue match would make the whole family untestable.
    * The buy & hold baseline: always, unless the candidate is buy & hold itself.
    """

    new_code = any(IDEA_TYPE.match(kind) for kind in strategy_types(spec))
    mine = core_rules(spec)

    def blocks(similarity: fingerprint.Similarity) -> bool:
        if not similarity.duplicate:
            return False
        if similarity.source == "screen":
            return new_code
        if similarity.source == "baseline":
            return not (similarity.spec and core_rules(similarity.spec) == mine)
        return True

    return [s for s in nearest if blocks(s)]


@dataclass
class BuildOutcome:
    result: results.StageResult
    recorded: bool
    problems: list[str] = field(default_factory=list)


def code_files(ws: Workspace, version: VersionState) -> list[tuple[str, Path, Path]]:
    """(strategy type, module, test file) for every strategy this idea had to write."""

    own = [
        kind
        for kind in strategy_types(version.spec)
        if IDEA_TYPE.match(kind) and kind.startswith(f"{version.idea}_")
    ]
    return [
        (t, ws.strategy_code_dir / f"{t}.py", ws.strategy_tests_dir / f"test_{t}.py")
        for t in dict.fromkeys(own)
    ]


def build_check(
    ws: Workspace, criteria: Criteria, version: VersionState, commit: bool = True
) -> BuildOutcome:
    criteria = for_version(ws, criteria, version)
    card_path = ws.root / version.card_path
    if card_hash(card_path) != version.card_hash:
        raise Refused(f"{version.card_path} was edited after registration; use `research revise`")
    if version.stages.get("build", {}).get("verdict") == "PASS":
        raise Refused(f"{version.idea} v{version.version} is already built")
    if version.failed:
        raise Refused(f"{version.idea} v{version.version} is closed ({version.status})")
    card = read_card(card_path)

    problems: list[str] = []
    checks: list[Check] = []
    coverage = []
    files = code_files(ws, version)
    for kind, module, tests in files:
        if not module.exists() or not tests.exists():
            problems.append(f"{kind}: expected {ws.relative(module)} and {ws.relative(tests)}")
    if not problems and files:
        lint = ruff_problems(ws.root, [p for _, m, t in files for p in (m, t)])
        problems += lint
        checks.append(at_most("ruff problems", len(lint), 0))
        for _, module, tests in files:
            measured = measure(ws.root, module, tests)
            coverage.append(measured)
            name = module.stem
            checks.append(at_least(f"tests pass ({name})", float(measured.tests_passed), 1))
            checks += [
                at_least(f"line coverage ({name})", measured.line, criteria.get("build.min_line_coverage"), "{:.0%}"),
                at_least(f"branch coverage ({name})", measured.branch, criteria.get("build.min_branch_coverage"), "{:.0%}"),
            ]  # fmt: skip
            if not measured.tests_passed:
                problems.append(f"tests failing for {name}:\n{measured.output}")
            if measured.missing_lines or measured.missing_branches:
                problems.append(
                    f"{name}: uncovered lines {measured.missing_lines}, "
                    f"branches {measured.missing_branches}"
                )

    if not problems:
        refresh_ideas()
        try:
            check_builds(card.spec, card.optimise)
        except CardError as error:
            problems.append(str(error))

    base = results.StageResult(version.idea, version.version, "build", checks)
    if problems or not all(c.passed for c in checks):
        base.failures = problems
        return BuildOutcome(base, recorded=False, problems=problems)

    # Duplicate rules, now that defaults can be filled in from the built classes.
    ideas = Journal(ws.journal_path).ideas()
    tolerance = criteria.get("dedup.param_tolerance")
    others = {k: v for k, v in ideas.items() if k != version.idea}
    own = {version.idea: ideas[version.idea]} if version.idea in ideas else {}
    universe = version.universe
    found = conflicts(
        others, card.spec, card.optimise, card.timeframe, tolerance, universe=universe
    )
    # The user's --retest at registration also covers the failed ideas it overlaps here.
    found = [c for c in found if not (c.failed and version.retest)]
    found += [
        c
        for c in conflicts(
            own, card.spec, card.optimise, card.timeframe, tolerance, version.idea, universe
        )
        if c.kind == "exact" and c.version != version.version
    ]
    failures = [f"duplicate: {c.describe()}" for c in found]

    symbols = criteria.get("data.fingerprint_symbols")
    universe = dev_universe(ws, criteria, card.timeframe, symbols)
    costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
    frame = fingerprint.compute(card.spec, universe, costs)
    store = fingerprint.FingerprintStore(ws.fingerprints_dir)
    own_hash = spec_hash(card.spec)
    nearest = store.nearest(frame, criteria, exclude_idea=version.idea, exclude_hash=own_hash)
    failures += [f"behavioural duplicate of {s.describe()}" for s in blocking(nearest, card.spec)]

    windows = [window(bars, symbol, card.timeframe) for symbol, bars in universe.items()]
    result = results.StageResult(
        version.idea,
        version.version,
        "build",
        checks,
        failures=failures,
        provenance=results.provenance(ws, criteria, version, own_hash, windows),
        extra={
            "nearest": [s.to_dict() for s in nearest[:5]],
            "coverage": [c.__dict__ for c in coverage],
            "modules": [ws.relative(m) for _, m, _ in files],
        },
        notes=[f"nearest earlier strategy: {nearest[0].describe()}"] if nearest else [],
    )
    extra_paths = [p for _, m, t in files for p in (m, t)]
    if result.verdict == "PASS":
        key = store.key(own_hash, card.timeframe, universe_key(universe))
        extra_paths += store.save(
            key,
            frame,
            {
                "spec": card.spec,
                "spec_hash": own_hash,
                "timeframe": card.timeframe,
                "source": "idea",
                "idea": version.idea,
                "version": version.version,
                "label": f"{version.idea} v{version.version} {version.title}",
                "symbols": symbols,
                "exchange": criteria.get("data.exchange"),
                "funding_alignment": funding_alignment(criteria),
            },
        )
    results.record(ws, version, result, extra_paths, commit=commit)
    return BuildOutcome(result, recorded=True)
