"""Strategy research pipeline: register ideas, gate them, keep the record.

Every subcommand applies the thresholds in research/criteria.yaml, writes its result under
research/ideas/<id>-<slug>/v<n>/, appends to research/journal.jsonl and commits those files
locally (never pushes).

Examples:
    python -m scripts.research seed
    python -m scripts.research new path/to/draft_card.md
    python -m scripts.research status
    python -m scripts.research build-check i002
    python -m scripts.research revise i002 path/to/draft_v2.md --reason "short side lost in every regime"
    python -m scripts.research abandon i002 --reason "needs open-interest data we do not have"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from algotrade.research.buildcheck import build_check
from algotrade.research.cards import CardError, read_card
from algotrade.research.criteria import load_criteria
from algotrade.research.index import write_index
from algotrade.research.journal import Journal, TrialLedger
from algotrade.research.registry import Refused, abandon, find_version, register, revise
from algotrade.research.seed import seed
from algotrade.research.split import HoldoutViolation
from algotrade.research.vcs import GitError
from algotrade.research.workspace import Workspace

PROBLEMS = (Refused, CardError, HoldoutViolation, GitError, FileNotFoundError)


def cmd_seed(ws: Workspace, args: argparse.Namespace) -> int:
    summary = seed(ws, load_criteria(ws.criteria_path), commit=not args.no_commit)
    print(
        f"Seeded: {summary['book_trials']} book-strategy configurations (idea i001, failed), "
        f"{summary['screened']} screened catalogue configurations, "
        f"{summary['fingerprints']} fingerprints; {summary['trials']} distinct trials in total."
    )
    return 0


def cmd_new(ws: Workspace, args: argparse.Namespace) -> int:
    criteria = load_criteria(ws.criteria_path)
    version = register(
        ws, criteria, read_card(Path(args.card)), retest=args.retest, commit=not args.no_commit
    )
    print(f"Registered {version.idea} v1 '{version.title}' -> {version.card_path}")
    print(f"Strategy spec: {version.spec}")
    return 0


def cmd_revise(ws: Workspace, args: argparse.Namespace) -> int:
    criteria = load_criteria(ws.criteria_path)
    version = revise(
        ws, criteria, args.idea, read_card(Path(args.card)), args.reason, not args.no_commit
    )
    print(f"Registered {version.idea} v{version.version} -> {version.card_path}")
    return 0


def cmd_abandon(ws: Workspace, args: argparse.Namespace) -> int:
    version = abandon(ws, args.idea, args.reason, commit=not args.no_commit)
    print(f"{version.idea} v{version.version} abandoned: {args.reason}")
    return 0


def cmd_status(ws: Workspace, args: argparse.Namespace) -> int:
    path = write_index(ws)
    ideas = Journal(ws.journal_path).ideas()
    print(f"{TrialLedger(ws.trials_path).count()} configurations evaluated so far")
    for idea in sorted(ideas.values(), key=lambda state: state.idea):
        latest = idea.latest
        reason = f" - {latest.failure}" if latest.failed else ""
        print(f"{idea.idea} v{latest.version} [{idea.status}] {idea.title}{reason}")
    print(f"Index written to {ws.relative(path)}")
    return 0


def cmd_build_check(ws: Workspace, args: argparse.Namespace) -> int:
    criteria = load_criteria(ws.criteria_path)
    version = find_version(ws, args.idea, args.version)
    outcome = build_check(ws, criteria, version, commit=not args.no_commit)
    result = outcome.result
    for check in result.checks:
        print(check.line())
    for note in result.notes:
        print(f"note: {note}")
    if not outcome.recorded:
        print("BUILD NOT READY (nothing recorded; fix and run again):")
        for problem in outcome.problems:
            print(f"  - {problem}")
        return 1
    print(f"BUILD {result.verdict}")
    for reason in result.reasons:
        print(f"  - {reason}")
    return 0 if result.verdict == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    main.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    main.add_argument("--no-commit", action="store_true", help="Write results but do not commit")
    sub = main.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="Import pre-journal experiments (once)").set_defaults(run=cmd_seed)
    new = sub.add_parser("new", help="Register an idea card")
    new.add_argument("card", help="Draft card (Markdown with YAML front matter)")
    new.add_argument("--retest", help="USER ONLY: reason to retest a failed idea's region")
    new.set_defaults(run=cmd_new)

    rev = sub.add_parser("revise", help="Register a revised version of an idea")
    rev.add_argument("idea")
    rev.add_argument("card")
    rev.add_argument("--reason", required=True)
    rev.set_defaults(run=cmd_revise)

    aba = sub.add_parser("abandon", help="Close an idea that cannot be built or tested")
    aba.add_argument("idea")
    aba.add_argument("--reason", required=True)
    aba.set_defaults(run=cmd_abandon)

    sub.add_parser("status", help="Refresh research/index.md and list ideas").set_defaults(
        run=cmd_status
    )
    build = sub.add_parser("build-check", help="Gate new strategy code before testing it")
    build.add_argument("idea")
    build.add_argument("--version", type=int)
    build.set_defaults(run=cmd_build_check)
    return main


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parser().parse_args(argv)
    ws = Workspace(root=args.root.resolve())
    try:
        return args.run(ws, args)
    except PROBLEMS as error:
        print(f"REFUSED: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
