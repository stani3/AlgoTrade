"""Strategy research pipeline: register ideas, gate them, keep the record.

Every subcommand applies the thresholds in research/criteria.yaml, writes its result under
research/ideas/<id>-<slug>/v<n>/, appends to research/journal.jsonl and commits those files
locally (never pushes).

Examples:
    python -m scripts.research seed
    python -m scripts.research new path/to/draft_card.md
    python -m scripts.research status
    python -m scripts.research build-check i002
    python -m scripts.research feasibility i002
    python -m scripts.research validate i002
    python -m scripts.research holdout i002
    python -m scripts.research freeze i002
    python -m scripts.research report i002 validation
    python -m scripts.research incubate i002                  # only after the user approved
    python -m scripts.research incubation-report i002
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
from algotrade.research.feasibility import feasibility
from algotrade.research.freeze import freeze
from algotrade.research.holdout import holdout
from algotrade.research.incubation import incubation_report
from algotrade.research.incubation import start as start_incubation
from algotrade.research.index import write_index
from algotrade.research.journal import Journal, TrialLedger
from algotrade.research.registry import Refused, abandon, find_version, register, revise
from algotrade.research.reproduce import reproduce
from algotrade.research.seed import seed
from algotrade.research.split import HoldoutViolation
from algotrade.research.validate import validate
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


def print_result(result) -> int:
    for check in result.checks:
        print(check.line())
    for key, value in result.extra.get("chosen", {}).items():
        print(f"chosen {key} = {value}")
    if result.report:
        print(f"report: {result.report}")
    print(f"{result.stage.upper()} {result.verdict}")
    for reason in result.reasons:
        print(f"  - {reason}")
    return 0 if result.verdict == "PASS" else 1


def cmd_feasibility(ws: Workspace, args: argparse.Namespace) -> int:
    criteria = load_criteria(ws.criteria_path)
    version = find_version(ws, args.idea, args.version)
    result = feasibility(
        ws, criteria, version, commit=not args.no_commit, report=not args.no_report
    )
    return print_result(result)


def cmd_validate(ws: Workspace, args: argparse.Namespace) -> int:
    criteria = load_criteria(ws.criteria_path)
    version = find_version(ws, args.idea, args.version)
    result = validate(ws, criteria, version, commit=not args.no_commit, report=not args.no_report)
    if result.extra.get("stake"):
        print(f"stake {result.extra['stake']:g}x")
    return print_result(result)


def cmd_holdout(ws: Workspace, args: argparse.Namespace) -> int:
    if args.force and not args.reason:
        raise Refused("a forced second holdout look needs --reason (the user's)")
    criteria = load_criteria(ws.criteria_path)
    version = find_version(ws, args.idea, args.version)
    result = holdout(
        ws, criteria, version, force=args.force, reason=args.reason or "",
        commit=not args.no_commit, report=not args.no_report,
    )  # fmt: skip
    return print_result(result)


def cmd_freeze(ws: Workspace, args: argparse.Namespace) -> int:
    criteria = load_criteria(ws.criteria_path)
    version = find_version(ws, args.idea, args.version)
    frozen = freeze(ws, criteria, version, commit=not args.no_commit)
    folder = ws.version_dir(version.idea, version.slug, version.version)
    print(f"Frozen {frozen['idea']} v{frozen['version']} as {frozen['tag']}: {frozen['spec']}")
    print(f"stake {frozen['stake']}x on {frozen['timeframe']} bars")
    print(f"Decision page: {ws.relative(folder / 'decision.md')}")
    return 0


def cmd_incubate(ws: Workspace, args: argparse.Namespace) -> int:
    version = find_version(ws, args.idea, args.version)
    record = start_incubation(ws, version, when=args.start, commit=not args.no_commit)
    print(f"Incubation of {version.idea} v{version.version} started {record['start']}")
    print(f"Paper-trade it with: python -m scripts.paper_trade {version.idea}")
    return 0


def cmd_incubation_report(ws: Workspace, args: argparse.Namespace) -> int:
    criteria = load_criteria(ws.criteria_path)
    version = find_version(ws, args.idea, args.version)
    outcome = incubation_report(
        ws, criteria, version, commit=not args.no_commit, report=not args.no_report
    )
    for check in outcome.result.checks:
        print(check.line())
    for note in outcome.result.notes:
        print(f"note: {note}")
    if outcome.result.report:
        print(f"report: {outcome.result.report}")
    print(
        f"INCUBATION {outcome.status.upper()} ({outcome.days:.0f} days, {outcome.trades:.0f} trades)"
    )
    return 1 if outcome.status == "fail" else 0


def cmd_report(ws: Workspace, args: argparse.Namespace) -> int:
    criteria = load_criteria(ws.criteria_path)
    version = find_version(ws, args.idea, args.version)
    print(f"Report written to {reproduce(ws, criteria, version, args.stage)}")
    return 0


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

    feas = sub.add_parser("feasibility", help="Davey's limited testing on development data")
    feas.add_argument("idea")
    feas.add_argument("--version", type=int)
    feas.add_argument("--no-report", action="store_true", help="Skip the HTML report")
    feas.set_defaults(run=cmd_feasibility)

    val = sub.add_parser("validate", help="Walk-forward, deflated Sharpe and Monte Carlo stake")
    val.add_argument("idea")
    val.add_argument("--version", type=int)
    val.add_argument("--no-report", action="store_true")
    val.set_defaults(run=cmd_validate)

    hold = sub.add_parser("holdout", help="The one look at the holdout data")
    hold.add_argument("idea")
    hold.add_argument("--version", type=int)
    hold.add_argument("--force", action="store_true", help="USER ONLY: a second look")
    hold.add_argument("--reason", help="Why the user forces a second look")
    hold.add_argument("--no-report", action="store_true")
    hold.set_defaults(run=cmd_holdout)

    frz = sub.add_parser("freeze", help="Freeze, tag and summarise a strategy that passed")
    frz.add_argument("idea")
    frz.add_argument("--version", type=int)
    frz.set_defaults(run=cmd_freeze)

    inc = sub.add_parser("incubate", help="Start incubation of a frozen strategy (user approved)")
    inc.add_argument("idea")
    inc.add_argument("--version", type=int)
    inc.add_argument("--start", help="Start time (default: now, UTC)")
    inc.set_defaults(run=cmd_incubate)

    incr = sub.add_parser("incubation-report", help="Score the shadow forward test and fills")
    incr.add_argument("idea")
    incr.add_argument("--version", type=int)
    incr.add_argument("--no-report", action="store_true")
    incr.set_defaults(run=cmd_incubation_report)

    rep = sub.add_parser("report", help="Rebuild a stage's HTML report from the committed result")
    rep.add_argument("idea")
    rep.add_argument("stage", choices=["feasibility", "validation", "holdout"])
    rep.add_argument("--version", type=int)
    rep.set_defaults(run=cmd_report)
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
