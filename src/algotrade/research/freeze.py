"""``research freeze``: fix the strategy that passed the holdout, tag it, write the decision page.

``frozen.json`` is exactly what incubation trades: the spec with the parameters chosen from the
feasibility plateau, the timeframe, the stake from the walk-forward Monte Carlo, the symbols and
the funding alignment the strategy was tested with.
The commit is tagged ``strategy/<id>-v<n>``, so the code that was validated can always be
checked out. ``decision.md`` is the one-page summary shown to the user, with the reports, before
they decide whether to incubate.
"""

from __future__ import annotations

import json

import pandas as pd

from . import vcs
from .criteria import Criteria
from .holdout import load_result
from .index import write_index
from .journal import Journal, VersionState, now
from .registry import Refused
from .split import for_version, funding_alignment, version_symbols
from .workspace import STAGES, Workspace


def tag_name(version: VersionState) -> str:
    return f"strategy/{version.idea}-v{version.version}"


def weaknesses(ws: Workspace, version: VersionState) -> list[str]:
    """Plain-language warnings from the feasibility diagnostics and the walk-forward."""

    folder = ws.stage_dir(version.idea, version.slug, version.version, "feasibility")
    found = []
    sides = folder / "diagnostics_sides.csv"
    if sides.exists():
        frame = pd.read_csv(sides, index_col=0)
        for side in ("long", "short"):
            if (
                side in frame.index
                and frame.at[side, "trades"] > 0
                and frame.at[side, "sum_return"] < 0
            ):
                found.append(f"{side} trades lost money overall in development data")
    regimes = folder / "diagnostics_regimes.csv"
    if regimes.exists():
        frame = pd.read_csv(regimes)
        weak = frame[(frame["median_sharpe"] < 0) & (frame["share_of_time"] >= 0.15)]
        found += [
            f"loses in the '{row.regime}' regime ({row.share_of_time:.0%} of the time, median "
            f"Sharpe {row.median_sharpe:.2f})"
            for row in weak.itertuples()
            if row.regime != "warm-up"
        ]
    oos = (
        ws.stage_dir(version.idea, version.slug, version.version, "validation")
        / "oos_by_symbol.csv"
    )
    if oos.exists():
        frame = pd.read_csv(oos, index_col=0)
        losers = frame.index[frame["sharpe"] < 0].tolist()
        if losers:
            found.append(f"negative out-of-sample Sharpe on {', '.join(losers)}")
    return found


def decision_markdown(ws: Workspace, version: VersionState, frozen: dict) -> str:
    ideas = Journal(ws.journal_path).ideas()[version.idea]
    lines = [
        f"# Decision: incubate {version.idea} v{version.version} '{version.title}'?",
        "",
        (
            "Every gate below passed. Incubation (paper trading on testnet) only starts if the user "
            "approves; going live with real money is always the user's own step."
        ),
        "",
        f"- Frozen spec: `{json.dumps(frozen['spec'])}`",
        f"- Timeframe: {frozen['timeframe']}, symbols: {', '.join(frozen['symbols'])}",
        f"- Stake (size multiplier meeting Davey's goals): {frozen['stake']}",
        f"- Git tag: `{frozen['tag']}`",
        *(
            [
                (
                    f"- Asset classes: {', '.join(frozen['universe'])}. Paper trading runs on "
                    "the Binance and Bybit testnets only, so this strategy stops here until a "
                    "venue for these markets is added (for example the NautilusTrader "
                    "Interactive Brokers adapter, or Alpaca paper trading)."
                )
            ]
            if "universe" in frozen
            else []
        ),
        "",
        "## Gates",
        "",
        "| Stage | Check | Value | Needs | Result |",
        "|---|---|---|---|---|",
    ]
    reports = []
    for stage in STAGES:
        if stage not in version.stages:
            continue
        result = load_result(ws, version, stage)
        for check in result["checks"]:
            value = check["value"]
            shown = f"{value:.3g}" if isinstance(value, float) else value
            verdict = "PASS" if check["passed"] else "FAIL"
            lines.append(f"| {stage} | {check['name']} | {shown} | {check['limit']} | {verdict} |")
        if result.get("report"):
            reports.append(f"- {stage}: `{result['report']}`")
    feasibility = load_result(ws, version, "feasibility")
    validation = load_result(ws, version, "validation")
    dsr = validation.get("deflated_sharpe", {})
    lines += [
        "",
        "## Parameters",
        "",
        (
            f"Chosen from the centre of the best plateau of the pre-registered grid: "
            f"`{json.dumps(feasibility.get('chosen', {}))}`."
        ),
        "",
        "## Overfitting",
        "",
        (
            f"Deflated Sharpe ratio {dsr.get('deflated_sharpe', 0):.2f} against "
            f"{dsr.get('trials', 0)} configurations tried across the whole research programme "
            f"(out-of-sample annualised Sharpe of the median symbol "
            f"{dsr.get('annualised_sharpe', 0):.2f})."
        ),
        "",
        "## Versions",
        "",
    ]
    for number, state in sorted(ideas.versions.items()):
        why = f" - revised: {state.reason}" if state.reason else ""
        lines.append(f"- v{number}: {state.status}{why}")
    lines += ["", "## Known weaknesses", ""]
    lines += [f"- {item}" for item in weaknesses(ws, version)] or ["- none flagged"]
    lines += ["", "## Reports", "", *reports, ""]
    return "\n".join(lines)


def freeze(ws: Workspace, criteria: Criteria, version: VersionState, commit: bool = True) -> dict:
    if version.stages.get("holdout", {}).get("verdict") != "PASS":
        raise Refused(f"{version.idea} v{version.version} needs a passing holdout first")
    if version.frozen is not None:
        raise Refused(f"{version.idea} v{version.version} is already frozen ({version.frozen})")
    criteria = for_version(ws, criteria, version)
    holdout = load_result(ws, version, "holdout")
    validation = load_result(ws, version, "validation")
    frozen = {
        "idea": version.idea,
        "version": version.version,
        "title": version.title,
        "spec": holdout["spec"],
        "timeframe": version.timeframe,
        "stake": validation.get("stake"),
        "symbols": version_symbols(criteria, version),
        "exchange": criteria.get("data.exchange") if version.universe == ("crypto",) else None,
        **({} if version.universe == ("crypto",) else {"universe": list(version.universe)}),
        "funding_alignment": funding_alignment(criteria),
        "criteria_hash": criteria.hash,
        "frozen_at": now(),
        "tag": tag_name(version),
    }
    folder = ws.version_dir(version.idea, version.slug, version.version)
    (folder / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    Journal(ws.journal_path).append(
        "frozen", idea=version.idea, version=version.version, tag=frozen["tag"]
    )
    state = Journal(ws.journal_path).ideas()[version.idea].versions[version.version]
    (folder / "decision.md").write_text(decision_markdown(ws, state, frozen), encoding="utf-8")
    if commit:
        paths = [folder / "frozen.json", folder / "decision.md", ws.journal_path, write_index(ws)]
        subject = f"research({version.idea}-{version.slug} v{version.version}): frozen"
        vcs.commit(ws.root, paths, subject)
        vcs.tag(ws.root, frozen["tag"], f"{version.idea} v{version.version} passed the holdout")
    return frozen
