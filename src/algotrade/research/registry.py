"""Registering, revising and abandoning ideas, with the duplicate rules enforced in code.

Before a card is stored it must not repeat any registered configuration exactly, overlap a
registered idea's parameter region on the same timeframe and universe, or be the same rules on
another timeframe or universe (that is a revision of the existing idea). Earlier ideas with the same family, horizon
and an input in common must each be answered with a ``differs_from`` note. Matches against a
failed idea are only allowed with an explicit ``--retest`` reason from the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from algotrade.instruments import symbols_for
from algotrade.strategies import STRATEGIES

from . import vcs
from .cards import Card, CardError, Registration, card_hash, require_valid, stamp
from .criteria import Criteria
from .dedup import canonical, config_hashes, get_path, grid_specs, known_type, region
from .index import write_index
from .journal import IdeaState, Journal, VersionState, now
from .workspace import STAGES, Workspace

IDEA_TYPE = re.compile(r"^(i\d{3})_")


class Refused(RuntimeError):
    """Registration refused by a research rule."""


@dataclass(frozen=True)
class Conflict:
    kind: str  # exact | overlap | variant
    idea: str
    version: int
    failed: bool
    title: str

    def describe(self) -> str:
        what = {
            "exact": "repeats a configuration already registered by",
            "overlap": "overlaps the parameter region of",
            "variant": "is the same rules on another timeframe or universe as",
        }[self.kind]
        state = "failed" if self.failed else "active"
        return f"{what} {self.idea} v{self.version} '{self.title}' ({state})"


def strategy_types(spec: object) -> list[str]:
    if isinstance(spec, dict):
        found = [spec["type"]] if "type" in spec else []
        for value in spec.values():
            found += strategy_types(value)
        return found
    if isinstance(spec, list):
        return [t for item in spec for t in strategy_types(item)]
    return []


def rename_new_types(spec: object, idea: str) -> object:
    """Prefix strategy types that do not exist yet with the idea id (``foo`` -> ``i007_foo``)."""

    if isinstance(spec, dict):
        out = {key: rename_new_types(value, idea) for key, value in spec.items()}
        kind = spec.get("type")
        if isinstance(kind, str) and kind not in STRATEGIES and not IDEA_TYPE.match(kind):
            out["type"] = f"{idea}_{kind}"
        return out
    if isinstance(spec, list):
        return [rename_new_types(item, idea) for item in spec]
    return spec


def conflicts(
    ideas: dict[str, IdeaState],
    spec: dict,
    grid: dict,
    timeframe: str,
    tolerance: float,
    skip_overlap_for: str | None = None,
    universe: tuple[str, ...] = ("crypto",),
) -> list[Conflict]:
    """Registered versions this configuration repeats (``exact``), overlaps on the same
    timeframe and universe (``overlap``), or repeats on another timeframe or universe
    (``variant``)."""

    mine = config_hashes(spec, grid)
    my_region = region(spec, grid, timeframe, tolerance)
    found = []
    for idea in ideas.values():
        for version in idea.versions.values():
            theirs = config_hashes(version.spec, version.grid)
            same_cell = version.timeframe == timeframe and version.universe == tuple(universe)
            kind = None
            if same_cell and mine & theirs:
                kind = "exact"
            elif idea.idea != skip_overlap_for:
                other = region(version.spec, version.grid, version.timeframe, tolerance)
                if my_region.overlaps(other):
                    kind = "overlap" if same_cell else "variant"
            if kind:
                found.append(Conflict(kind, idea.idea, version.version, idea.failed, idea.title))
    return found


def neighbours(ideas: dict[str, IdeaState], taxonomy: dict, skip: str | None = None) -> list[str]:
    """Ideas with the same family and horizon and at least one input in common."""

    inputs = set(taxonomy.get("inputs") or [])
    found = []
    for idea in ideas.values():
        if idea.idea == skip:
            continue
        theirs = idea.latest.taxonomy
        if (
            theirs.get("family") == taxonomy.get("family")
            and theirs.get("horizon") == taxonomy.get("horizon")
            and inputs & set(theirs.get("inputs") or [])
        ):
            found.append(idea.idea)
    return found


def unjustified(card: Card, ids: list[str], min_chars: int) -> list[str]:
    notes = card.differs_from
    return [idea for idea in ids if len(notes.get(idea, "").strip()) < min_chars]


def check_builds(spec: dict, grid: dict) -> None:
    """Every grid combination of a spec made of existing strategies must construct.

    Specs naming strategies that are not written yet are checked by ``build-check`` instead.
    """

    if not known_type(spec):
        return
    try:
        full = canonical(spec)
        for path in grid:
            get_path(full, path)
        for combo in grid_specs(spec, grid):
            canonical(combo)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise CardError(f"the spec or its optimise grid does not build: {error!r}") from error


def _check_types(spec: dict, own: str | None) -> None:
    for kind in strategy_types(spec):
        match = IDEA_TYPE.match(kind)
        if match and match.group(1) != own:
            raise Refused(
                f"strategy type '{kind}' belongs to idea {match.group(1)}; a variant of it goes "
                f"through `research revise {match.group(1)}`"
            )


def _store(
    ws: Workspace,
    journal: Journal,
    draft: Card,
    reg: Registration,
    slug: str,
    retest: str | None,
    commit: bool,
    universe: tuple[str, ...],
    symbols: list[str],
) -> VersionState:
    card = stamp(draft, reg)
    folder = ws.version_dir(reg.idea, slug, reg.version)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "idea.md"
    path.write_text(card.render(), encoding="utf-8", newline="\n")
    journal.append(
        "idea_registered",
        idea=reg.idea,
        version=reg.version,
        slug=slug,
        title=card.title,
        timeframe=card.timeframe,
        spec=card.spec,
        grid=card.optimise,
        taxonomy=card.taxonomy,
        source_id=card.front.get("source_id"),
        card_hash=card_hash(path),
        card_path=ws.relative(path),
        parent=reg.parent,
        reason=reg.reason,
        retest=retest,
        universe=list(universe),
        symbols=list(symbols),
    )
    paths = [folder, ws.journal_path, write_index(ws)]
    if commit:
        action = "registered" if reg.parent is None else f"revised - {reg.reason}"
        vcs.commit(ws.root, paths, f"research({reg.idea}-{slug} v{reg.version}): {action}"[:120])
    return journal.ideas()[reg.idea].versions[reg.version]


def register(
    ws: Workspace,
    criteria: Criteria,
    draft: Card,
    retest: str | None = None,
    commit: bool = True,
) -> VersionState:
    """``research new``: validate, check for duplicates, assign an id, store and commit."""

    require_valid(draft, criteria)
    journal = Journal(ws.journal_path)
    ideas = journal.ideas()
    _check_types(draft.spec, own=None)
    idea = journal.next_id()
    spec = rename_new_types(draft.spec, idea)
    check_builds(spec, draft.optimise)

    universe = draft.universe(criteria)
    found = conflicts(
        ideas, spec, draft.optimise, draft.timeframe, criteria.get("dedup.param_tolerance"),
        universe=universe,
    )  # fmt: skip
    blocking = [c for c in found if not (c.failed and retest)]
    if blocking:
        hints = []
        if any(c.kind == "variant" and not c.failed for c in blocking):
            hints.append(
                "A different timeframe or universe of an active idea goes through "
                "`research revise`."
            )
        if any(c.failed for c in blocking):
            hints.append("Failed ideas stay failed; only the user can override with --retest.")
        raise Refused(
            "duplicate of an earlier idea - this card:\n  - "
            + "\n  - ".join(c.describe() for c in blocking)
            + ("\n" + "\n".join(hints) if hints else "")
        )
    missing = unjustified(
        draft, neighbours(ideas, draft.taxonomy), criteria.get("dedup.min_differs_from_chars")
    )
    if missing:
        raise Refused(
            "these earlier ideas share family, horizon and an input; explain how this one "
            f"differs under differs_from: {', '.join(missing)}"
        )
    reg = Registration(idea=idea, version=1, registered=now(), spec=spec)
    symbols = symbols_for(criteria, universe)
    return _store(ws, journal, draft, reg, draft.slug, retest, commit, universe, symbols)


def revise(
    ws: Workspace,
    criteria: Criteria,
    idea: str,
    draft: Card,
    reason: str,
    commit: bool = True,
) -> VersionState:
    """``research revise``: a new version of an idea, within the revision budget."""

    if not reason or not reason.strip():
        raise Refused("a revision needs a reason tied to what the tests showed")
    journal = Journal(ws.journal_path)
    ideas = journal.ideas()
    if idea not in ideas:
        raise Refused(f"no idea {idea}")
    state = ideas[idea]
    used = len(state.versions) - 1
    budget = criteria.get("budgets.revisions_per_idea")
    if used >= budget:
        raise Refused(f"{idea} has used its revision budget ({used} of {budget})")
    late = [
        f"v{v.version} {stage}"
        for v in state.versions.values()
        for stage in STAGES[STAGES.index("validation") :]
        if stage in v.stages
    ]
    if late:
        raise Refused(f"no revisions after validation has run ({', '.join(late)})")
    require_valid(draft, criteria)
    _check_types(draft.spec, own=idea)
    spec = rename_new_types(draft.spec, idea)
    check_builds(spec, draft.optimise)
    universe = draft.universe(criteria)
    found = conflicts(
        ideas,
        spec,
        draft.optimise,
        draft.timeframe,
        criteria.get("dedup.param_tolerance"),
        skip_overlap_for=idea,
        universe=universe,
    )
    if found:
        raise Refused(
            "the revision is a duplicate:\n  - " + "\n  - ".join(c.describe() for c in found)
        )
    missing = unjustified(
        draft,
        neighbours(ideas, draft.taxonomy, skip=idea),
        criteria.get("dedup.min_differs_from_chars"),
    )
    if missing:
        raise Refused(f"explain under differs_from how this differs from: {', '.join(missing)}")
    latest = state.latest
    reg = Registration(
        idea=idea,
        version=latest.version + 1,
        registered=now(),
        spec=spec,
        parent=latest.version,
        reason=reason.strip(),
    )
    symbols = symbols_for(criteria, universe)
    return _store(ws, journal, draft, reg, state.slug, None, commit, universe, symbols)


def abandon(ws: Workspace, idea: str, reason: str, commit: bool = True) -> VersionState:
    """Close an idea that cannot be built or tested; it stays in the record as failed."""

    journal = Journal(ws.journal_path)
    ideas = journal.ideas()
    if idea not in ideas:
        raise Refused(f"no idea {idea}")
    latest = ideas[idea].latest
    journal.append("abandoned", idea=idea, version=latest.version, reason=reason)
    if commit:
        paths = [ws.journal_path, write_index(ws)]
        subject = f"research({idea}-{latest.slug} v{latest.version}): abandoned - {reason}"
        vcs.commit(ws.root, paths, subject[:120])
    return journal.ideas()[idea].latest


def find_version(ws: Workspace, idea: str, version: int | None = None) -> VersionState:
    ideas = Journal(ws.journal_path).ideas()
    if idea not in ideas:
        raise Refused(f"no idea {idea}")
    versions = ideas[idea].versions
    number = version or max(versions)
    if number not in versions:
        raise Refused(f"{idea} has no v{number}")
    return versions[number]
