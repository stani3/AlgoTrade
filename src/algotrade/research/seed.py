"""``research seed``: import what was tested before the journal existed (run once).

* The book's breakout + RSI + ATR bracket strategy, grid-searched on 4h (with and without the
  kill switch) and 1d, becomes idea i001 with a failed feasibility verdict, so it can never be
  quietly retested.
* The catalogue scans (``scripts.scan``) become screening trials: counted for the deflated Sharpe
  ratio and fingerprinted against near-copies, but not gated, so a catalogue rule can still be
  registered as an idea and tested properly.
* Buy & hold is fingerprinted as the baseline every idea must differ from.

All of it ran on the full history, holdout included; the records say so.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd

from algotrade.backtest.costs import EXCHANGE_COSTS
from algotrade.strategies import RULES, BracketStrategy, VolTarget, from_spec, to_spec

from . import fingerprint, results, vcs
from .cards import Card, card_hash
from .criteria import Criteria, at_least
from .dedup import spec_hash
from .index import write_index
from .journal import Journal, TrialLedger, now
from .registry import Refused
from .split import dev_universe
from .workspace import Workspace

PRE_JOURNAL = "full history 2019-2026, holdout included (pre-journal)"
SCAN_FILES = {  # file stem -> (timeframe, vol target used by the scan)
    "scan_1d": ("1d", None),
    "scan_1d_vt": ("1d", 0.25),
    "scan_4h_vt": ("4h", 0.25),
}
BOOK = {"type": "breakout_bracket"}

BOOK_BODY = """
## Hypothesis
A close at the highest (lowest) close of the last 48 four-hour bars, confirmed by RSI(30) above
(below) 50, starts a move that is worth more than a fixed ATR bracket risks.

## Why it should work
Breakouts with momentum confirmation catch the start of trends; a stop two ATRs away keeps the
loss small while a target four ATRs away lets winners pay for several losers.

## Rules
Enter at the next bar's open. Stop `stop_atr` x ATR and target `target_atr` x ATR from the fill,
ATR read on the signal bar. Wait 5 bars after a loss and 20 after a win; stop trading for good at
a 50% drawdown from the equity peak.

## Falsified if
The median Sharpe ratio across the ten-symbol universe stays near zero across the stop/target
grid, or the kill switch trips on most symbols.

## Outcome (recorded by `research seed`)
Tested before the research journal existed, on the full history: the best grid cell reached a
median Sharpe of about 0.25 (4h) and the kill switch fired on most symbols. Failed.
"""


def _book_card(version: int, timeframe: str, grid: dict) -> Card:
    front = {
        "id": "i001",
        "version": version,
        "registered": now(),
        "title": "Book breakout + RSI + ATR bracket",
        "source": "User's book example strategy (breakout, RSI filter, ATR stop and target)",
        "taxonomy": {"family": "breakout", "inputs": ["price"], "horizon": "days"},
        "timeframe": timeframe,
        "spec": to_spec(from_spec(BOOK)),
        "optimise": grid,
        "expected_trades_per_year": 10,
        "historical": True,
    }
    if version > 1:
        front.update(parent=1, revision_reason="same rules on daily bars (pre-journal)")
    return Card(front=front, body=textwrap.dedent(BOOK_BODY))


def _grid_values(frame: pd.DataFrame, keys: list[str]) -> dict:
    return {key: sorted(frame[key].unique().tolist()) for key in keys}


def _seed_book(
    ws: Workspace, criteria: Criteria, journal: Journal, ledger: TrialLedger
) -> list[Path]:
    reports = ws.root / "reports"
    sources = {
        1: ("4h", [reports / "grid_breakout_bracket_4h.csv", reports / "no_kill/grid_breakout_bracket_4h.csv"]),
        2: ("1d", [reports / "grid_breakout_bracket_1d.csv"]),
    }  # fmt: skip
    paths: list[Path] = []
    slug = "book-breakout-rsi-atr-bracket"
    for version, (timeframe, files) in sources.items():
        frames = []
        for path in files:
            if not path.exists():
                raise Refused(f"cannot seed: {path} is missing (re-run scripts.grid first)")
            frame = pd.read_csv(path)
            frame["kill_drawdown"] = 1.0 if "no_kill" in path.parts else 0.5
            frames.append(frame)
        grid_frame = pd.concat(frames, ignore_index=True)
        keys = ["stop_atr", "target_atr", "atr_length", "kill_drawdown"]
        grid = _grid_values(grid_frame, keys)
        if len(grid["kill_drawdown"]) == 1:
            del grid["kill_drawdown"]
        card = _book_card(version, timeframe, grid)
        folder = ws.version_dir("i001", slug, version)
        folder.mkdir(parents=True, exist_ok=True)
        card_path = folder / "idea.md"
        card_path.write_text(card.render(), encoding="utf-8", newline="\n")
        journal.append(
            "idea_registered",
            idea="i001",
            version=version,
            slug=slug,
            title=card.title,
            timeframe=timeframe,
            spec=card.spec,
            grid=grid,
            taxonomy=card.taxonomy,
            card_hash=card_hash(card_path),
            card_path=ws.relative(card_path),
            parent=card.parent,
            reason=card.front.get("revision_reason"),
            historical=True,
        )
        rows = []
        for record in grid_frame.to_dict("records"):
            spec = {**card.spec, **{k: record[k] for k in keys}}
            rows.append(
                {
                    "idea": "i001",
                    "version": version,
                    "stage": "feasibility",
                    "kind": "grid",
                    "label": f"breakout_bracket grid {timeframe}",
                    "spec_hash": spec_hash(spec),
                    "timeframe": timeframe,
                    "metric": record["median_sharpe"],
                    "window": PRE_JOURNAL,
                    "spec": spec,
                }
            )
        ledger.add(rows)
        best = float(grid_frame["median_sharpe"].max())
        profitable = float((grid_frame["median_sharpe"] > 0).mean())
        killed = float(grid_frame.loc[grid_frame["kill_drawdown"] < 1, "killed"].mean())
        state = journal.ideas()["i001"].versions[version]
        result = results.StageResult(
            "i001",
            version,
            "feasibility",
            [
                at_least(
                    "best median Sharpe in the grid",
                    best,
                    criteria.get("feasibility.core_min_median_sharpe"),
                ),
            ],
            metrics={
                "combinations": len(grid_frame),
                "share of combinations with median Sharpe > 0": profitable,
                "share of symbols where the kill switch fired": killed,
            },
            trials=len(grid_frame),
            notes=[f"Pre-journal grid search on the {PRE_JOURNAL}."],
            provenance=results.provenance(ws, criteria, state, spec_hash(card.spec)),
        )
        results.record(ws, state, result, commit=False)
        paths.append(folder)
    return paths


def _scan_specs(path: Path, vol_target: float | None, spec_dir: Path) -> dict[str, dict]:
    frame = pd.read_csv(path, index_col=0)
    specs = {}
    rules = {cls.name: cls for cls in RULES}
    for label in sorted(frame["strategy"].unique()):
        if label.startswith("spec:"):
            strategy = from_spec(spec_dir / f"{label[5:]}.json")
        else:
            strategy = rules[label]()
        if vol_target and not isinstance(strategy, VolTarget | BracketStrategy):
            strategy = VolTarget(strategy, annual_vol=vol_target)
        median = float(frame.loc[frame["strategy"] == label, "sharpe"].median())
        specs[label] = {"spec": to_spec(strategy), "median_sharpe": median}
    return specs


def seed(ws: Workspace, criteria: Criteria, commit: bool = True) -> dict:
    journal = Journal(ws.journal_path)
    if journal.seeded():
        raise Refused("the journal is already seeded")
    ledger = TrialLedger(ws.trials_path)
    store = fingerprint.FingerprintStore(ws.fingerprints_dir)
    costs = EXCHANGE_COSTS[criteria.get("data.exchange")]
    symbols = criteria.get("data.fingerprint_symbols")
    universes = {tf: dev_universe(ws, criteria, tf, symbols) for tf in ("4h", "1d")}
    paths = _seed_book(ws, criteria, journal, ledger)

    def save(spec: dict, timeframe: str, meta: dict) -> None:
        key = store.key(spec_hash(spec), timeframe)
        if not store.exists(key):
            frame = fingerprint.compute(spec, universes[timeframe], costs)
            base = {"spec": spec, "spec_hash": spec_hash(spec), "timeframe": timeframe}
            store.save(key, frame, {**base, **meta, "symbols": symbols})

    book = journal.ideas()["i001"]
    for version in book.versions.values():
        save(version.spec, version.timeframe, {"source": "idea", "idea": "i001",
             "version": version.version, "label": f"i001 v{version.version} {version.title}"})  # fmt: skip

    screened = 0
    reports = ws.root / "reports"
    for stem, (timeframe, vol_target) in SCAN_FILES.items():
        path = reports / f"{stem}.csv"
        if not path.exists():
            continue
        rows = []
        for label, item in _scan_specs(path, vol_target, ws.root / "specs").items():
            sizing = f", vol target {vol_target:.0%}" if vol_target else ""
            name = f"{label} ({timeframe}{sizing})"
            rows.append(
                {
                    "idea": "",
                    "version": 0,
                    "stage": "screen",
                    "kind": "screen",
                    "label": name,
                    "spec_hash": spec_hash(item["spec"]),
                    "timeframe": timeframe,
                    "metric": item["median_sharpe"],
                    "window": PRE_JOURNAL,
                    "spec": item["spec"],
                }
            )
            save(item["spec"], timeframe, {"source": "screen", "label": f"screened {name}"})
        screened += ledger.add(rows)

    for timeframe in ("4h", "1d"):
        save({"type": "buy_and_hold"}, timeframe, {"source": "baseline",
             "label": f"buy & hold ({timeframe})"})  # fmt: skip

    summary = {
        "book_trials": int((ledger.frame()["idea"] == "i001").sum()),
        "screened": screened,
        "fingerprints": len(store.keys()),
        "trials": ledger.count(),
    }
    journal.append("seeded", **summary, window=PRE_JOURNAL)
    paths += [ws.journal_path, ws.trials_path, ws.fingerprints_dir, write_index(ws)]
    if commit:
        vcs.commit(
            ws.root,
            paths,
            "research: seed the journal with pre-journal experiments",
            "Imports the book breakout bracket grids as failed idea i001, the catalogue scans as "
            "screening trials, and fingerprints for those plus buy & hold.",
        )
    return summary
