---
name: strategy-pipeline
description: Run the AlgoTrade strategy factory loop within its budget - new idea, build, feasibility (with at most one revision), validation, holdout and freeze - moving on to the next idea whenever one fails, and stop at the first strategy that passes with the approval package for the user. Use when the user asks to find, develop or research new strategies in a batch ("run the pipeline", "try a few ideas").
---

# Strategy pipeline

The loop around the stage skills. Each stage runs in a **fresh subagent** that is told to read
and follow that stage's `SKILL.md`; the files under `research/` carry everything between stages,
so no stage depends on chat history.

## Setup

1. `python -m scripts.research status` and note the trial count and the ideas so far.
2. Budget: `budgets.ideas_per_run` in `research/criteria.yaml` (new ideas this run; revisions
   do not count as new ideas but use that idea's revision budget). The user may give a smaller
   budget.
3. Note the current commit (`git rev-parse HEAD`) to list this run's commits at the end.

## Per idea

1. **Idea** - subagent: follow `.claude/skills/strategy-ideas/SKILL.md`, register exactly one
   new idea, and report its id, title and hypothesis.
2. **Build** - subagent: follow `.claude/skills/strategy-build/SKILL.md` for that id. If the build
   is a duplicate (BUILD FAIL) or cannot be implemented (abandoned), the idea is done.
3. **Feasibility** - subagent: follow `.claude/skills/strategy-feasibility/SKILL.md`.
   - FAIL with no revision: the idea is done.
   - A revision was registered: run Build and Feasibility once more for the new version.
4. **Validation** - subagent: follow `.claude/skills/strategy-validate/SKILL.md` up to and
   including freeze. A failure at validation or holdout ends the idea.
5. Record a line in the run log: id, version, the stage it stopped at, and the main reason.

Every stage commits its own result locally; check `git status` is clean between stages.

## Stop

- A strategy was frozen: stop the loop and present its approval package yourself (the
  `strategy-validate` steps 4.1-4.4: reports sent and opened, decision summary, the question).
- Or the budget is used up.

## Final summary

- Each idea tried: id, title, how far it got, why it stopped (from `research/index.md`).
- Trial count before and after the run (the deflated Sharpe bar rose with it).
- The best candidate, if any, with links to its reports.
- `git log --oneline <start>..HEAD` - the commits this run made. Nothing is pushed; pushing is
  the user's call.

## Never

Pass `--retest` or `--force`, edit `research/criteria.yaml`, push, exceed the budget, run
stages out of order, or start incubation: that needs the user's explicit yes after they have
seen the approval package.
