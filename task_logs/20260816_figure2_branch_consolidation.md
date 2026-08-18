# Figure 2 branch and worktree consolidation

Date: 2026-08-16  
Task: `FIG2-DEV9-HELDOUT27`

## Outcome

Figure 2 development now has one integration branch and one development
worktree:

- branch: `integration/figure2-e1-h3-20260816`
- worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-figure2-integration-20260816`
- integration merge commits: `48cdc95` and `8c41857`

The repository root remains the clean `main` worktree. This consolidation did
not merge the development branch into `main`, start a Provider canary, or grant
publication authority.

## Preserved inputs

The integration branch contains all committed history from:

| Source | Preserved tip |
|---|---|
| `main` | `c9098fe` |
| former `codex/figure2-e1-h3-20260816` | `2aab71f` |
| former `feat/figure2-dev9-heldout27-20260815` | `8e2172d` |
| former `feat/e1-h3-20260815` | `40bd86b` |
| former detached Codex auth worktree | `f8e16d0` |

Ancestor checks against the pushed integration tip all passed before source
branches or worktrees were removed.

## Conflict adjudication

- Kept the browser-verified session-v2 implementation in which one immutable
  Codex-account/API model connection powers both Copilot conversation and the
  governed Research Agent handoff. The parallel analysis-only selector was not
  allowed to replace that newer contract.
- Integrated the Dev9 provider owner split: concrete OpenAI-compatible and
  Anthropic transports now live in `providers/clients.py`, while factory,
  selection, and trust responsibilities remain separate.
- Combined both SOFA-2 test lines instead of selecting one file wholesale:
  Arrow/nullable input and high P/F regressions from `main` coexist with the
  Dev9 ordinal-aggregate guard.
- Preserved the exact Pi runtime diagnostic code as demoted support metadata in
  the unified provider panel.
- Removed one duplicate fake-service method introduced by automatic merge.

## Verification

- Planner patch before consolidation: `45 passed`; Ruff and diff check passed.
- Dev9 provider split before consolidation: `542 passed, 2 skipped`; after
  mechanical import cleanup, the direct provider matrix was `174 passed` and
  Ruff passed.
- Integrated provider and Progressive Planner matrix: `219 passed`.
- Integrated data, export, full6, SOFA-2, and time-axis matrix:
  `224 passed, 4 skipped`.
- Integrated Web/Codex matrix: first pass `316 passed, 1 failed`; the sole
  failure identified the missing demoted diagnostic-code title. After repair,
  the owning static/UI matrix was `80 passed`; Node syntax and diff checks
  passed.

These are focused integration checks, not a full exact-head CI receipt and not
an E1 11-stage result.

## Cleanup ledger

- Removed four obsolete clean worktrees after verifying that their committed
  tips were reachable from the pushed integration branch.
- Deleted the corresponding local and remote source branches.
- Stopped and removed the stale read-only runner smoke container
  `d5aa3f6f4f8c...`, which had been stuck for about 30 hours and held the Dev9
  worktree as its current directory.
- The unrelated, unmerged clinical-bounds commit `ee33539` was not injected
  into Figure 2. It was preserved as the pushed tag
  `archive/concept-clinical-bounds-ee33539` before its local branch was removed.

## Remaining authority boundary

E1 is still development-only and has not completed its 11-stage path. The next
run must use an exact integration HEAD and matching runner image. E2 through H3,
Qualification12, and Held-out27 remain blocked behind E1 completion and the
existing scientific/freeze gates.
