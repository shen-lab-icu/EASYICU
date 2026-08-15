# WEB-COPILOT-COCKPIT-LITE completion evidence

> Date: 2026-07-29
> Module: web
> Branch: `codex/web-copilot-cockpit-lite-20260729`
> Isolated worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-copilot-cockpit-lite`
> Base: `fix/external-review-20260724-p0-p1@f0fdbce`

## Outcome

The low-change Guided Copilot plan is implemented as two independently reviewable
frontend commits:

1. `a8b946f feat(web): add Guided study brief cockpit`
2. `2c4c08f feat(web): add fail-closed Guided scientific review`

The existing three-column Guided shell, conversation flow, bounded slots,
StudyContext, Agent handoff, job payloads, and review API remain the source of
truth. No backend route, API schema, scheduler, or parallel state store was added.

## Patch 1 — persistent research cockpit

- The right rail is now a persistent Study Brief that distinguishes:
  - inputs actually applied when a run starts: source and research question;
  - research context saved for interpretation: cohort, modules, outcome, time
    window, comparator, export, and analysis goal;
  - missing and conflicting values, without converting defaults into confirmed
    user choices.
- The project rail exposes the active title, phase, attention state, and next
  decision; the local project path remains secondary tooltip metadata.
- At 1024 px and 1180 px, the existing shell hides the project rail and preserves
  the 322 px Study Brief; project/phase context remains visible in the Guided
  header.
- The rendering/read-model contract lives in the dedicated
  `screens-guided-study-workspace.js` owner. StudyContext exposes a metadata-only
  view instead of leaking mutable private state into Guided.

## Patch 2 — fail-closed scientific review

- A dedicated `screens-guided-agent-review.js` owner projects the existing
  `/api/agent-runs/review` payload into one of four phases:
  `executing`, `blocked`, `scientific_review`, or `planning`.
- `scientific_review` is reachable only when gate shape, analysis-only envelope,
  readiness, artifact hashes, evidence ledger, privacy scan, StudyContext receipt,
  terminal run identity/revision, denominator, and sign-off integrity agree.
- Missing, malformed, stale, contradictory, or tampered values fail closed to
  `blocked`; `null`, empty strings, and booleans are not coerced into valid zeroes.
- The original terminal run result remains the identity authority; a review
  response cannot replace it and launder a different run into a ready state.
- Guided always reports `reportable=false` and `draft_unlocked=false`. Human
  sign-off and manuscript unlock remain owned by Agent Projects.

## Ownership and static checks

- Route owners:
  - `static/js/screens-guided-study-workspace.js`
  - `static/js/screens-guided-agent-review.js`
  - `static/css/guided-study-workspace.css`
  - `static/css/guided-study-workspace-legacy.css`
- No feature code was added to `app.js`, `tweaks.js`, `tweaks.css`, or another
  catch-all owner.
- Guided workspace CSS: balanced braces/comments; no `!important`, no `:has(...)`,
  and no Patient, Cohort, Cross-DB, Agent-route, or unrelated route selectors.
- Review/workspace read-model owners have no direct API transport, DOM,
  `localStorage`, or private StudyContext access.
- `git diff --check` passed.

## Automated verification

- Node contract suites:
  - `guided_agent_review.test.js`: 30 cases passed.
  - `guided_gate_state.test.js`: 12 cases passed.
  - `guided_study_workspace.test.js`: 17 cases passed.
- JavaScript syntax checks passed for all changed/new owners.
- Nine targeted Python Guided/static route and UX tests passed.
- The broader affected Python selection collected 94 tests: 91 passed and three
  unrelated baseline assertions failed:
  1. a test assumes the checkout directory is literally named `EASYICU`, which is
     false for this isolated worktree;
  2. two stale Cross-DB handoff assertions still expect code in `screens-viz.js`.
  None of the failing files or assertions were changed by these two patches.

## Desktop browser QA

The real app at `/#guided` was checked at the required desktop/laptop viewports.
All four had `document.clientWidth == document.scrollWidth` and no visible
overflow or clipping.

| Viewport | Project rail | Study Brief | Body behavior |
|---|---|---|---|
| 1024×768 | hidden | 322 px | internal vertical scroll |
| 1180×800 | hidden | 322 px | internal vertical scroll |
| 1280×720 | 232 px | 322 px | internal vertical scroll |
| 1440×900 | 232 px | 322 px | no forced body scroll |

Production-module harness checks also rendered:

- ready scientific review with denominator `94,458`, evidence actions enabled,
  `not reportable`, and draft locked;
- blocked scientific review with recovery actions to Agent Projects and research
  question revision.

Screenshots are under
`output/ui-qa/20260729_cockpit_lite/`, including:

- `patch2-planning-{1024x768,1180x800,1280x720,1440x900}.png`
- `07-scientific-review-1024x768.png`
- `08-scientific-review-blocked-1024x768.png`

## Deliberately deferred

Ask/Plan/Run modes, typed RunPlan, persistent DomainDiff/branching, and
cross-refresh job/run restoration remain outside this low-change slice. The
current Agent Projects handoff is valid in the active session, but this work does
not claim a new persisted deep link that restores the exact review run after a
full page refresh.
