# WEB-COPILOT-COCKPIT-LITE — progressive guidance

Date: 2026-07-29
Branch: `codex/web-copilot-cockpit-lite-20260729`

## Scope

Reduce the Guided Copilot setup burden without changing its backend contracts, scientific review gate, project-memory model, or expert direct-entry surfaces.

## Implemented

- Replaced the first screen's simultaneous setup summary with one primary decision: choose a research goal.
- Removed the duplicate goal suggestion row and delayed project-folder setup until after goal selection.
- Added four explicit presentation stages:
  - `start`: goal selection only;
  - `project`: project-memory choice only;
  - `configure`: the current unresolved decision plus already-saved context;
  - `review`: the complete study brief, including execution-bound and context-only fields.
- Hid the inactive left workflow rail on the start screen and kept Projects as a secondary shortcut.
- Preserved the fail-closed scientific review renderer and all existing execution handoffs.

## Ownership

- `screens-guided-progressive-shell.js` owns start-screen goal cards and stage projection.
- `screens-guided-study-workspace.js` owns the staged study-brief view model.
- `guided-progressive-shell.css` owns start-stage shell layout.
- `guided-study-workspace.css` owns shared setup/workspace presentation.
- `guided-study-workspace-review.css` owns review and blocked-result presentation.
- No route-specific code was added to broad `app.js`, `app.css`, `tweaks.js`, or `tweaks.css`.

## Verification

- Node syntax checks: passed for all changed Guided JavaScript owners.
- Pure JavaScript contracts:
  - progressive shell: 12 cases passed;
  - study workspace: 27 cases passed.
- Focused Python tests: 5 passed.
- Web static/UX suite: 80 passed, 1 unrelated existing worktree-name assertion failed. The failing callback provenance test derives `project_ref.hint` from the checkout directory and therefore reads `easyicu-copilot-cockpit-lite` instead of the main-checkout name `EASYICU`.
- CSS ownership scan: no foreign Patient, Cohort, Cross-DB, Extraction, Settings, Dictionary, States, or Tutorial route markers.
- CSS structural scan: balanced braces/comments; no new `!important` or `:has(...)`.
- `git diff --check`: passed.

## Browser QA

Browser: local FastAPI preview at `http://localhost:8876/#guided`
Viewport: 1280 × 720 desktop

- Start state: four goal cards, zero duplicate suggestion chips, left rail hidden, and one right-rail decision.
- Project-memory state: two relevant choices, selected goal retained, left rail restored, and one right-rail decision.
- Both states: `document.scrollWidth === window.innerWidth === 1280` and `document.scrollHeight === window.innerHeight === 720`.
- Console: no errors or warnings.

Evidence:

- `output/ui-qa/20260729_copilot_progressive_guidance/01-start.png`
- `output/ui-qa/20260729_copilot_progressive_guidance/02-project-memory.png`
- `output/ui-qa/20260729_copilot_progressive_guidance/03-before-after.png`
- `output/ui-qa/20260729_copilot_progressive_guidance/04-right-rail-before-after.png`
- `design-qa.md`

## Boundaries

- No backend API, persistence schema, execution module, or scientific-result logic changed.
- The branch has not been merged or pushed.
