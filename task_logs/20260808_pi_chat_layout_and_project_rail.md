# PI-COPILOT-CHAT-LAYOUT — bottom composer and compact project rail

- Date: 2026-08-08
- Branch: `feat/pi-copilot-shell`
- Implementation commit: `374d5f1`
- Owner: Web / Guided Pi Copilot

## User-visible problem

The empty Pi conversation placed its composer near the top of the page and left most of the viewport blank underneath. Text was too small for the primary task, while the project rail exposed four competing metadata lines per project, including a local filesystem path and a permanently visible remove action.

## Root cause

`guided-pi.css` used a five-row grid for a DOM whose stale-authority and error rows are conditional. When both conditional rows were absent, auto-placement assigned `.gpi-compose` to the `minmax(0, 1fr)` row. At the inspected 1933×1354 desktop viewport, the composer therefore grew to 991.2 px and the transcript received only 249.8 px.

This was a structural layout bug, not an empty-state spacing choice.

## Reference pattern

The redesign follows the mature shell pattern used by `pi-gui`: a full-height workspace, independently scrolling transcript, bottom composer, approximately 292 px navigation rail, concise thread rows, and row actions revealed on hover/focus. EasyICU retains its own research-project semantics, authority boundaries, backend, tokens, and right-side study workspace.

- `pi-gui` reference: <https://github.com/minghinmatthewlam/pi-gui>
- Relevant upstream concepts: `apps/desktop/src/styles/main.css`, `styles/sidebar.css`, and `styles/timeline.css`.

## Implementation

1. Replaced the conditional five-row Pi grid with an explicit column flex layout.
   - Header, stale/error notices, and composer are fixed-height rows.
   - `.gpi-log` is the only flexible, independently scrolling region.
   - `.gd-pi-shell` and the active conversation owner now contain overflow explicitly.
2. Docked the composer at the bottom of the live viewport.
   - Composer content shares the transcript's 820 px readable measure.
   - Textarea and message body typography increased to 15 px.
   - Empty-state supporting text increased to 13.5 px; tool and activity receipts were increased proportionally.
3. Simplified the project rail.
   - Rail width increased from 232 px to 292 px.
   - Default project rows now show title, concise localized state, and time only.
   - Local filesystem paths were removed from the normal rail view.
   - Remove actions are hidden until hover or keyboard focus.
   - The redundant `Workspace` heading was removed.
4. Restored ownership boundaries.
   - Project-row rendering moved from the 6,000-line `screens-guided.js` shell into `screens-guided-projects.js`.
   - Rail CSS moved out of the 2,300-line `guided.css` file into the new `guided-projects.css` owner.
   - Presence and absence regressions prevent Guided Pi, extraction, pipeline, Patient, or Cohort selectors from leaking into the project owner.

## Verification

- Focused Pi/Web/Guided suite: `154 passed`.
- Node syntax checks: `screens-guided.js`, `screens-guided-projects.js`, and `screens-guided-pi.js` passed.
- CSS comment/brace scan: `guided.css`, `guided-projects.css`, and `guided-pi.css` balanced.
- `git diff --check`: passed.
- Live desktop browser measurement after the fix:
  - viewport/document: 1933×1354, no document scroll growth;
  - project rail: 292 px;
  - conversation log: 1045.3 px and independently scrollable;
  - composer: 193.3 px, bottom exactly aligned with the viewport;
  - textarea/message font: 15 px;
  - document, main shell, project rail, and Pi log horizontal overflow: all false.
- Interaction check: the bottom textbox accepted and focused typed text; the temporary text was removed without sending a model request.
- No configure, run, cancel, patient-data, or external-model action was performed during this UI QA.

## Visual evidence

- `output/ui_audit/pi_chat_layout_20260808/01-before.png`
- `output/ui_audit/pi_chat_layout_20260808/02-after.png`
- `output/ui_audit/pi_chat_layout_20260808/03-before-after.png`

These files are regenerable UI QA evidence, not scientific source data.

## Remaining scope

- The full Pi agent timeline and safe tool receipts remain implemented; this patch changes layout and readability only.
- Configure-grant and authority stale/rebind canaries remain the next functional Pi task.
- The right-side study workspace was deliberately preserved because it owns conversational study-slot progress; this UI patch did not collapse or duplicate its backend responsibility.
