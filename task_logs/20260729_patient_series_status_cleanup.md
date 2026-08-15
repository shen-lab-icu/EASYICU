# Patient time-series status-summary cleanup

- Date: 2026-07-29
- Task: `PATIENT-SERIES-STATUS-CLEANUP`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: Patient Review time-series page below the feature catalog / trajectory gallery

## User-facing problem

The page repeated a developer-oriented explanation, four contract status cards, and three `ready` chips after the actual time-series workspace. These elements did not help the user inspect or act on patient data, duplicated state already visible above, and created a large low-value block before the audit and next-step actions.

## Implemented contract

- Removed the green “Two focused time-series views / 两种时间序列视图” note.
- Removed the four trajectory-review contract status cards.
- Removed the three review-mode `ready` chips.
- Kept the feature catalog / trajectory gallery workspace unchanged.
- Kept the collapsed exact-value audit entry.
- Kept the bottom “Data reviewed — what next? / 数据已审阅 —— 下一步？” actions.
- Backend `trajectory_review` data remains intact for other consumers; only the redundant Patient time-series rendering was removed.
- No CSS or backend owner changed. The patch shrinks the legacy `screens-viz.js` host instead of adding another override.

## Browser evidence

Official eICU Demo Patient Review at 1521×1354:

- The page moves directly from the Patient time-series workspace to the collapsed exact-value audit entry.
- The old explanatory note, `01 · Entity scope / 实体范围` contract cards, and three `ready` chips are absent from the DOM.
- `精确值审计矩阵`, `数据已审阅 —— 下一步？`, `队列统计`, and `进入研究项目` remain present.
- No horizontal overflow or clipping was visible in the audited viewport.

Screenshots:

- Before: `output/ui-qa/20260729_patient_status_block_audit/02-status-block-viewport.jpg`
- After: `output/ui-qa/20260729_patient_status_block_audit/03-after-status-block-removed.jpg`

## Verification

- `node --check src/easyicu/webserver/static/js/screens-viz.js`: passed.
- Patient frontend owner / CSS ownership suite: `7 passed`.
- Patient + static route suite: `78 passed, 1 known unrelated failure`.
- The only failure is the pre-existing worktree-name callback hint assertion (`easyicu-copilot-cockpit-lite` vs `EASYICU`).
- New regression extracts `ptSeries()` and prevents the removed note, contract cards, or mode chips from returning while requiring the exact-value audit entry.
- `git diff --check`: passed.
