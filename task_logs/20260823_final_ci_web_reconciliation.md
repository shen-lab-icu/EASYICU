# Final CI Web WIP reconciliation (2026-08-23)

## Scope and baseline

- Historical local checkout: `main@1e5cda1`, dirty with concurrent Research Agent and Web work.
- Current remote baseline: `origin/main@8115f933f54b260c392ea9cac75828294ea75d9c`.
- Reconciliation branch: `codex/final-ci-web-reconcile-20260823`, created from the remote baseline in a separate worktree.
- No Provider call, research run, patient-data access, push, or merge was performed.

The original main worktree was not stashed, reset, or checked out because an existing Web process was still serving that worktree and a separate desktop-app worktree was building. Before reconciliation, two local backup refs were created without changing the real index or worktree:

- `backup/final-ci-full-wip-20260823T132020Z` -> `570adfde73b0eefbc7f64ac30883eeae82bd0552`
- `backup/final-ci-web-wip-20260823T132020Z` -> `3ee30a2e90a475d81157c3352c51e303c93ed621`

## Merge result

The Web-only snapshot contained 20 paths. Nineteen merged automatically onto `8115f93`. The only content conflict was `static/js/screens-guided-pi.js`:

- remote `8115f93` had moved project preparation into `screens-guided-pi-project.js`;
- the older Web WIP had edited the former inline `prepareProject()` body to project legacy titles through the product-label owner.

Resolution preserved the remote owner split: `screens-guided-pi.js` remains a thin call to `window.EU_GUIDED_PI_PROJECT.prepare(...)`. The Web WIP still projects the persisted title when `bindProject()` constructs `state.project`, so the project owner receives the user-visible title without duplicating initialization logic or growing the main Copilot module again.

The reconciliation also preserved the remote final-CI fixes outside the WIP, including the live v3 scorer digest, the 13-item Outcome catalog count, the project-preparation owner asset, helper ownership fixes, resource baseline, golden bundle, and optional H1 dependency boundary.

## Verification

- JavaScript contracts: `24/24` passed.
- Web/Copilot/static ownership and route matrix: `281 passed, 5 warnings`.
- Original final-CI high-risk set (repository hygiene, resource baseline, golden bundle, H1 runtime contract, full Figure 2 evaluator): `285 passed, 8 warnings`.
- Research Agent architecture gates: all `5/5` green; size/budget subset `141 passed`.
- Progress lint: all six module pages passed; one pre-existing warning remains for `论文图件/CURRENT.md` at 42,082 bytes.
- `git diff --check`: clean.

Isolated browser QA used `EASYICU_HOME=/tmp/easyicu-final-ci-browser-state-20260823` and port `8519`, then stopped both the browser session and server:

- viewport `1280x720`, English and Chinese Guided Copilot;
- document horizontal overflow `0` in both languages;
- console `0` errors / `0` warnings;
- `window.EU_PRODUCT_LABELS` and `window.EU_GUIDED_PI_PROJECT.prepare` both present;
- legacy `Pi Copilot` projected to a discriminating fallback;
- visible page text contained no `Pi Copilot`;
- static `product-labels.js` returned content-derived SHA-256 ETag and `Cache-Control: no-cache`.

This is a compatibility/reconciliation checkpoint only. It does not establish full exact-head CI for the new Web commit, desktop UAT completion, Provider readiness, E1 execution, or paper authority.
