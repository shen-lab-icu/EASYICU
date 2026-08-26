# 2026-08-26 EasyICU branch consolidation

## Outcome

- Consolidated the active Literature/Planner, Writer, Copilot data-preview, H1 reader, scorer-audit, and current figure/product lineages into `codex/easyicu-group-demo-20260826`.
- Preserved the stricter current Writer and article-display owners while integrating H1 current-ledger resume, typed owner manuscript projection, survival reader repair, citation authority, and publication-figure fail-closed behavior.
- Recorded the committed septic-shock candidate with an `ours` merge so its history is reachable without enabling a phenotype that its still-dirty audit worktree is actively quarantining.
- Reduced local branch refs from 21 to 4 and registered worktrees from 18 to 8. The remaining branch refs are the consolidated demo branch, `main`, the dirty desktop lane, and the dirty septic-shock audit lane.
- Removed ten verified-clean, fully merged worktrees. No dirty worktree was removed or modified.

## Recoverability

- `archive/final-ci-full-wip-20260823T132020Z` -> `570adfde73b0eefbc7f64ac30883eeae82bd0552`
- `archive/final-ci-web-wip-20260823T132020Z` -> `3ee30a2e90a475d81157c3352c51e303c93ed621`
- `archive/unified-before-desktop-dirty-audit-20260824` -> `ffe3a070b21fd57dd10b3aa9a8fbe7f7ebb5778e`
- Existing pre-consolidation restore tag: `backup/group-demo-before-full-consolidation-20260826` -> `54ab610`.

## Validation

- H1 Writer/Reader focused matrix: `159 passed, 1 deselected`.
- Scorer focused matrix: `23 passed`.
- Clinical boundary matrix after the history-only septic-shock merge: `84 passed`.
- Combined Copilot, data workbench, Planner/literature, Writer/Reader, article-display, publication-figure and clinical matrix: `820 passed, 1 deselected`.
- Ruff, Python diff check, and syntax checks for every `screens-guided-pi*.js` owner passed.
- One merge-interaction defect was reproduced and fixed: explicit-only phenotype selection now fails before a stale exact-covariate binding can mask the user decision gate (`3949a38`).

## Scope boundary

- No full exact-head CI or release package rebuild was run.
- Local `main` (46 changed paths), desktop (28 changed paths), and septic-shock audit (16 changed paths) were preserved untouched.
- This checkpoint is suitable for the requested live Web demonstration after a focused browser smoke, but is not release-ready or paper-authorized.
