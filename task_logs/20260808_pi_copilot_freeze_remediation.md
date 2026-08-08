# Pi Copilot V1 freeze remediation

- Date: 2026-08-08
- Module: web
- Task: `PI-COPILOT-V1-FREEZE-REMEDIATION`
- Branch: `feat/pi-copilot-shell`
- Review input: `/Users/haibo/.codex/attachments/15a684da-a93b-495a-aba3-626604ce5b70/pasted-text.txt`
- Visual reference: `/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/codex-clipboard-ff0de24f-c147-4b48-a7e3-66602616bbb7.png`

## Outcome

The two remaining release-blocking boundaries from the review are closed. Pi control tools cannot race inside one upstream multi-tool batch, and an existing Guided research project now enters Pi through an explicit Host-owned migration into its project-owned StudyContext. The review's three adjacent P2 gaps—pure GET behavior, installed-runtime integrity, and provider-call restoration after compaction—are also closed.

The Guided Pi conversation was revised against the supplied Codex activity screenshot. A running turn now exposes one current semantic action. A completed turn becomes one collapsed activity summary with duration; expanding it reveals bounded EasyICU receipts without exposing model chain-of-thought.

## Release-boundary changes

### 1. Sequential authority mutations

- `easyicu_update_study_context`, `easyicu_run`, `easyicu_cancel`, and `easyicu_request_replan` declare Pi's upstream `executionMode: "sequential"`.
- A regression runs the real upstream Agent loop with global parallel tool execution and one assistant message containing `update_study_context + run`.
- Both one-turn grants are present, but only the first mutation succeeds. The second call observes `pi_session_authority_stale`.

### 2. Existing project migration

- `guided_sessions.py` publishes the typed, read-only `GuidedProjectStudySetup` contract.
- It compiles exact saved coordinates from Guided session/draft metadata: purpose, data source/export, cohort, modules, outcome, time window, comparator, export format, analysis goal, and confirmations.
- Pi project initialization is an explicit POST operation, not a side effect of listing sessions.
- Complete saved metadata migrates into a new project-owned StudyContext. Incomplete metadata returns `pi_project_initialization_required`; the user must explicitly activate Pi before an empty StudyContext is created and the missing fields are collected in the same conversation.
- The persisted binding includes schema `easyicu.project-studycontext-migration/1`, a canonical source digest, migrated field names, and status.
- Existing legacy Pi sessions are reconciled only when they identify one unambiguous StudyContext; disagreement fails closed.

### 3. Pure GET boundary

- Session GET/list no longer creates a ProjectAuthority mapping, creates a StudyContext, or rewrites a legacy session.
- Both unbound and legacy-bound negative cases are locked by tests.

### 4. Installed runtime integrity

- The installation manifest now records the actual Node version.
- All installed Pi package manifests and executable `.js`, `.mjs`, and `.cjs` files are hashed after installation and re-hashed at startup.
- A mutation to installed `dist/index.js` invalidates the runtime even when package versions remain unchanged.
- The real temporary install covered 674 executable files under Node 24.11.0.

### 5. Provider-call restoration

- Every authorized provider call appends a stable Pi SessionManager custom receipt using schema `easyicu.shell-budget/1`.
- Session reopen restores the latest receipt instead of estimating solely from assistant-message count.
- A compaction-containing transcript restores the receipted call count and preserves the session call ceiling.

## Activity UI comparison

The UI owner remains `static/css/guided-pi.css` and `static/js/screens-guided-pi.js`; no feature rules were added to a catch-all CSS/JS bucket.

- During execution: one inline current-action row and running pip.
- After execution: one semantic summary, separate duration, collapsed by default.
- On disclosure: individual tool action, bounded summary, stable code, owner, and lifecycle result.
- Persisted transcript: the same grouped activity model reconstructs after reopen.
- Assistant copy: escape-first inline bold/code rendering.

Evidence:

- `task_logs/browser_audit_20260808_pi_freeze_round/04-live-tool-state.jpg`
- `task_logs/browser_audit_20260808_pi_freeze_round/06-expanded-final.jpg`
- `task_logs/browser_audit_20260808_pi_freeze_round/07-final-markdown.jpg`
- `task_logs/browser_audit_20260808_pi_freeze_round/08-reference-comparison.jpg`
- `design-qa.md`

## Verification

- Focused Pi/Web contracts: `55 passed`.
- Method-kernel collection guard: `22 passed, 1 skipped`.
- Python lint for changed owners: passed.
- Node syntax checks: passed.
- Real Pi runtime install: `239` packages audited, `0` vulnerabilities; runtime integrity smoke passed with `674` executable hashes.
- Wheel and sdist build: passed; required Pi entrypoints, lockfile, event projection, and budget owner are present.
- Browser: three real `gpt-5.6-luna` read-only turns completed; no body/transcript/composer horizontal overflow at `1572 x 1354`.
- Repository-wide current-worktree suite: `12,479 passed, 70 skipped, 155 failed` in 52m22s. The failures reproduce on the untouched starting commit `1cc6b6d`; representative failures include the frozen Figure 2 scorer-tree digest mismatch plus downstream Canonical9/preflight assertions that never reach their intended state. They are not regressions introduced by this Pi patch.

One unrelated full-suite collection defect was exposed before the product suite could start: an empty parameter declaration passed pytest's `NotSetType` into a display-only `ids` lambda. The test-harness ID callback now uses `getattr`; the scientific reachability contract is unchanged.

The repository-wide gate remains blocked outside the Web/Pi owner. The frozen Figure 2 paper scorer manifest records tree digest `66f6266b072217dba9a3ada96cdab845b7a09d753d307671a943d6f42976d543`, while the current scorer tree computes `c1702070cac633e39460cc07dd1afa9eed7e9a2f8b7a69f6a2aeef9d63f5e434`. This task deliberately does not reseal that scientific authority: doing so requires the Canonical9 benchmark owner to review and authorize the changed scorer tree. A detached worktree at baseline `1cc6b6d` reproduced all four sampled red tests unchanged, including one direct digest mismatch and three downstream/preflight failures.

## Deliberately retained non-blocking boundaries

- Provider DNS validation still has theoretical DNS TOCTOU. No second-resolution pseudo-fix was added; a future answer requires a transport/pinning design.
- Real Anthropic and Google provider canaries require external credentials and were not claimed.
- A USD hard stop is unavailable without an explicit pricing profile; token and provider-call ceilings remain hard stops.
- The Node sidecar runs as the local OS user in a private working directory; this is not represented as an operating-system filesystem sandbox.
- Local current-worktree validation does not equal GitHub current-SHA CI. No push or remote workflow was performed in this task.
