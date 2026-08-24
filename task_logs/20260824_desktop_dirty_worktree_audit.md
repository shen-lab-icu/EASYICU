# Desktop dirty-worktree audit and selective migration

Date: 2026-08-24

## Outcome

The 28 tracked modifications in the older Desktop worktree were audited path by path. They were not disposable caches: most represented real Copilot, Project Monitor, data-package, and progressive-Planner work. The useful case-neutral changes were selectively migrated into the unified isolated product candidate. The older dirty worktree remains untouched as recovery evidence.

- Source worktree retained: `/Users/haibo/Documents/GitHub/EASYICU-desktop-app-v1`
- Source state retained: `codex/easyicu-desktop-app-v1@d4f7990`, 28 modified tracked paths
- Unified worktree: `/Users/haibo/Documents/GitHub/EASYICU-unified-product-20260823`
- Unified branch: `codex/easyicu-unified-product-20260823`
- Pre-audit recovery ref: `backup/unified-before-desktop-dirty-audit-20260824@ffe3a07`
- Migration commits: `88a29e9`, `dfeab35`, `f228a85`
- Push, merge, stash, prune, deletion: not performed

## What was migrated

1. Shell help actions now route to the single official `#guided` EasyICU Copilot. The historical Page Guide dock is no longer instantiated; `window.EUPageGuide` remains a compatibility alias that only routes or focuses the official Copilot.
2. The current PHI-screened host user turn carries typed concept authorization. Model output cannot self-authorize a concept selection.
3. Sparse event availability is represented by a typed, strict-binary data receipt without exposing patient counts.
4. Copilot and Project Monitor resolve the same host-owned pipeline workspace; run history preserves gate reason and checks while the Monitor remains read-only.
5. Analysis validation, numeric verification, operational mappings, and publication readiness are projected separately. A blocked publication gate no longer falsely rewrites validated analysis as analysis failure.
6. The progressive Planner received general shape, module, output-role, literature-roster, method-layer, and prompt-envelope constraints.

## Deliberately not migrated

The older dirty Planner automatically rewrote host coordinates such as schema version, plan digest, step id, role, module, objective, dependencies, and action when model output drifted. Its own focused test matrix exposed the problem: the coordinate-drift negative test no longer raised. The unified candidate therefore keeps those coordinates fail-closed. Parser repairs may normalize bounded content, but they may not silently rebind host authority.

The older dirty worktree's comparable focused matrix was `505 passed, 3 failed`; two failures were stale static assertions and one was the scientific coordinate-drift failure above. The unified candidate's post-migration matrix is green.

## Verification

- Focused Python matrix across Planner, data-package review, Copilot contracts/gateway/workflow/static routes, and UX reliability: `511 passed`, 5 warnings.
- Canonical JavaScript contracts: `27/27` passed.
- Ruff, Node syntax, Python compile, and Git diff checks: passed.
- Browser QA at `127.0.0.1:8897` with an isolated `EASYICU_HOME`:
  - Home contains one visible `EasyICU Copilot` launcher and no `#cpDock`; the launcher routes to `#guided`.
  - Guided contains the official account/API connection surface, no Page Guide text/dock, and no visible duplicate launcher.
  - Data Extraction contains a visible Copilot entry that routes to `#guided`.
  - `#agent` renders `Research Project Monitor`, has no conversation input or duplicate dock, and states that requirements/model/run initiation stay in Guided Copilot.
  - Home, Guided, Extraction, and Project Monitor had zero horizontal overflow; console warning/error count was zero.
- No provider was invoked, no credential was entered, and no real patient data was used.

## Exact-source desktop package

Because the previous release was built from `26960a7`, it did not contain these migrated changes. A new Apple Silicon package was therefore built from clean code commit `f228a85` and saved without overwriting older releases.

- App: `output/releases/f228a85/EasyICU.app`
- DMG: `output/releases/f228a85/EasyICU_1.0.0_aarch64.dmg`
- DMG size: `444284269` bytes
- DMG SHA-256: `6d69aba0c7ead5bba0b7a19cba00106723ba2359303b4111627b6a553252b683`
- App signature: strict/deep verification passed; ad-hoc signature, no Team ID
- Architecture: arm64
- DMG verification: valid checksum; the App mounted from the DMG also passed strict/deep signature verification
- Bundled Guided Copilot and shell-entry JavaScript matched the `f228a85` source byte for byte
- Native smoke: Home loaded at dynamic port `58207`, the Copilot launcher opened `#guided`, and Cmd+Q released the port and all package processes

This supports an internal Apple Silicon product candidate. It does not establish notarization, Intel compatibility, full-database clinical validity, provider-backed scientific correctness, full exact-head CI, or formal manuscript readiness.
