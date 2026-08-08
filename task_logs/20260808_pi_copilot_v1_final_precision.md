# Pi Copilot V1 final precision hardening

- Date: 2026-08-08
- Module: web
- Task: `PI-COPILOT-V1-FINAL-PRECISION`
- Branch: `feat/pi-copilot-shell`
- Starting commit: `f9b7de43d51be104a64e138bec687c35fa030f40`
- Review input: `/Users/haibo/.codex/attachments/d9b443b0-a00b-4298-ad32-9f7c88853780/pasted-text.txt`

## Outcome

The two remaining V1 P1 precision gaps are closed without expanding Pi authority or changing the UI. Existing Guided project migration now preserves filesystem-path identity, and `runtime_integrity_verified` now covers the complete installed production JavaScript dependency tree rather than only `@earendil-works/pi-*` packages. The small project-initialization concurrency P2 is also closed with a per-project transaction lock.

## Changes

### Exact scientific input path

- `active_export.path` and `extraction.export_dir` use a bounded exact-path sanitizer: only leading/trailing whitespace is removed; internal characters are unchanged.
- Migration no longer routes paths through prose whitespace normalization.
- The end-to-end regression uses a path containing double spaces, parentheses, and Chinese characters and verifies exact preservation from Guided slots through the StudyContext patch.
- Typed historical time-window objects such as `{"hours": 24, "anchor": "ICU admission"}` are preserved instead of stringified; legacy text windows still map to `preset`/`label`.

### Complete production dependency integrity

- Installer command is now `npm ci --ignore-scripts --omit=dev`.
- Installation schema is `easyicu.pi-runtime-installation/2`; the private runtime revision includes `install2`, so an older partial-integrity receipt is never silently reused.
- Every installed production `package.json`, `.js`, `.mjs`, and `.cjs` under `node_modules` is hashed and compared exactly at startup.
- The regression proves the directly executed `typebox` package and entrypoint are included, and that mutating TypeBox invalidates the runtime.
- A real install under Node 24.11.0 audited 239 packages with 0 vulnerabilities and recorded 11,044 dependency files. Gateway status reported `runtime_integrity_verified=true`.

### Atomic project initialization

- `resolve → migrate/create → bind → legacy-session reconciliation` now runs under one per-project reentrant lock.
- Different projects are not globally serialized.
- A two-thread regression proves concurrent initialization creates exactly one StudyContext and both callers receive the same binding.

### Same-owner test drift

- Five static-route assertions still pinned `api.js?v=20260808-pi-authority1` after production moved to `pi-authority2`; the stale assertions now match the shipped asset identity.

## Verification

- New precision regressions: `5 passed` (three red-to-green defects plus two frozen legacy time-window shapes).
- Complete Pi owner suite: `58 passed`.
- Provider/StudyContext/static-route adjacency: `101 passed`.
- Ruff and `git diff --check`: passed.
- Real Pi runtime install: 239 audited, 0 vulnerabilities, 11,044 hashed dependency files.
- Gateway installation status: Node 24.11.0 supported; dependency installed, entrypoint available, runtime integrity verified.
- Wheel and sdist: built successfully with `uv build`; Pi installer, lockfile, and sidecar entrypoint are present in the wheel.

No browser rerender was repeated because this patch changes no HTML, CSS, or UI behavior. The previous V1 visual QA remains the relevant evidence.

## Retained boundaries

- `missing_required` remains conservative for legacy projects that do not persist a typed analysis family. It may ask a descriptive/data-quality user to confirm initialization, but it cannot run with invented science; no new text heuristic was added.
- Installation Node version remains audit metadata while startup enforces the supported Node range; this work does not claim exact Node executable reproducibility.
- DNS request-time TOCTOU, Anthropic/Google credentialed canaries, and a USD budget profile remain documented V2/non-blocking work.
- The known Canonical9 scorer-authority failures are outside the Web/Pi owner and were not resealed.
- A new remote current-SHA CI result requires pushing the new commit. This task creates the local commit only and does not claim remote CI evidence.
