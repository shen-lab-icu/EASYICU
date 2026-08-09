# Pi Workspace final-review closure — 2026-08-09

## Scope and baseline

- Review baseline: `fix/pi-workspace-review-20260809@d963809398742a075aad76850bfb482728b679d1`.
- Implementation commit: `c004696efe0f6716925d829661ef9248406634ef`.
- Task ID: `PI-WORKSPACE-FINAL-REVIEW-CLOSURE`.
- Active module: Web / Guided Pi Workspace.
- Phase: final boundary closure before Workspace freeze.
- Frozen boundary retained: EasyICU Research Copilot continues to own plans,
  runs, gates, evidence, readiness, and manuscript authority.

## Review decisions and changes

1. **Truncated-read whole-file replacement (P1) — closed.**
   `easyicu_write_project_file` and `ProjectWorkspace.write_file()` are now
   create-only. Existing files can change only through the exact-edit owner,
   which reads the complete bounded source inside the Host and requires the
   current `expected_sha256`. A 100,000-character one-line regression proves
   that a 24,000-character truncated read cannot authorize destructive
   replacement.
2. **Workspace base root symlink (P1) — closed.** The declared outer workspace
   path is retained before resolution and rejected when it is a symbolic link.
   The boundary is checked again before project access, with stable owner codes
   for a symlinked or changed root. Outside content remains unchanged.
3. **Direct preview provenance (conditional P1) — closed.** The preview route
   now returns Host-owned wrapper chrome with an immutable
   `Workspace artifact · Unvalidated` notice and places model-authored HTML in
   a nested `sandbox="allow-scripts"` iframe. The same notice therefore exists
   in the product panel and when the URL is opened directly; the model document
   cannot edit or cover it.
4. **Node static-check environment (P2) — hardened.** `node --check` receives
   only `PATH`, `HOME`, `TMPDIR`, `LANG`, and `LC_*`. `NODE_OPTIONS`, provider
   credentials, proxy variables, and the remaining WebApp environment are not
   inherited.
5. **Scoped CI completeness (P2) — closed.** The gate now includes the packaged
   installer tests, parse-checks all three sidecar owner modules, and is also
   triggered when its shared native FastAPI browser helper changes.
6. **Research artifact provenance (P2) — closed.** The Host artifact endpoint
   projects the run owner's gate, readiness, sign-off state, reportability, and
   claim ceiling. The preview chrome visibly distinguishes analysis-only,
   reportable, stale-signoff, and unavailable governance states.

## Verification

- `93 passed`: Pi route, workspace, contract, static, gateway, provider, and
  packaged installer suites.
- `115 passed`: adjacent native static-route, provider-tool, Web security, and
  hosted-relay security suites.
- `43 passed`: route snapshots plus focused Workspace/static owner regressions.
- Ruff, four Node parse checks, workflow YAML parsing, CSS owner/foreign-route
  scan, CSS brace/comment scan, and `git diff --check` passed.
- Real Chromium report:
  `output/playwright/pi-preview-security-review2/report.json`.
  Both product and direct views show Host provenance; both nested hostile
  documents return `BLOCKED:SecurityError`. At a 1280px viewport,
  `scroll_width == viewport_width`, and the banner and preview frame both have
  non-zero dimensions.

## Supported concurrency and explicit non-claims

- V1 compare-and-swap locks are process-local. The supported native WebApp
  write configuration is one process; multi-worker atomicity needs an
  OS-backed lock or transactional store before it can be claimed.
- Files manually placed into the isolated workspace remain governed by the
  existing external-model disclosure and PHI/credential prohibition. A future
  import manifest may improve origin tracking, but no incomplete manifest was
  added in this closure.
- Push checks for the implementation SHA are running as scoped security run
  `31296758756` and repository CI `31296758750`. A PR-head run is still required
  before release because this evidence note will advance the branch once more.
