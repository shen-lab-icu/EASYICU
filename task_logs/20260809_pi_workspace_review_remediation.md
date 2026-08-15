# Pi Workspace external-review remediation — 2026-08-09

## Scope and baseline

- Review baseline: `bbd15d0d62393e53b049e80c8fb400d4ef576356`.
- Unified starting point: `main@4ab68685a77bdc26ac5fa89226411782865d5685`.
- Delivery branch: `fix/pi-workspace-review-20260809`.
- Implementation commit: `e711874` (`fix(web): harden Pi workspace release boundaries`).

The patch preserves the frozen Research Copilot authority boundary. It changes
only the newer project-artifact workspace, its preview chrome, its host grants,
and the scoped release gate.

## Closed findings

1. **Direct preview URL origin isolation (P1).** The preview response now owns
   `CSP: sandbox allow-scripts`, `frame-ancestors 'self'`, and
   `Referrer-Policy: no-referrer`; the iframe still omits `allow-same-origin`.
   A hostile HTML browser regression proves that both the product iframe and a
   direct top-level preview receive `BLOCKED:SecurityError` when reading an
   EasyICU-origin localStorage sentinel.
2. **Workspace-root symlink escape (P1).** Both the shared `projects` directory
   and each hashed project root fail closed before read/write when they are
   symbolic links. Stable owner codes distinguish the two boundaries, and
   outside content remains unchanged.
3. **Host-owned scientific provenance (conditional P1).** Every workspace
   resource now carries immutable `workspace_artifact / scientific_evidence=false /
   unvalidated / unsupported` metadata. The preview renders an unvalidated,
   not-scientific-evidence banner outside the model-authored iframe.
4. **Raw workspace egress disclosure (P2).** First-use consent, the workspace
   composer, and the packaged workspace skill say that file contents may be
   sent to the configured Pi model and prohibit PHI, patient rows, credentials,
   and private clinical data.
5. **Grant semantics (P2).** One-use controls call `consume_once()`; project
   writes require the separate reusable `has_capability("workspace_write")`
   contract. The ambiguous `provided_actions` escape surface was removed.
6. **Multi-session lost update (P2).** Replacements and exact edits require the
   current `expected_sha256`; stale or missing digests fail with stable 409
   codes. An in-process per-project lock makes compare-and-swap atomic across
   separate `ProjectWorkspace` instances; a two-writer regression permits one
   winner and rejects the other.
7. **Bounded-check wording.** Check receipts now identify
   `check_scope=bounded_static_syntax`; neither the tool nor UI promotes the
   result to scientific or clinical validation.
8. **Current-SHA release evidence.** A path-scoped GitHub Actions workflow runs
   the Pi contract suite plus the hostile Chromium test whenever this boundary
   changes. Remote status is recorded only after the branch is pushed.

## Verification

- `86 passed`: Pi routes, workspace, contract, static, gateway, and provider
  configuration suites.
- `102 passed`: native static routes, provider tools, and provider/privacy
  boundary regressions.
- `3 passed`: packaged Pi installer/manifest regressions.
- Ruff passed on every changed Python owner and test; Node syntax, YAML parse,
  CSS ownership/foreign-selector checks, brace/comment checks, and
  `git diff --check` passed.
- Browser report:
  `output/playwright/pi-preview-security/report.json` with
  `iframe_storage_result=BLOCKED:SecurityError`,
  `direct_storage_result=BLOCKED:SecurityError`, and `passed=true`.

## Explicit non-claims

The review's final V1 strategy says to stop after the scoped Workspace boundary
and current-SHA gate. Accordingly, this commit does not claim to close the
older cross-provider DNS pinning, live Anthropic/Google credential canaries, or
a model-price-derived USD ceiling. Those require a separate provider-security
contract and real provider credentials/pricing inputs; existing token and call
ceilings remain unchanged. No fake network canary or zero-cost price table was
introduced.
