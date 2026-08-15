# Data/Web cross-layer correctness remediation — 2026-08-12

- Modules / tasks: `DATA-FIX1` and `WEBAPP-FASTAPI-NATIVE-QA · PATIENT-CROSSDB-VISUAL-PARITY`
- Scope: D1–D3 and W1–W3 from the 2026-08-12 systematic review
- Workspace policy: focused owner/contract tests only; no full exact-head matrix during Web E1 iteration

## Outcome

All six requested defects are closed in the current working-tree snapshot. The changes establish one typed owner for database identity, content-based freshness receipts for both public cache and conversion recovery, explicit 28-day outcome semantics, an immutable Cross-DB selection receipt carried into StudyContext/Agent, Pi project-state round-trip, and a recoverable registered Cross-DB job lifecycle.

The existing publication blocker is intentionally unchanged: `src/easyicu/data/concept-dict.LOCK.json` is still `finalized=false`, so a new clean six-database native-v2 extraction, QC pass and deliberate lock finalization remain required before sealing.

## Data remediation

### D1 — explicit 28-day mortality only

- `webserver/cohort_review.py` no longer derives `mort_28d` from hospital death plus length of stay.
- Missing 28-day events are not coerced to `False`; a survival curve requires an explicit event flag and a valid compatible time value.
- Missing or contradictory event/time evidence is excluded and remains non-reportable instead of being represented as survival.
- The former positive regression that locked the fallback was replaced with explicit-event, missing-event and contradiction contracts.

### D2 — schema-first database identity

- New owner `databases/detection.py` exposes `detect_database_identity()` and typed `DatabaseDetectionError`.
- Base loader, DataConverter and Web data I/O delegate to the same schema-first decision. Path basename is secondary evidence only; unrelated ancestor names are ignored.
- Negative contracts cover misleading `mimic-iv`/`eicu` path names, schema conflicts and unrelated ancestors.

### D3 — content-based freshness

- New owner `content_identity.py` produces stable SHA-256 content receipts plus bounded stat identity.
- Public cache fingerprints persist and validate content receipts instead of trusting size/mtime alone.
- DataConverter status stores `source_content_receipt`; a legacy completed status without a receipt rebuilds, and the source is verified to remain stable across conversion.
- Negative contracts mutate a source to different same-size bytes, restore its original mtime, and prove that both API cache and Parquet conversion invalidate.

## Web remediation

### W1 — exact Cross-DB selection handoff

- Cross-DB review responses now include a path-free `selection_receipt` with selected source IDs, labels, database identities, path hashes and one selection digest.
- StudyContext persists the receipt as `crossdb_selection`; single-source handoffs clear it.
- Agent renders the exact selected receipt and never recomputes scope from the global source registry. A plan-only multi-source context cannot fall back to one active source's path or stay count; an unavailable denominator displays `—`.
- End-to-end contracts use a six-source registry with a two-source selection to lock exact count, identity and digest round-trip.

### W2 — Pi project configuration round-trip

- Existing-project “研究配置” remains inside the Pi conversation and loads the authoritative workflow instead of switching to an empty legacy shell.
- Project binding re-renders after asynchronous session/workflow projection, preserving project ID, StudyContext revision, saved slots and scientific progress.
- Static and browser contracts lock “existing project → study setup” without returning to `0/8` or creating a duplicate project.

### W3 — registered Cross-DB job continuity

- New `POST /api/jobs/crossdb-summary` returns a job ID and path-free source/deadline/lease receipt.
- Registered and raw Cross-DB review share bounded pointer-only continuity, SSE progress, cancel, reconnect and selection-invalidation behavior.
- Source paths are leased for the real background read lifetime. Removal fails closed while leased; cancellation and deadline terminal states do not release the lease before the reader actually returns.
- Stable deadline and cancellation contracts cover job status, resource release and exact source snapshot identity.

## Verification

- Requested owner/contract suites: **46 passed** (`13` D1/D2 + `8` D3 + `25` W1/W2/W3).
- Python lint: Ruff passed on all touched Python and test files.
- Frontend parse: `node --check` passed for every modified JavaScript owner.
- Patch hygiene: `git diff --check` passed for the touched-file set.
- No full suite was run, in accordance with the E1 development policy.

One adjacent test remains red and is not a regression from D1–D3/W1–W3:

- `tests/test_webserver_crossdb_setup_frontend.py::test_crossdb_results_owner_executes_navigation_and_single_chart_contract`
- Its concurrently changed Node fixture now requires Cross-DB result tabs to implement roving `tabindex`, `aria-controls` and ArrowRight navigation. That P2 accessibility work was outside the six requested defects; `screens-viz-crossdb-results.js` was not changed for it. The failure is recorded rather than hidden or used to broaden this patch.

## Browser evidence

Current-snapshot desktop QA at 1440×900 exercised Home → official Patient demo → two-source Cross-DB review → Agent handoff → Pi project configuration:

- Registered Cross-DB displayed loading/progress and a working cancel action; a rerun completed successfully.
- With the registry restored to six sources, Agent showed exactly the two selected official sources, `Cross-DB exports 2`, digest-bound provenance and denominator `—`; it did not show the prior single MIMIC path or 140-stay detail.
- Selecting the existing E1 project projected its authoritative workflow (`3/7`) and existing setup; the session did not fall back to legacy `0/8`.
- Browser console: no errors. The temporary exact-snapshot server was stopped after QA.

The six-source registry and the pre-existing E1 StudyContext/scientific content were restored after the test. Pi sessions were rebound to the restored authoritative revision so the QA did not leave a synthetic Cross-DB scope attached to the user's E1 project.

## Remaining release work

1. Re-extract all six databases with the current native-v2 contracts from a clean commit, complete QC, update exact hashes and deliberately finalize the concept lock.
2. Address the separately recorded Cross-DB tab keyboard/focus contract, SPA title/focus/live announcements, Demo source identity/provenance and external-font/CSP hardening in their owner tasks.
3. Run the full exact-head matrix only at the E1 freeze/merge/release or formal-experiment checkpoint.
