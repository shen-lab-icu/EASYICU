# Pi/external integration CI reconciliation

- Date: 2026-08-08
- Task: `PI-EXTERNAL-INTEGRATION-CI-RECONCILIATION`
- Shared remote branch: `fix/external-review-20260724-p0-p1`
- Integration worktree branch: `integration/pi-external-20260808`
- Code checkpoint: `25edc214742e856b570b2c1e09dd135da57396b3`
- Pi review/rollback checkpoint: `feat/pi-copilot-shell@bbd15d0` (unchanged)

## Outcome

The merged Pi, Web, data/export, and research-agent histories were reconciled at
their actual owner boundaries rather than by weakening tests. The code checkpoint
was fast-forwarded to the shared remote branch. No provider call, patient-data
read, formal M3 run, canary, or immutable paper image build was performed.

## Reconciled contracts

- CI now checks out sufficient Git history for the full-six release ancestry
  proof and the QC scripts retain Python 3.10 compatibility.
- Figure 2 acceptance binds the current scorer tree and current Table One schema;
  an obsolete schema remains an explicit negative test.
- Table One generation no longer emits an impossible measurement-provenance
  helper call when the plan declares no measured/count pairs.
- E1-E3 offline preflight preserves real control flow without publishing an
  unauthorized offline effect estimate.
- Pandas datetime assignment and Arrow large-string schemas are normalized at
  their owning boundaries for current dependency versions.
- Pi one-use run authorization is consumed before bound-context inspection, so
  an unauthorized caller cannot probe project state.
- The 288-feature catalog, packaging notices/runtime data, supersession fixture,
  and case-neutral shared-prompt contracts were brought back into agreement.
- The first exact-SHA ordinary matrix exposed three pandas 3 / Arrow 25 seams:
  two schema assertions bypassed the native-v2 normalized-schema owner, and
  MIMIC episode clocks retained a microsecond unit before a nanosecond LOS
  fallback was assigned. The assertions now consume the production schema
  owner and the duration owner normalizes datetime clocks before clipping.
- The next Python 3.12 matrix exposed a SciPy edge case: an all-identical
  two-group rank test can return a non-finite p-value. The Table One owner now
  defines this mathematically degenerate, fully tied case as `p=1.0` while all
  other non-finite rank-test results remain fail-closed.

## Local verification

- Figure 2 Canonical9 offline gate: 555 collected; 505 passed, 50 skipped,
  0 failed in 63.18 seconds.
- Affected Python suites outside Figure 2: 238 passed, 3 skipped, 0 failed.
- Table One/helper/resource/release cluster: 82 passed, 1 skipped, 0 failed.
- Web JavaScript owner wrappers: 14 passed, 0 failed.
- Arrow publication plus vasopressor-duration owner files after the CI finding:
  43 passed, 0 failed on pandas 3.0.5 and pyarrow 25.0.0.
- All focused Table One suites after the fully tied rank-test fix: 83 passed,
  0 failed.
- Ruff, workflow YAML parsing, JavaScript syntax checks, and `git diff --check`:
  passed.

## Exact-SHA remote CI

- Accepted ordinary CI:
  <https://github.com/shen-lab-icu/EASYICU/actions/runs/31279403274> — all seven
  jobs succeeded: wheel/sdist installation, Python 3.10/3.11/3.12 tests, and
  Windows/macOS/Ubuntu portability.
- Accepted research-agent CI:
  <https://github.com/shen-lab-icu/EASYICU/actions/runs/31279403256> — both
  Python 3.10 and 3.11 jobs succeeded.
- Superseded ordinary CI:
  <https://github.com/shen-lab-icu/EASYICU/actions/runs/31277319266> — Python
  3.11 succeeded and its Python 3.12 failure produced the fully tied rank-test
  patch; it is not the accepted checkpoint.
- Superseded diagnostic ordinary CI:
  <https://github.com/shen-lab-icu/EASYICU/actions/runs/31275253733> — its late
  Python 3.11 failures produced the final data compatibility patch; it is not
  the accepted checkpoint.
- Status: accepted exact-SHA ordinary and research-agent matrices are green.

## Scope boundary

This checkpoint validates code and offline contracts only. It does not advance
Canonical9 beyond 4/9, authorize a paper result, run M3, read a production cohort,
or call the configured local/provider model. Those actions remain separate,
explicitly authorized stages after exact-SHA CI and independent review.

## Local service handoff

- The stale browser tab had no live listener on port 8765. A new background
  server was started from the clean integration worktree at documentation head
  `00026c823a798c1acc567092381e7757c8440fad`; its production-code parent remains
  the CI-accepted `25edc214742e856b570b2c1e09dd135da57396b3`.
- The pinned Pi 0.84.1 runtime revision
  `0.84.1-1d13000d610d-install2` was installed under the private user runtime,
  not into the Git worktree. The install resolved 238 packages with zero known
  vulnerabilities.
- `GET /api/catalog`, the native HTML shell, Guided Pi JavaScript, and the Pi
  preview owner all returned HTTP 200. Public Pi status reported `ready`, no
  blockers, research/workspace modes, verified `gpt-5.6-luna` availability,
  credential storage `private_local_file_0600`, and no returned secrets.
- Browser automation policy did not permit an automated localhost tab reload;
  the operator must refresh the already-open tab once. No model message or tool
  action was sent during this service handoff.
