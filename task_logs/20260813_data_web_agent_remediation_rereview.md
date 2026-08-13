# Data / Web / Agent remediation re-review — 2026-08-13

- Modules: `DATA-FIX1`, `WEBAPP-FASTAPI-NATIVE-QA · PATIENT-CROSSDB-VISUAL-PARITY`, `FIG2-CANONICAL9-GATE`
- Requested scope: fix D1–D3 and W1–W3; independently inspect the existing A1–A3 working-tree changes; re-review the affected paths for omissions
- UI method: direct implementation and real-browser QA; no Figma or design-artifact work
- Test policy: directly related owner/contract tests plus minimum adjacent checks; no Provider run and no full exact-head matrix during Web E1 iteration
- Workspace: changes were isolated from the shared dirty snapshot, committed atomically by owner area, and verified again from a clean detached worktree; unrelated concurrent edits were preserved

## Outcome

The six requested Data/Web defects are closed in committed history. A1–A3 were also re-read at their owner boundaries and exercised through their focused contracts. The second pass found six adjacent omissions; all six were repaired before final verification:

1. the Cohort frontend still described the removed hospital-death/LOS 28-day derivation;
2. paper-replication outcome normalization still collapsed explicit 28/90/365-day mortality into generic `death`;
3. the Web layer still had a third database detector, plus MIMIC bucket/unreadable-schema edge cases;
4. Cohort Summary and Patient feature-coverage memoization still used only weak file metadata;
5. a Cross-DB receipt could still fall back to the global single export if its lifecycle stage drifted;
6. Pi briefly exposed the legacy `0/8` aside while the authoritative workflow request was in flight.

After those repairs, the focused re-review found no remaining P0/P1 defect in the inspected D1–D3, W1–W3, or A1–A3 paths. This is not a claim that the entire repository or release matrix is green.

## Commit ledger

- `20e1b30 fix(data): bind source identity and selection policy` — D2 schema-first database identity and the core D3 content-receipt owners/callers/tests.
- `dbd5521 fix(agent): separate runtime authority and timing semantics` — A1–A3 owner changes and focused contracts.
- `a8c0ddd fix(data): preserve fixed-horizon outcome semantics` — D1 plus the adjacent Cohort/Patient D3 memoization closures.
- `5939844 feat(agent): enforce publication-grade scientific review` — later Agent integration on top of A2's dependency-neutral finding injection.
- `d00d990 feat(web): unify evidence-bound research workflow` — W1–W3, their negative/lifecycle contracts, and the integrated Web/Copilot owner updates.

At final verification, branch `fix/pi-workspace-review-20260809` and `origin/fix/pi-workspace-review-20260809` both resolved to `d00d990` before this evidence-only log commit.

## Data fixes

### D1 — fixed-horizon mortality remains explicit and nullable

- `webserver/cohort_review.py` accepts 28-day mortality only from a dedicated fixed-horizon event field. Hospital mortality and hospital LOS are no longer fallback evidence.
- Missing event flags remain unknown rather than becoming `False`; KM membership requires both an observed event flag and compatible event/follow-up time.
- A `true` flag after day 28 and a `false` flag with follow-up shorter than 28 days are excluded consistently from the event-rate denominator and survival curves.
- `screens-viz.js` no longer presents the retired derivation and now labels the basis as a dedicated flag plus follow-up.
- `research_agent/replication/paper.py` preserves explicit `mort_28d`, `mort_90d`, and `mort_365d` endpoint identity instead of collapsing them into generic in-hospital death.

### D2 — one schema-first database identity owner

- New owner `databases/detection.py` is consumed by Base, DataConverter, and Web data I/O.
- Prepared-table schema outranks the selected directory basename; unrelated ancestors are ignored. Mixed evidence, unreadable identity tables, and unknown prepared layouts fail closed with stable codes.
- Official upper-case table names are discovered by enumerating children case-insensitively, so behavior does not depend on APFS accepting a lower-case spelling.
- MIMIC converter bucket folders are not treated as MIMIC-III evidence because both generations may use that prepared layout.
- Explicit public aliases pass through the same normalization owner.

### D3 — content-aware cache and conversion recovery

- New owner `content_identity.py` issues SHA-256 receipts bound to stable size/mtime/ctime/device/inode evidence and rejects files that change while being hashed.
- Public concept-cache fingerprints persist a content-receipt index and hash again when cheap evidence changes. A same-size mutation with restored mtime produces a new cache key.
- DataConverter records the input receipt for each completed conversion, rebuilds legacy receipt-less statuses, and verifies the source again after conversion before publishing a completed status.
- Cohort Summary and Patient feature-coverage in-memory cache signatures now include ctime/device/inode (and manifest digest when present), closing the same-size/restored-mtime adjacent paths found in re-review.

## Web fixes

### W1 — exact Cross-DB selection is the source boundary

- Registered/raw Cross-DB results issue a path-free `crossdb-selection-v1` receipt containing the exact selected source IDs, labels, database identities, path hashes, count, and canonical digest.
- StudyContext persists and validates that receipt; single-export routes clear it and do not retain a stale multi-source scope.
- Agent renders the receipt count/scope/digest and does not read the global registry for denominator, path, or selected-source count.
- A plan-only context fails closed even if the receipt is damaged, and a valid multi-source receipt fails closed even if its lifecycle stage drifts. Neither case can bind the active single MIMIC export.

### W2 — existing project/revision round-trip stays in Copilot

- Agent emits a typed, one-use, path-free project-binding handoff containing project ID, StudyContext ID, and exact revision.
- Pi validates the revision before publishing the immutable project mapping and rechecks it after resolution; a TOCTOU race does not leave a partial binding.
- “Study setup” remains in the bound Pi conversation and sends the authoritative path-free Study Setup Receipt instead of switching to an empty legacy configuration.
- While the workflow request is pending, the aside now shows the bound project/revision and an honest loading state. It never flashes the legacy `0/8` summary before the authoritative projection arrives.

### W3 — registered summary uses the background-job lifecycle

- The production browser client has no synchronous registered-summary helper or fallback. It submits `POST /api/jobs/crossdb-summary` and receives job ID, exact selection receipt, deadline, and path-lease receipt.
- Registered and raw Cross-DB jobs share bounded local continuity metadata, SSE progress, cancellation fence, reconnect/probe, and source-selection invalidation.
- Cancellation/deadline can make the user-facing job terminal while an uninterruptible reader drains, but the source lease and JobManager capacity remain held until that reader actually returns.
- The old synchronous backend endpoint remains a compatibility/test surface; no production UI code calls it. Removing that endpoint would be a separate breaking API change, not required to close the foreground Promise path.

## Agent A1–A3 verification

### A1 — code failures do not impersonate isolation failures

- Only narrow backend/startup symptoms trigger a fixed, host-owned isolation probe.
- Generated stderr is only a probe trigger; it cannot authorize host fallback or an environment-failure classification.
- `NameError` and ordinary `Bad file descriptor` tracebacks retain the Coder-repair route. A probe-confirmed backend failure remains typed and fail closed.

### A2 — planning dependency direction

- `planning/replan_gate.py` does not import execution/agent layers.
- The execution owner compiles immutable `ValidationFinding` values and injects them through the public replan contract; Planning consumes receipts rather than runtime selector state.

### A3 — `*_first_time` is observation time

- Materializer, ResearchContext representation semantics, and the Coder prompt consistently define `*_first_time` as the first non-null observation inside the materialization window.
- An explicit zero/event-negative value is still an observation. Missing time does not prove event/treatment absence.
- Onset/initiation/time-zero requires typed event authority or a qualifying transition derived from the bound long trajectory.

## Verification evidence

### Final integrated focused regression

- Command selected the D1–D3/W1–W3/A1–A3 owner, negative, boundary, lifecycle, and executable JavaScript contracts.
- Result: **123 passed, 0 failed** (`3` pre-existing warnings), Python 3.11.15 from the repository `.venv`.
- This final batch includes same-size/restored-mtime mutations, misleading/mixed/unreadable database schemas, nullable/contradictory 28-day outcomes, two-source/six-registry scope, plan-stage drift, Pi revision races, registered job cancel/deadline/draining capacity, child diagnostic spoofing, dependency direction, and first-observation semantics.

### Clean exact-head verification after commits

- A detached clean worktree at `d00d990` passed **102 focused tests**: Agent A1–A3 `68`, Data D1–D3 `8`, and Web W1–W3 `26`.
- The Web batch exercised the registered-summary job's success, cancellation, deadline, draining-capacity, and path-lease paths; exact Cross-DB selection receipts; stale/tampered revision rejection; and the existing-project Copilot round-trip.
- `node --check` passed for all affected API, Cross-DB, StudyContext, Agent, and Guided owners.
- Ruff passed for all affected Python owners; `git diff --check 20e1b30^..HEAD` passed; the clean verification worktree had no tracked or untracked residue.
- `lint_progress.py` passed all six module handoffs with three size warnings; `lint_main_plan.py` passed with eight dashboard rows.

### Static quality gates

- Ruff on all affected Python owners and focused tests: **passed**.
- `node --check` on `api.js`, Cross-DB/StudyContext/Agent owners, and `screens-guided-pi.js`: **passed**.
- `git diff --check`: **passed** for the shared working tree.

### Real-browser QA

An isolated current-snapshot server on `127.0.0.1:8876` was exercised at desktop viewport through Agent Projects → existing StudyContext project → Guided Copilot.

- Agent route: title `Agent Projects — EasyICU`, `h1` focused, no horizontal overflow.
- Handoff receipt was consumed once and carried the exact project ID and StudyContext revision.
- With the workflow response deliberately delayed, the aside showed `Bound project · r3 / Loading authoritative configuration…` at 100/300/700/1500 ms; legacy `0/8` was absent at every sample.
- After settlement, the authoritative projection showed `2/7 required stages complete`, retained `Bound StudyContext`, focused the Guided `h1`, and had no horizontal overflow.
- Runtime API surface: synchronous registered-summary helpers were `undefined`; `startCrossdbReviewSummaryJob` was a function.
- Browser console: 0 errors and 0 warnings (Chromium emitted only its verbose autocomplete suggestion for the Provider ID input).

The temporary server and Playwright session were stopped after QA.

## Remaining boundaries / handoff

1. The publication blocker is unchanged: `src/easyicu/data/concept-dict.LOCK.json` remains intentionally unfinalized and must be closed only by a clean six-database native-v2 re-extraction, QC, exact-hash update, and deliberate seal.
2. No Provider/full exact-head CI was run; Canonical9 remains 4/9 and paper authority remains frozen.
3. `content_identity.py`, `databases/detection.py`, their callers, and their focused tests were committed atomically in `20e1b30`; the final clean-checkout regression confirmed that no import dependency is missing.
4. Job state is intentionally in-memory: browser refresh/temporary SSE loss reconnects while the server lives, but a server restart invalidates the bounded pointer and reports the job unavailable. Cross-process durable recovery is not claimed.
5. Large route owners remain structural debt (`screens-guided.js`, `screens-viz.js`, `screens-agent.js`); this remediation did not mix a characterization/split refactor into correctness hotfixes.
