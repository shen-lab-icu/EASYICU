# E1 preflight blocker closure

Date: 2026-08-14
Branch: `fix/pi-workspace-review-20260809`
Exact reviewed HEAD: `9e52733`

## Outcome

The research-agent offline framework release gate passes on a clean detached
worktree at `9e52733`: 135 tests passed, with resource-context, architecture,
module-graph, and normalized golden checks all green. This authorizes a bounded
fresh Web E1 run; it is not itself the Web E1 acceptance result or Canonical9
Provider authorization.

## Closed blockers

- Numeric claims carry typed effect-scale/estimand identity. Mixed OR/HR/RR
  sentences, transformed log-ratio fields, swapped estimates, and conflicting
  declared/source scales fail closed.
- Sealed StepResultEnvelope products are the live final-validation and writer
  authority. Writer tables resolve to immutable, digest-verified EvidenceStore
  content rather than mutable step output paths.
- Pre-selection universe access requires an explicit typed owner capability;
  ordinary generated steps cannot observe `EASYICU_UNIVERSE_PARQUET`.
- Substantive runtime replans cannot replace a human-approved Plan. They stop
  before registration/application with exact current/candidate/review digests.
- Human-review decisions and checkpoint state converge across crash windows;
  Web recovery reuses exact decisions and reconciles a missing global index
  from checksummed run-local recovery seeds.
- Provider unknown usage retains worst-case prompt and completion reservations,
  while human-review wait is excluded from persisted active wall-clock time.
- CodeRunner and Docker output capture are bounded; process groups are reaped
  on normal, timeout, and interruption paths. Docker mounts reject sockets and
  special files, secrets use owner-only transient env files, and paper-facing
  profiles require `network=none`.
- Provider URLs reject non-global address classes including CGNAT. MCP schemas
  are closed, extraction needs explicit patient-data scope, remote direct HTTP
  needs real TLS files, and trusted-proxy mode is explicit.
- The golden fixture now exposes reviewable validator/severity/reason/step
  identities rather than only an opaque set hash.
- Architecture governance scans duplicate helpers at every function scope and
  gates module/top-level-module growth; the module graph remains acyclic.

## Commits

- `c8238ae` scientific authority and durable recovery
- `b731a2a` initial runner/MCP boundary hardening
- `f008cbf` architecture-budget-preserving owner split
- `e1dcc5a` first golden adjudication
- `236f8e6` scientific bypass closure
- `6b441d0` bounded runtime and recovery surfaces
- `2bd8d22` governance and reviewable golden evidence
- `9e52733` robustness owner capability call correction

## Verification

- Clean framework release: 135 passed, 13 warnings.
- Follow-up focused matrices: 401 passed, 1 skipped.
- Architecture ratchet: `execution/phase.py` 5,944 LOC;
  `run_execute_phase` 4,987 lines; no lower-is-better regression.
- Module graph baseline: 519 modules, 1,961 edges, zero cyclic SCCs.
- Repository hygiene, Ruff, and `git diff --check`: passed.

## E1 operating boundary

1. Restart Web on exact committed HEAD, not the concurrent dirty worktree.
2. Use the official provider endpoint or reviewed loopback proxy.
3. Keep remote MCP disabled for E1.
4. Use the paper-facing Docker profile with `network=none`.
5. Run Planner-only to the durable human-review pause first, then approve the
   exact digest and continue Plan -> execute -> evidence -> figure -> manuscript.

## Deferred non-E1 work

- Runtime replan review is intentionally fail-closed and requires a fresh
  reviewed run; in-place mid-execution approval/resume remains a product task.
- Arbitrary custom remote provider hostnames still need connect-time peer
  pinning against DNS rebinding before production exposure.
- MCP path enforcement retains a read/write TOCTOU window until directory-FD
  no-follow operations replace path-based checks; remote MCP stays out of E1.
- Native Windows locking and live Docker-daemon behavior have focused simulated
  coverage but still need their platform integration lanes.

The separately staged planning batch present in the working tree is not part of
`9e52733` and was not modified or included in these commits.
