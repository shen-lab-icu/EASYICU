# PHI outbound and execution-identity hardening

Date: 2026-07-22
Branch: `codex/phi-outbound-hardening-20260722`
Base: `1434aa9`

## Scope

- External repair providers receive only a host-owned structured diagnostic
  envelope; raw stdout/stderr remains local.
- Non-reviewed observed categorical literals and cohort extrema are withheld.
- Every production external-provider entrypoint uses the canonical factory and
  records the non-secret authorization decision and endpoint.
- A content-addressed `ExecutionIdentity` binds submission profile, runner,
  image digest, network policy, provider/model authorization, prompt pack, Git
  state, seed, and host-runner authorization.
- Figure 2 paper acceptance and `--reuse-existing` require exact identity;
  host-runner authorization is never paper authority.
- Figure 2 acceptance additionally requires an operator-frozen expected
  identity supplied outside the result tree; nine mutually consistent but
  arbitrary result-declared identities cannot authorize themselves.
- Legacy memory write failures are non-fatal after a successful run.
- Unknown/custom provider adapters are treated as unmanaged external
  transports, receive only the closed diagnostic envelope, and are never
  paper eligible.
- Provider usage is returned with the same response through
  `complete_with_usage`; the metering layer does not lock around network calls
  or trust shared `last_usage`, including through fallback and reproducibility
  wrappers.
- Active and superseded findings remain distinct benchmark dimensions.

## Verification

- Review-fix focused matrix: 168 passed, 7 pipeline tests deselected because
  the checked-in Docker image carries the pre-branch source digest.
- Concurrency regressions prove two 0.15-second provider calls stay parallel
  through both metering and reproducibility wrappers while preserving exact
  per-role usage.
- Ruff, Black, py_compile: passed.
- Architecture gate: zero lower-is-better regressions.
- Module graph: zero cyclic SCC regressions.

The Docker-backed pipeline tests remain intentionally unrefreshed because the
existing image source digest predates this branch. That environment mismatch
was not hidden or weakened. No external provider, patient dataset, or online
benchmark was used.
