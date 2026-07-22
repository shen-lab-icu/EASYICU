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
  transports, are denied before any prompt delivery, and are never paper
  eligible.
- Provider usage is returned with the same response through
  `complete_with_usage`; the metering layer does not lock around network calls
  or trust shared `last_usage`, including through fallback and reproducibility
  wrappers.
- Active and superseded findings remain distinct benchmark dimensions.

## Second-review P1 follow-up

Three additional review findings were reproduced and closed without relaxing
the fail-closed boundaries:

1. Exact run reuse now binds the benchmark data seed and a digest of the
   actual input authority.  DataFrame values and external-file bytes are part
   of the full execution identity, while the independently frozen submission
   environment identity remains task-input agnostic so one operator freeze can
   authorize all nine distinct tasks.
2. External Planner prompts still withhold non-reviewed categorical literals,
   but expose deterministic opaque level tokens.  The host maps those tokens
   back to the locally observed labels before applying the existing exact
   Table 1 contract, so privacy no longer makes the contract unsatisfiable.
3. Executable entrypoints under `scripts/` and `examples/` now use the provider
   factory.  The static ownership scan covers both trees, and `OpenAIClient`
   itself rejects an unmanaged external destination before transport, so an
   unscanned direct constructor cannot bypass operator authorization.

The full reuse identity and the frozen paper-environment identity are
deliberately separate coordinates: reuse must match the task data exactly;
paper acceptance must match an operator-owned environment freeze outside the
result tree and also verifies each score against its full manifest identity.

## Third-review P1 follow-up

The final three authority penetrations were reproduced and closed:

1. Paper eligibility now requires a non-empty, valid input-authority digest.
   Figure 2 acceptance independently rejects a missing input authority in the
   frozen identity, score row, or run manifest; a matching frozen environment
   alone cannot authorize an unbound run.
2. Table 1 opaque tokens remain in the public `AnalysisPlan`.  A separate
   digest-bound, host-only `TableOneExecutionBinding` holds real observed
   levels for trusted execution and validation.  Captured Planner, Replanner,
   Coder, and repair prompts contain none of the private labels.
3. Every production prompt delivery goes through the generic provider graph
   authorization boundary.  Routers, fallbacks, metering, and reproducibility
   wrappers are traversed recursively; an unmanaged custom leaf is rejected
   before `complete`, even when the external-provider environment opt-in is
   present.

## Verification

- Review-fix focused matrix: 168 passed, 7 pipeline tests deselected because
  the checked-in Docker image carries the pre-branch source digest.
- Concurrency regressions prove two 0.15-second provider calls stay parallel
  through both metering and reproducibility wrappers while preserving exact
  per-role usage.
- Ruff, Black, py_compile: passed.
- Architecture gate: zero lower-is-better regressions.
- Module graph: zero cyclic SCC regressions.
- Second-review focused matrix: 156 passed.
- Pipeline configuration mirror, Ruff, py_compile, diff-check, architecture
  gate, and module graph all passed after the follow-up.
- Third-review offline matrix: 303 passed, 2 skipped.  This includes exact
  identity/Figure 2 acceptance, all four Table 1 prompt surfaces, custom
  provider and nested router/fallback negatives, diagnostic envelopes, visual
  QA, data foundation, Tier-2 jury, and idea-mining provider paths.
- Third-review Ruff, py_compile, diff-check, architecture gate, and module
  graph passed.  No architecture baseline refresh was used: the touched core
  metrics are unchanged or improved.

The wider 203-test diagnostic shard produced 197 passes plus six environment
or baseline failures.  After fixing the one follow-up configuration omission,
the remaining five failures are the existing Docker image/source-digest
mismatch; they do not reach the changed identity, privacy, or provider code.
The separately known cost-role expectation remains a baseline issue on both
this branch and mainline and was not mixed into this security follow-up.

The Docker-backed pipeline tests remain intentionally unrefreshed because the
existing image source digest predates this branch. That environment mismatch
was not hidden or weakened. No external provider, patient dataset, or online
benchmark was used.
