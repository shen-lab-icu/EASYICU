# StepAuthorityCapsule execution/resume integration

Date: 2026-07-16
Branch: `refactor/agent-control-plane`
Base: `823169f`

## Scope closed in this batch

This batch connects the previously isolated StepAuthorityCapsule storage
primitive to the execution loop without making it a second publication or
EvidenceStore authority.

- Initial generation now follows `reserve -> provider call -> content blob ->
  terminal receipt -> capsule -> checkpoint`.
- Repair recovery joins the durable logical-attempt receipt to the exact parent
  capsule and before/after code digests before adoption.
- The newest monotonic checkpoint is the only capsule selector; no directory
  scan, orphan discovery, or fallback to an older capsule is allowed.
- Exact execution seals can materialize a synthetic RunResult and skip repeated
  generation, concept audit, and sandbox execution while current deterministic
  gates still run.
- Auditor, validator, engine, provider, runner, interpreter/package, Docker image,
  mount, environment-file, and input identities are digest-bound. Scientific
  coordinates remain Planner/Coder owned.
- Revalidation writes a non-success transient checkpoint immediately, so a crash
  cannot leave an older `ok` attempt current.
- Output replay uses a crash-recoverable swap protocol and rejects symlinked or
  ambiguous output/backup states.
- The working `run_input_capsule.json` must match its sealed EvidenceStore bytes;
  missing, changed, or symlinked copies fail closed before code generation.
- Non-typed aliases are consumable only from the current successful producer;
  the Critic may inspect only exact evidence IDs registered by its own in-flight
  attempt, never an unpublished alias.
- Docker capability state is cleared when switching backends and copied into
  parallel step workers. A custom runner cannot inherit a stale Docker snapshot.
- Agentic CLI delegation remains available standalone, but capsule/exact-once
  mode deliberately uses the receipt-aware fallback Coder until a dedicated
  CLI transport adapter exists.

## Verification

- Final resume/provider/capsule/contract/perf suite: `286 passed` in 232.38 s,
  executed outside the outer tool sandbox so the runner's own macOS
  `sandbox-exec` path was exercised.
- Characterization + meta-generalization suite: `61 passed` in 3.89 s.
- Golden characterization: two consecutive green runs (`2/2` within the 61-test
  run, then `2/2` in 5.22 s).
- Blocker-focused Agentic/Docker/capsule/input-authority set: `20 passed`.
- Black, targeted Ruff, and `git diff --check`: passed.
- Independent adversarial review: ACCEPT; no remaining must-block fail-open,
  duplicate-payment, or authority-fallback finding.

## Deliberate non-blocking boundaries

1. Agentic CLI is not yet a capsule provider. Enabling it in exact-once mode
   requires its own durable reservation/transport and a digest covering CLI
   executable, version, model, and configuration.
2. Receipt atomic replacement protects process-crash recovery; this batch does
   not claim power-loss durability until file and parent-directory fsync are
   specified and tested.
3. Coder-provider identity currently invalidates more downstream audit/execution
   state than strictly necessary. This is safe but leaves a performance
   optimization for the next control-plane refinement.
4. Physical evidence aliases are still published by the existing EvidenceStore
   mechanism. Current-success filtering prevents failed attempts from being
   consumed, but a fully atomic alias/checkpoint two-phase promotion remains
   structural debt.
5. A fully exhausted historical provider receipt still needs an append-only,
   authority-bound continuation mechanism for validator-only re-audit. Current
   behavior fails closed rather than granting fresh calls.

## Next action

Claude performs a read-only review of this commit range. After acceptance, keep
the batch frozen; do not resume E3 until the remaining planned control-plane
extraction and milestone regression gate are complete.
