# Research-agent resume authority / provider-budget hardening (2026-07-15)

## Scope

This batch hardened the shared EasyICU research-agent engine before any further
H2 development execution. It did not add benchmark-specific prompts, variable
names, scientific methods, estimands, or primary deterministic runners.

The work addressed the confirmed adversarial findings around:

- immutable run identity and selective deterministic resume revalidation;
- exact host authority for probe and cohort-materializer checkpoints;
- typed input/product/evidence authority and validate -> seal -> register order;
- statistical/provenance fail-close behavior;
- scoped coder context, minimal patch repair, and aggregate typed repair reasons;
- one durable per-step budget for real provider attempts and transport retries;
- Markdown evidence binding and manuscript current-authority consumption.

## Commits

- `06e067d perf(agent): bound provider calls and localize repairs`
- `9837a3c fix(agent): enforce typed evidence authority end to end`
- `c9bdf8b fix(agent): revalidate resume checkpoints before reuse`
- `ba69619 test(agent): follow extracted deterministic gate boundary`
- `b05776b test(agent): assert aggregate typed repair boundary`
- `de4af7f perf(agent): audit only contract-valid code digests`

## Confirmed Claude findings closed

1. Host deterministic cohort materialization now has an exact, closed authority
   contract. It verifies owner, kind, producer, generation mode, evidence id,
   canonical cohort SHA, Parquet row count, and checkpoint accounting. It is not
   generalized to arbitrary deterministic steps.
2. Mechanical preflight now catches reconciliation exceptions suppressed by
   `finally` control flow, including calls in `try`, handler, or `else` suites,
   import aliases, and simple name aliases. Nested unexecuted function bodies do
   not create false calls.
3. OpenAI SDK retries remain disabled so there is one retry owner. HTTP
   408/409/429/500/502/503/504, connection failures, and finite Retry-After are
   handled by the explicit loop. Every real transport attempt consumes the same
   durable budget; non-finite Retry-After values fall back safely.
4. Bullets, nested lists, blockquotes, and assertive ATX headings no longer
   exempt qualitative result claims from evidence binding.

## Resume and evidence behavior

- Current deterministic validator fingerprints are persisted per successful
  checkpoint. Unchanged checkpoints take a zero-work path.
- Validator/code drift replays sealed artifacts only. It does not invoke Coder,
  runner, Analyzer, or LLM concept audit.
- Failed replay appends `resume_validator_invalid`, retires current aliases
  atomically, and invalidates dependent downstream checkpoints.
- A failed alias retirement rolls the manifest back; a corrupt newest
  checkpoint remains fail-closed.
- Step attempt history stays append-only while current authority remains
  latest-per-step.
- Same-stem figure formats are treated as representations of one logical
  figure, with editable vector authority preferred. Two distinct real products
  claiming one semantic alias still fail closed.

## Provider-call acceleration

- The configured per-step provider limit is shared by initial generation,
  patch repair, full rewrite fallback, compatibility repair, Analyzer, concept
  auditor, fallback clients, and transport retries.
- Reservations are written atomically before the provider call and restored on
  resume. A corrupt receipt or exhausted allowance prevents the call.
- Cohort-prose translation, which occurs before the ordinary step loop, now
  uses the unique declared `table:analysis_cohort` step as its budget owner (or
  a stable host pseudo-step if no unique owner exists). The owner is latched
  across replan/resume and cannot be renamed to refresh the budget.
- Scoped coder context no longer pads to 36 arbitrary variables and keeps an
  authoritative `source_concept` companion family atomically, even when that
  family exceeds the soft column cap.
- Deterministic semantic/mechanical gates still run before every execution,
  but the expensive LLM concept audit now runs only once for the exact code
  digest that has executed successfully and passed the early host-owned output
  contracts. Runtime- or contract-broken drafts no longer consume repeated
  semantic-audit calls.
- Runtime repair transport failures and no-op repairs are retried inside the
  repair layer without re-executing an unchanged known-failing digest. All
  retries remain bounded by the logical repair allowance and durable provider
  budget; terminal fallback retains the causal `repair_failed` reason.
- The host-owned trajectory stability executor now publishes truthful typed
  input consumption receipts for the exact paths and SHA-256 identities it
  already verifies, including tabular row counts and partial receipts on a
  later fail-closed path. No method, cluster count, seed, threshold, exposure,
  outcome, cohort, or estimand is selected by this receipt layer.

## Verification

- Final focused integration set before the audit-ordering pass: `804 passed`
  in `353.98s`.
- Claude-adjacent AST/retry/Markdown counterexamples: `139 passed`.
- Pre-step cohort translation budget and execute integration: `69 passed`.
- Resume revalidation + execute contract: `68 passed`.
- Materializer authority counterexamples: `18 passed`.
- Meta-generalization probe: `14 passed`.
- Deferred audit/runtime-repair, provider, resume, contract, and meta group:
  `159 passed`.
- Trajectory executor/pipeline and visual-governance group: `31 passed`.
- Complete `test_pipeline.py`: `263 passed` across four disjoint shards.
- Final complete `tests/research_agent` coverage: `3796 passed, 10 skipped`,
  zero failures. Four disjoint file shards covered every `rg --files` test;
  two ignored-but-collected bootstrap/exit-status files were then run
  explicitly (`9 passed`). Collection count and the one module-level skip were
  reconciled before handoff.
- Ruff, `py_compile`, and `git diff --check`: passed.

## H2 copied-run revalidation

The original run was not modified. An APFS copy of
`research_output/_diagnostic_h2_8317_dagfix_20260714/H2_vasopressor_causal/aware/run_20260714T090014_75dd3c`
was prepared with the same materialized-column registration used by the
benchmark runner.

Run-input/evidence authority migration itself produced `invalidated={}`. The
new deterministic replay then found one real historical fail-open in
`01_target_trial_protocol`: its sealed script catches every exception from
`reconcile_binary_event_presence`, records an error payload, and continues to
write outputs. The old outer checkpoint was nevertheless `status=ok`.

Consequently:

- `00_probe` revalidated successfully;
- `01_target_trial_protocol` became `resume_validator_invalid`;
- 02--06 were invalidated transitively because their current authority depends
  on the unsafe Step 01 checkpoint;
- the engine correctly refused a Step-07-only resume.

This is not a compatibility false positive and must not be bypassed. H2 now
needs a safe Step 01 repair/re-execution followed by deterministic downstream
revalidation (and re-execution only where authority cannot be re-established).
The original run directory remains unchanged and no model endpoint was called
during this audit.
