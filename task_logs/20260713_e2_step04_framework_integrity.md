# E2 Step04 framework-integrity review and scoped fixes

Date: 2026-07-13  
Task: `FIG2-CANONICAL9-GATE` / E2 development stress run  
Branch: `main`

## Outcome

E2 Step03 and its planner-authorized supporting figure are clean. The first
fresh Step04 execution produced correct cohort and mortality counts but exposed
generic framework-integrity gaps. A later same-run Step04 execution on
`main@d6405ce` passed the host integrity checks: both typed inputs loaded,
`94,458 -> 50,640` keys and all 52 shared value columns reconciled, bounded
intervals stayed within `[0,1]`, and the Critic passed. The architecture was
then hardened further through `main@5bae8c6`; therefore one final Step04-only
rerun on the committed framework remains required before the run can advance.
This is still a development stress run, not the paper-facing canonical batch.

## Evidence reviewed

- Run: `research_output/_diagnostic_e2_fresh_20260712/bench_e2_gpt-56-luna/E2_lactate_mortality/aware/run_20260712T210917_7ec41b`
- Clean Step03 figure:
  `steps/03_lactate_distribution_and_missingness_figure/outputs/distribution_availability.png`
- Step04 output reviewed:
  `steps/04_death_incidence_and_absolute_risk/outputs/step_summary.json`
- Step04 exact resolved inputs:
  `resolved_inputs/04_death_incidence_and_absolute_risk.json`

The old Step04 result reconciles the locked cohort (`94,458`) and the supplied
subset (`50,640`) correctly in an independent read-only audit, but its summary
also reports the same subset input as `loaded=false`, verifies only stay-id
membership rather than shared values, and contains
`prevalence_ci_high=1.0000000000000002`. These defects make the artifact
development-only even though the underlying descriptive counts are correct.

## Framework changes

- `3d1164b fix(agent): bind effect scope before code generation`
  - Coder, repair, and AgenticCoder now share one closed effect-authorization
    predicate.
  - Authorization requires an exact effect-method owner plus a declared typed
    effect product, or a schema-locked model roster; inferred analysis family
    cannot expand scope.
  - A clean Critic `pass` no longer carries a contradictory regenerate request.
- `eedcfbc fix(agent): verify typed subset inputs host-side`
  - Typed-input summary claims are checked against host-resolved bindings.
  - A checked subset reconciliation must name distinct exact inputs, unique key
    columns, and every shared non-key value column; the host replays the full
    comparison and rejects self-comparison or convenient-column selection.
  - Probability/risk/prevalence/fraction outputs and their applicable
    confidence bounds are strictly bounded to `[0,1]`, without constraining
    RR/OR/HR/RD scales.
  - The new owner is isolated in
    `src/easyicu/research_agent/audits/step_summary_integrity.py`; the shared
    `validators.py` did not absorb the roughly 700-line implementation.

No benchmark ID, lactate variable, database, or Figure 2 answer was added to
production routing or prompts.

### Final architecture closure

- `0ff1a36 fix(agent): freeze typed lineage across replans`
  - Binds current evidence to the immutable producer step and plan-level
    scientific scope, verifies evidence kinds, and checks exact statistic
    payload values.
- `791642d fix(agent): constrain planner-owned result rendering`
  - Keeps product selection and scientific ownership in the Planner while
    limiting deterministic rendering to closed typed products.
- `81a4f17 fix(agent): preserve bound renderer inputs`
  - Makes render-only code retain all resolved table/statistic bindings and
    prevents scalar dtype guards from using bitwise boolean negation.
- `762d60c fix(agent): verify figure source semantic obligations`
  - Requires value-bearing, hash-current source lineage and validates effect,
    prediction, interval, curve, and multi-parent semantics without case
    keywords.
- `5bae8c6 fix(agent): persist blocked manuscript critiques`
  - Writes an explicit fail-closed critique artifact on writer pause, gate
    failure, or critic exception.

## Verification

- Focused effect/critic checks: 74 passed before the integrity batch.
- Integrated validator/routing/meta/coder/execute batch: `304 passed`.
- Ruff, `py_compile`, and `git diff --check`: green.
- Read-only replay against the old Step04 summary reports six expected errors:
  two missing typed-input row counts, one nested loaded-state contradiction,
  one checked-reconciliation/unloaded contradiction, one incomplete host
  reconciliation contract, and one out-of-range prevalence CI bound.
- Final committed-tree verification:
  - complete `tests/research_agent/test_pipeline.py`: `260 passed`;
  - requested F1/F2/meta gates: `175 passed`;
  - method-suite, declared-product, coder, typed-lineage, schema, and writer
    focused tests: `237 passed`;
  - total: `672 passed`, with Ruff, `py_compile`, and diff checks green.

## Runtime diagnosis and acceleration implication

The old Step04 took about 851.5 seconds. Two coder calls consumed about 707.3
seconds (83.1%); the contract repair alone cost about 330 seconds. Moving the
closed product authority and typed-input reconciliation contract into the
first coder prompt is therefore the safest measured acceleration: it removes
avoidable full-script regeneration without weakening concept or execution
gates. No cache, oracle, answer injection, or benchmark-specific runner was
introduced.

## Next action

Rerun only `04_death_incidence_and_absolute_risk` in the same aware-arm
development run with `REUSE_STEP_CODE=0`, so the committed framework and fresh
agent code are tested together.
Accept it only if it is `status=ok`, emits no unauthorized effect product,
passes host key-and-all-shared-value replay, contains internally consistent
input bindings, keeps all bounded metrics within `[0,1]`, and ideally requires
zero contract repair. Do not rerun Step03. If this gate is clean, resume from
the Step04 figure through the remaining E2 chain, then continue E3 Step06/07
and its figures. H3 must wait for an Agent-authored scientific replan; do not
lower its stability threshold or manually alter k, seed, or outcome steps.
