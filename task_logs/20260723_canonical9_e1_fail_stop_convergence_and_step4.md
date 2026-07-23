# Canonical9 E1 fail-stop, convergence, and Step 4 closure

Date: 2026-07-23 EDT
Task: `FIG2-CANONICAL9-REALRUN`
Scope: MIMIC-IV full0717-v2, aware arm, local Luna Provider

## Honest status

- Paper-facing Canonical9 remains **0/9** until a fresh source-bound run passes
  the full scorer. No diagnostic run is promoted.
- The latest diagnostic run is preserved at
  `/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_adaptive_63c8711/e1_sepsis3_prevalence_mortality/aware/run_20260723T112040_81f957`.
- Steps 01--03 succeeded. Step 04 was sent to the Coder and failed before
  transport because its initial prompt was 43,097 bytes, above the 42,000-byte
  gate. The Paper workflow then incorrectly started Step 05 despite the failed
  required predecessor. The request was interrupted manually.
- That interrupted historical Step 05 receipt remains visibly pending inside
  the invalid diagnostic run. It is retained as failure evidence, not repaired
  in place or reused as paper authority.

## Framework fixes

1. Paper submission profiles are sequential and fail immediately after any
   non-`ok` required step. Later steps, transition callbacks, and replanning are
   suppressed, so a failed Coder step cannot spend a Writer/model call.
2. Initial-generation interruptions now terminalize the Provider transport
   receipt as failed even for `KeyboardInterrupt`/`SystemExit`, while preserving
   the original interruption.
3. The structured Step 04 contract
   `missingness_and_measurement_frequency_audit` with exact analysis-cohort
   input and two declared table products is owned by the deterministic
   missingness runner. It emits a concrete `measurement_availability.csv` and
   reads the manifest-selected revised plan rather than a stale original plan.
   The Coder prompt is eliminated rather than compressed close to the limit.
4. The deterministic penalized-convergence repair is versioned to
   `penalized_convergence_contract_v2`. It accepts only a boolean traced to the
   `success` field of a reviewed `scipy.optimize` result. A free variable,
   literal, iteration-count heuristic, or custom optimizer cannot be promoted
   to convergence authority. Historical v1 receipt ids remain parseable.
5. A Planner step declared as the primary adjusted-association analysis cannot
   carry a secondary-only typed model roster. It must contain at least one
   primary requirement; a proxy remains secondary instead of being relabelled.
6. Current Canonical9 input selection now binds `npj_dm/20260719`, matching the
   runtime default. The archived `20260718` profile remains immutable. The
   Figure 2 scorer-tree digest was reauthorized after the schema change.
7. The five legacy integration fixtures now use factory-registered built-in
   offline mocks. Unknown custom clients remain fail-closed.

## Exact data replay

The deterministic Step 04 implementation was replayed on the exact sealed
94,458-stay E1 cohort without a Provider call:

`/Volumes/外置硬盘/easyicu_data/e1_step4_replay.5IdbCM`

Result: `ok`; 21/21 declared inputs resolved; zero missing inputs; both declared
table products were materialized.

## Verification

- Integrated non-Docker matrix: **511 passed**.
- Figure 2 scoring-input authority after reauthorization: **35 passed**.
- Canonical9 typed selector `--check`: ready, 9 tasks,
  SHA-256 `7c1421ade83561d7727a8f6865cbbe99ffbe312437587064d64614bade793210`.
- Ruff, Black, Python compilation, `git diff --check`: passed.
- Architecture lower-is-better gate: exact baseline, no regression.
- Module graph: no new cycle.

The immutable image `easyicu-research-agent:source-5e567eb` was built from the
clean package source, with image digest
`sha256:04a0650bd576b02af6890a347dac6303fcf41cc7b43f0b78260cc9cf56fd2467`.
The full post-repair source-bound integration file then passed **13/13** with
its pytest temp root on the Colima-mounted external drive. The script-integrity
case now simulates a host-side post-execution digest change because the
production container correctly mounts the executable script read-only.

A fresh E1 run must create new execution identity and operator freeze evidence;
the interrupted batch must not be resumed.

## Fresh Planner synonym diagnostic and structural routing closure

The first fresh source-bound launch after the fixes was preserved at:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_adaptive_56cb992/e1_sepsis3_prevalence_mortality/aware/run_20260723T120740_86c1a7`

It reached a valid six-step plan but was stopped before any analysis step ran.
The Planner expressed the same descriptive Step 04 science with a third
compositional spelling:

- method: `missingness_and_informative_measurement_audit`
- products: `table:missingness_audit` and
  `table:measurement_availability_audit`

The historical standard-executor selector recognised only two exact
method/product spellings.  This was a routing defect, not a reason to add the
new literal to another allowlist.  The replacement classifier now binds the
closed analysis kind structurally:

1. the only typed input scope is `artifact:analysis_cohort`;
2. exactly two typed table products are declared;
3. one product is the missingness audit and one is a measurement/source
   availability audit;
4. method and product tokens are drawn from a closed descriptive-audit
   vocabulary;
5. any model/effect, test, figure, longitudinal analysis, score-quality
   analysis, extra product, unknown product, or extra typed input fails closed.

All three observed Planner spellings select the same zero-Provider deterministic
executor.  The new `measurement_availability_audit` spelling is materialised as
its own concrete CSV and is bound in `step_summary.json`; it is not satisfied by
an undeclared alias.

The exact archived fresh plan now selects
`missingness_source_availability_audit` offline before the Coder path.  The
diagnostic launch made real planning/replanning Provider requests before the
manual interruption, so it remains invalid development evidence and is never
resumed or scored.

Verification for this increment:

- structural routing/negative controls and adjacent ownership tests: **69
  passed**;
- fail-stop, convergence, primary model roles, prompt budgets, Provider budget,
  execution identity, trajectory compaction, and routing matrix: **389
  passed**;
- architecture lower-is-better metrics: exact, with no `execution/phase.py`
  growth;
- module graph: no new cycle;
- resource/context envelope: unchanged.  Its checked-in source digest was
  realigned to the already-reviewed `agents/core.py` interruption fix from the
  preceding increment; no resource selection or prompt metric changed;
- Ruff, Black, Python compilation, and `git diff --check`: passed.

This increment still does not make E1 paper-facing.  A new immutable image,
execution identity, production-input authority, and operator declaration are
required before the next fresh E1 launch.

## Implicit locked-cohort scope exposed by the next fresh E1

The next fresh, fully authorized E1 launch is preserved at:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_adaptive_19e7720/e1_sepsis3_prevalence_mortality/aware/run_20260723T122453_d88e08`

Steps 01--03 completed successfully, including the zero-Coder grouped Table 1.
The Planner's revised Step 04 retained the closed method/product contract but
listed only bare physical columns.  It omitted the redundant
`artifact:analysis_cohort` coordinate because every AnalysisStep already runs
against the orchestrator-owned, locked `COHORT_PARQUET`.  The first structural
matcher required that explicit coordinate, so it declined ownership and the
step entered the Coder path.  The run was stopped during that request; no Step
04 script or result was accepted, and no later step started.

The Provider budget receipt proves interruption accounting worked as intended:
Step 04 `initial_generation.transport.state="failed"`,
`error_type="KeyboardInterrupt"`, and `provider_calls=1`.  It is not left
pending and will never be resumed.

The corrected scope rule now accepts either:

- the explicit sole typed source `artifact:analysis_cohort`; or
- no typed source, meaning the framework's implicit locked `COHORT_PARQUET`.

Any additional `artifact:`, `table:`, or `dataset:` source still rejects
standard-executor ownership.  The exact archived `analysis_plan_revision_2.json`
now selects `missingness_source_availability_audit` offline before the Coder.
This is a framework-level current-cohort convention, not a benchmark keyword or
case-specific prompt.

## E1 scientific-contract failure and batch canary closure

The first authorized `f7b4d1b` E1 run is preserved at:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_adaptive_f7b4d1b/e1_sepsis3_prevalence_mortality/aware/run_20260723T124316_20acfc`

It is diagnostic-only and must never be upgraded. Step 01 applied
`sep3_sofa2_max >= 1` as primary eligibility, reducing the 94,458-stay
universe to 33,997 exposure-positive stays. Step 02 then correctly failed
because its closed Table 1 comparison still required both exposure levels. The
per-question fail-stop suppressed Steps 03--06, but the formal batch
incorrectly initialized E2 before the operator interrupted it. E2 never
reached execution and is not evidence.

Implementation commit `09f8e5c` closes this as three case-neutral contracts:

- Planner typed-binding validation rejects a primary cohort whose predicates
  leave fewer than two levels of a downstream Table 1 comparison or required
  primary estimand. Replaying the exact archived E1 plan produces the expected
  `collapse a downstream closed comparison` rejection.
- The two canonical empty-comparison runtime diagnostics are plan/data
  contradictions. They fail closed with `llm_repair_used=false`; changing
  Python cannot restore rows removed by an upstream cohort definition.
- A formal Canonical9 batch must begin with E1. E2--E9 are
  `batch_canary_blocked` unless E1 is publication-ready, manuscript-ready,
  zero-error, has a valid locked evaluation envelope, and its exact paper
  scorecard is `gate_reportable`.

Provider transport accounting is now written before each call under
`.runtime/provider_transport_receipts/`. Receipts contain hashes and usage,
never prompt or response text, and terminalize as completed, failed, or
cancelled on `KeyboardInterrupt`.

Verification on source-bound image `easyicu-research-agent:source-09f8e5c`
(`sha256:e4e999a639810e3d9d58847205bfb766961c83403bb4073565bb689badddf3a9`):

- scientific/authority/control matrix: 234 passed, 4 deselected;
- Provider/outbound/Table 1/cost matrix: 132 passed;
- missingness/runner selected matrix: 35 passed after replacing the obsolete
  unauthorized Mock subclass fixture;
- three historical scripts now construct Provider clients only through the
  reviewed factory;
- Ruff, Black, Python compilation, diff-check, architecture/resource
  baselines, and the 315-module/31-package/1,028-edge zero-cycle graph pass.

Paper-facing progress remains 0/9. The next action is a new immutable execution
identity, operator declaration, and fresh E1 canary. Only a `gate_reportable`
E1 releases the remaining eight questions.

## Fresh E1 canary: comparator P0 closed, Step 04 API mismatch exposed

The clean `4132eea` canary is preserved at:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_adaptive_4132eea/e1_sepsis3_prevalence_mortality/aware/run_20260723T132615_44441a`

The real Luna plan kept the 94,458-stay eligibility cohort independent of the
Sepsis-3 exposure. Step 01 completed, and the host-owned grouped Table 1 in
Step 02 completed with both comparison levels. Step 03 prevalence/outcome
summary also completed. This closes the specific comparator-erasure failure
from the `f7b4d1b` diagnostic run on a real Provider response.

Step 04 failed closed after its two bounded repairs. The last repair correctly
added explicit binary-domain checks but called the host primitive as
`strict_numeric_input(series, name=variable)`. The primitive accepted only the
Series positional argument, so Docker returned:

`TypeError: strict_numeric_input() got an unexpected keyword argument 'name'`

The method now accepts mutually consistent `name` or `column` diagnostic
labels. They cannot select data or weaken numeric validation. The exact failed
Step 04 script was replayed against the same cohort with the patched source and
exited zero, producing both declared tables and `step_summary.json` under an
external-drive verification directory.

The diagnostic run also exposed a producer/verifier schema drift at the
posthoc scoring boundary. Current run status emits explicit execution,
artifact, scientific, and paper-authorization axes plus per-step scientific
completion states, while the strict scorer allowlist still rejected those
fields as unknown. The scorer now requires and validates the exact completion
schema, every explicit axis, an empty scientific-incomplete list, and one
closed completion-state object per required step. The v3 rubric scorer-tree
digest was updated to bind the corrected verifier. False completion axes and
malformed step-state objects have negative regression coverage.

This run remains diagnostic-only and must never be upgraded. E2--E9 were
correctly recorded as `batch_canary_blocked`; no downstream analysis run was
started. Paper-facing progress remains 0/9 pending a new clean commit, image,
execution identity, authority declaration, and fresh E1 canary.

Verification for the closure increment:

- exact failed Step 04 script replay: exit 0, all three declared outputs
  produced on the external drive;
- descriptive primitives and adjacent repair selection: 34 passed;
- scorer, scoring-input, task sealer, bench integration, safety issuer, rubric,
  completion axes, and descriptive primitives: 227 passed;
- Ruff and Black: passed.

## Fresh E1 canary: derived-cohort cardinality must be runtime-owned

The next authorized `8f62c65` E1 canary is preserved at:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_adaptive_8f62c65/e1_sepsis3_prevalence_mortality/aware/run_20260723T135252_091c6c`

The preceding scientific-contract and Step 04 API fixes held. The Planner
defined a seven-step workflow, and Step 01 preserved both exposure levels. Its
eligibility rule removed two stays with missing admission-time authority,
materialising a 94,456-row analysis cohort from the 94,458-row run input.
Step 02 then failed because generated code compared the current
`COHORT_PARQUET` length with the initial ResearchContext cardinality, hard-coded
as 94,458. One concept-audit repair and the bounded code repair retained the
same stale literal, so the step failed twice. The run terminalised as
`diagnostic_only`; the batch canary prevented E2--E9 from starting.

This is a cross-step runtime-coordinate defect rather than a new E1-specific
scientific rule. `ResearchContext.cohort.n_stays` can describe the pre-
eligibility run input, while every downstream step consumes the current locked
cohort. The closure therefore adds three case-neutral controls:

- both host and Docker runners read only the Parquet footer and expose its
  current cardinality as the reserved, host-owned
  `EASYICU_COHORT_ROWS` coordinate;
- Coder authority states that ordinary denominators come from the loaded frame
  and any explicit integrity assertion must compare against that runtime
  coordinate, never the initial context count;
- the mechanical AST gate follows only a frame loaded from the host-owned
  `COHORT_PARQUET` path. A positive literal compared with its direct length is
  replaced deterministically with the host coordinate before any LLM repair.
  Unrelated table lengths and already-dynamic checks are unchanged.

The exact archived Step 02 script produces one such finding and the
`execution_cohort_runtime_row_count_v1` repair changes only the stale literal:

`if locked_n != int(__import__("os").environ["EASYICU_COHORT_ROWS"]):`

Pre-image verification before the immutable-image replay:

- new negative/positive runner and AST cases plus Provider-budget, typed-
  artifact-lineage, and cohort-schema adjacency: **163 passed**;
- architecture lower-is-better gate: no regression; `execution/phase.py`
  remains exactly at baseline and `gates/preflight.py` is seven lines smaller;
- resource/context envelope and zero-cycle module graph: passed;
- Ruff, Black, Python compilation, and `git diff --check`: passed.

Paper-facing progress remains 0/9. The `8f62c65` run is immutable diagnostic
evidence only. A source-bound image must replay the archived Step 02 script
successfully before a new authority and fresh E1 are issued.

The first `38beac7` source-bound replay confirmed the new row-count contract:
the Docker environment contained `EASYICU_COHORT_ROWS=94456`, and the original
94,458-versus-94,456 exception disappeared. Execution then reached the next
lineage boundary and exposed a second projection mismatch. The Coder had copied
`observed_shape.opaque_levels` from its outbound-safe prompt view into a lookup
against the digest-verified local ResearchContext JSON. The local schema
correctly stores the private execution binding as `observed_domain.levels`, so
the script raised `KeyError: 'observed_shape'`.

This does not justify exposing category literals to the Provider. The
case-neutral closure keeps both views:

- Provider prompts retain only `observed_shape.opaque_levels`;
- the generated source contains no real category label;
- a narrow AST finding rewrites only the exact outbound projection lookup to
  `observed_domain.levels` before execution, so
  `closed_categorical_counts` receives the SHA-bound local levels;
- unrelated shape metadata and already-correct runtime lookups are unchanged.

The two projection-bridge controls and the preceding row-count/runner and
lineage matrix pass **165/165**. Architecture remains below the frozen
lower-is-better baseline, with no module cycle or resource-envelope expansion.
A rebuilt image must now replay the same archived script through both corrected
boundaries before authority issuance.

The rebuilt `source-93c4126` image
(`sha256:0fcf94ac171cc7bf4a65531de2ffdc55504ac259411f68a90a6e64721d32fcb4`)
then replayed that exact archived Step 02 successfully: return code 0 in 1.92
seconds with both declared outputs. The first launch against this image stopped
before any Provider call when one cold Colima runtime-provenance subprocess
exceeded its fixed 60-second capture timeout; the identical capture immediately
passed in 1.86 seconds with all 11 capabilities. Because no code, image, input,
or rubric changed, a fresh R2 authority was issued rather than weakening the
runtime gate.

## Fresh E1 R2: numeric-binary local level binding

The authorized R2 canary is preserved at:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_adaptive_93c4126_r2/e1_sepsis3_prevalence_mortality/aware/run_20260723T142906_35cc18`

It independently closed all earlier barriers in one run:

- runtime capability capture passed with 11 capabilities;
- Step 01 retained all 94,458 stays and both Sepsis-3 groups while using the
  runtime row-count coordinate rather than a literal;
- Step 02 used the zero-Coder grouped Table 1 executor and completed with both
  comparison groups;
- Steps 03 and 04 completed, with Step 04's bounded post-mutation repair
  executing successfully in Docker.

Step 05 then exposed a host-context inconsistency. The exact
`sep3_sofa2_max` descriptor said `dtype=float64`, `n_unique=2`,
`is_binary=true`, `min=0`, and `max=1`, but omitted `observed_domain.levels`.
Generated code correctly deferred private level binding to the digest-verified
local ResearchContext, so both bounded script executions failed closed with:

`ValueError: Expected two closed levels for sep3_sofa2_max`

This cannot be fixed reliably by another LLM rewrite because the missing fact
belongs to the host. The case-neutral correction makes
`observed_domain_for_series` bind `[0, 1]`, `[0.0, 1.0]`, or `[False, True]`
only when both values of an actually observed numeric binary domain are
present. A one-level constant never receives an invented comparator. The local
ResearchContext retains the typed levels, while every Provider projection
continues to receive only `binary_numeric_indicator`, cardinality, and opaque
tokens.

The R2 run remains `diagnostic_only`; it completed 5/9 execution states and
the batch canary correctly marked E2--E9 `batch_canary_blocked`. No downstream
question was initialized. The first adjacent matrix after the correction
reported **444 passed**. Its five red cases are outside this change: four old
custom `_SequenceLLM` fixtures are now rejected by the Provider registry before
callback, and one Docker test expected the preceding source digest while the
worktree was intentionally dirty. Focused observed-domain, outbound, Table 1,
and runtime-level binding checks are green; the new immutable image must next
rebuild the context and replay the archived Step 05 script before another
authority is issued.
