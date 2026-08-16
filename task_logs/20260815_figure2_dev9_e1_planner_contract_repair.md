# Figure 2 Dev9 E1 Planner contract repair

- Date: 2026-08-15
- Branch baseline: `feat/figure2-dev9-heldout27-20260815` at `efa408f`
- Authority: development diagnostic only; not paper authority
- Task: `e1_sepsis3_prevalence_mortality`

## Run evidence

- Input binding: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_efa408f_e1_input_20260815/development_binding_receipt.json`
- Exact-source runner image: `easyicu-research-agent:efa408f-dev`
- Image id: `sha256:1632234e38ece3610d3e1281351725c5995beec752f719a0a7d3b7f49bd35eab`
- Runtime validation: Docker, `--network none`, status `ready`, 11 method capabilities
- Run root: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260815_efa408f_e1_dev04`
- Provider attempts: 5 Planner calls; 153,077 accounted tokens; estimated cost USD 2.09401
- Execution boundary: no generated analysis code executed; the run stopped in Planner parsing

## Failure classification

The five attempts reached the same strict Planner owner through five distinct
contract violations: an unknown Table 1 key, two headline-primary steps, an
unknown distribution key, replacement of host-safe level tokens with guessed
strings, and the cross-family action id `descriptive.table_one` inside an
association plan. This is a general machine-readability defect in the Planner
contract projection, not an E1 numeric or clinical exception.

## Repair

- The retry guide now derives the exact accepted Table 1 and exposure/outcome
  distribution keys from their Pydantic owner models.
- Opaque binary level examples now come from `opaque_level_tokens(2)` and are
  used consistently for arrays and scalar selectors.
- The generic distribution example is secondary and states that the complete
  plan may contain at most one primary step.
- The scientific-action catalog now publishes a closed current-family
  allowlist and states that cohort, Table 1, raw distribution, and figure-only
  support steps do not acquire cross-family action ids.
- The compact retry reminder repeats the exact action allowlist.

## Verification

- 162 focused Planner/schema/action/prompt-budget/parser tests passed.
- Ruff checks passed on all six changed source/test files.
- `git diff --check` passed.
- Prompt smoke: 44,949 bytes; opaque-level example present; distribution example
  secondary; cross-family action guard present.

## Next

The first repair was committed as `8ca246b`, built as
`easyicu-research-agent:8ca246b-dev`, and rerun from a fresh `dev05` root.
Never resume or reuse `dev04`.

## Dev05 result

- Run root: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260815_8ca246b_e1_dev05`
- Provider attempts: 5 Planner calls; 156,288 accounted tokens; estimated cost
  USD 2.12344.
- Execution boundary: no generated analysis code executed; all five failures
  remained inside Planner validation.
- The remaining defects varied across attempts (invalid distribution interval,
  invalid Table 1 adjustment/mode, counts-only fields that declared inference,
  a cross-family action, and a non-enum literature design element). This showed
  that another prose-only retry patch would continue paying to discover the
  schema one field at a time.

## Strict transport-schema repair

- Owner: `agents/plan_payload.py` derives a closed transport projection from
  `AnalysisPlan`; host Pydantic and scientific validators remain authoritative.
- Representation-only adaptations are limited to `display_labels` key/value
  rows and nullable fixed-key robustness overrides, decoded before validation.
- Provider capability is explicit and fail-closed. It is not inferred from a
  model name or an HTTP 200 response.
- The exact Schema is immutable, counted in Planner prompt and hard-stop
  reservations, forwarded through the reproducibility/hard-stop/meter wrapper
  chain, and bound into provider transport policy v2 and ExecutionIdentity.
- Benchmark exceptions now persist safe structured-attempt metadata (stage,
  issue location/type, digest, finish reason, and usage) without raw response or
  parser text.
- CLI ownership is separated: `--planner-strict-json-schema` configures the
  provider transport and is recorded under `provider_transport_options`; it is
  never passed into `PipelineConfig`.

## Strict-schema evidence

- Schema name: `easyicu_analysis_plan_v1`
- Schema authority SHA-256:
  `609e9d662c9776b2399c55cce23c700ce1621b3583bc5b5038da3d3076042d6e`
- Wire payload: 25,701 bytes; 28 definitions; every object property required;
  every object closed with `additionalProperties=false`.
- Live exact-schema probe: provider accepted the code-generated schema and
  returned `finish_reason=stop`; 5,161 prompt + 113 completion = 5,274 tokens.
- Focused verification to this point: 249-test wrapper/provider/retry suite had
  no cached failures; an additional 92 provider/identity/schema tests passed,
  including the composed production wrapper chain. Ruff, compileall, and
  `git diff --check` passed.

## Next

Commit the strict transport repair, build an exact-source image for that commit,
validate source/image/runtime identity, and start E1 from a fresh `dev06` root
with `--planner-strict-json-schema`. Never resume or reuse `dev05`.

## Dev06 result

- Source commit: `d51c718`; image: `easyicu-research-agent:d51c718-dev`.
- Run root: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260815_d51c718_e1_dev06`.
- Provider attempts: 5 Planner calls; 190,043 accounted tokens; estimated cost
  USD 2.62937.
- Execution boundary: no generated analysis code executed; all five attempts
  remained inside Planner/schema/article-contract validation.
- Strict transport removed malformed JSON as the failure class, but it still
  admitted arbitrary prose in `scientific_capability`. Later attempts also
  lost already-correct scientific coordinates when the retry projection became
  too large and fell back to an under-specified shape.
- Initial transport was 119,083 bytes; retries grew to 138,936--140,092 bytes.
  The Planner was the only live role consumer not checked again at raw
  transport time, so those oversized retries bypassed the reviewed 120,000-byte
  ceiling.

## Dev06 owner-contract repair

- `agents/plan_payload.py` now derives the strict
  `scientific_capability` enum from the capability-owner vocabulary. The live
  schema is 26,007 bytes with authority SHA-256
  `2ce0b07af32e5eebfecec270e033ecb21c32e91c15f992ff75b917111178e15a`.
- `pipeline.py` wraps live Planner generation in the declared
  `planner_plan_generation` prompt-budget consumer. Every initial and retry
  request is therefore measured by the same transport boundary.
- Strict requests omit the duplicate illustrative JSON object and compact only
  syntax already carried by the closed wire schema. Scientific decisions,
  owner products, fail-closed semantics, citations, typed context, and plan
  coordinates remain in the prompt.
- Retry projection now preserves action, capability, citation/design bindings,
  typed specifications, and sensitivity ids. If a full projection cannot fit,
  the final rung is explicitly labelled as a coordinate table rather than as a
  partial `AnalysisPlan`.
- The reconstructed exact E1 initial request is 107,853 bytes: 81,846 message
  bytes including the schema-authority note plus 26,007 schema bytes. This
  leaves 12,147 bytes below the reviewed 120,000-byte boundary; the full
  analysis-action catalog remains selected.
- A live provider probe accepted the revised exact schema and returned
  `finish_reason=stop`. This proves transport compatibility only, not Planner
  or scientific success.

## Dev06 repair verification

- 128 focused structured-schema, retry-projection, transport-budget, and
  catalog-ladder tests passed.
- 262 additional prompt/scientific-contract regressions passed, including
  Table 1, exposure/outcome distribution, adjusted-model roster, plan roles,
  concept allowlists, family switching, and fixed prompt headroom.
- The previously completed 218-test provider/wrapper/transport suite remains
  applicable because the final compaction changed only strict Planner prompt
  rendering; Ruff and `git diff --check` passed after the final edit.

## Next

Commit this general owner-boundary repair, build and validate a new exact-source
image, and start E1 from a fresh `dev07` root. Never resume or reuse `dev06`.

## Dev07 result

- Source commit: `bed6368`; exact image:
  `easyicu-research-agent:bed6368-dev`, image id
  `sha256:2dafa1d788c47ca81cef25628c930c51c7f92dbd18f58255ce1b66338347012f`.
- Runtime/source validation passed under `network=none` with all 11 method
  capabilities.
- Run root: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260815_bed6368_e1_dev07`.
- The first Planner call returned HTTP 200 and a strict-schema response. It
  consumed 32,975 tokens (24,409 prompt; 8,566 completion), estimated cost USD
  0.50107, then produced seven host validation findings.
- Before any second provider call, the retry projector failed closed: the
  complete 11-step coordinate ledger was 5,150 bytes, above its 4,500-byte
  local envelope. No generated analysis code executed and no scientific
  coordinate was silently discarded.

## Dev07 projection repair

- The final coordinate-ledger rung now interns repeated actions,
  capabilities, methods, products, citation keys, design elements, and
  sensitivity ids in a deterministic string table.
- References use tagged `["s", index]` pairs, so an invalid literal integer
  from a non-strict provider cannot be confused with an interned string.
- The representation remains explicitly labelled as prior-coordinate evidence,
  not a partial `AnalysisPlan`; the next response must still emit the complete
  strict schema and full literature-binding applications.
- A strengthened 11-step E1-shaped regression with three literature bindings
  per step, multiple outputs/sensitivities, and the invalid binary endpoint
  shape now fits under 4,500 bytes and round-trips every asserted authority
  coordinate. Eight focused projection/schema/transport tests, Ruff, and
  `git diff --check` passed.

## Next

Commit the lossless projection repair, build a new exact-source image, and
start E1 from a fresh `dev08` root. Never resume or reuse `dev07`.

## Dev08 result

- Source commit: `3b794c3`; exact image:
  `easyicu-research-agent:3b794c3-dev`, image id
  `sha256:24840004d3c30984fb255649a9b0868857e01a7c92aec6509501c75b7b1ced97`.
- Runtime/source validation passed under `network=none` with all 11 method
  capabilities.
- Run root: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260815_3b794c3_e1_dev08`.
- All five Planner calls stayed inside the reviewed transport envelope. Their
  total request sizes, including the 26,007-byte strict schema, were 107,853,
  112,987, 113,590, 113,607, and 114,021 bytes.
- The calls accounted for 128,673 prompt and 33,947 completion tokens (162,620
  total; estimated cost USD 2.30514). No generated analysis code executed.
- Attempt 1 had six host-validation findings, attempt 2 had three, and attempts
  3--5 converged on one identical finding: the schema-/2 distribution declared
  an incomplete closed interval tuple. This confirms the lossless retry ledger
  worked and isolates the remaining defect from transport or coordinate loss.

## Dev08 interval-contract repair

- The distribution owner now diagnoses every member of the coupled schema-/2
  tuple and reports the received values: `wilson`,
  `patient_cluster_robust_wald`, and non-null confidence must be present
  together.
- The strict initial Planner directive and the retry guide now publish that
  coupled invariant explicitly. The repeated-unit interval method remains
  declared while dependence is null before host binding; it does not authorize
  the Planner to invent grouping authority.
- The ordinary worked distribution example was also restored to its executable
  auxiliary role, and its regression now checks the host-owned opaque level
  tokens rather than obsolete literal 0/1 values.
- Ninety-one focused distribution-owner, Planner strict-schema, retry-guide,
  prompt-example, and prompt-budget tests passed. Ruff and `git diff --check`
  passed.

## Next

Commit the general interval-contract repair, build and validate a new
exact-source image, and start E1 from a fresh `dev09` root. Never resume or
reuse `dev08`.
