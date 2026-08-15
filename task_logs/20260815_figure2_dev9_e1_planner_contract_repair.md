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
