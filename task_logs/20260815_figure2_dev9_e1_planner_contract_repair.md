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

Commit the repair, build a new exact-source runner image, and start E1 in a new
`dev05` root. Never resume or reuse `dev04`.
