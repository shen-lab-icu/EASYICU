# E1 step-scoped plausibility authority repair

Date: 2026-07-27
Task: `FIG2-E1-PLAUSIBILITY-SCOPE-20260727`
Status: offline repair and focused verification complete; no Provider call or E1 launch yet.

## Failure reproduced

The third fresh E1 canary stopped at `01_define_analysis_cohort` after the
plausibility receipt gate compared its output with every ranged variable in the
study-wide `ResearchContext`.

Read-only replay source:

```text
/Volumes/外置硬盘/easyicu_data/canonical9_runs/
batch_20260727_luna_miiv_dev_eee248b_e1_canary3/
e1_sepsis3_prevalence_mortality/aware/run_20260728T021420_06f488
```

The step's actual sealed input authority was:

```text
planner inputs: []
raw input contracts: [sep3_sofa2_max]
sep3_sofa2_max analysis_plausibility_range: absent
sep3_sofa2_max retain_and_flag policy: absent
```

Therefore the correct flag-only obligation set was empty. The old gate instead
used 38 unrelated ranged variables from the 104-variable global context, while
the unconditional Coder instruction induced a false
`plausibility_audit.sep3_sofa2_max` claim.

## Repair

- Added immutable `FlagOnlyPlausibilityScope`, compiled from the exact
  digest-verified `resolved_inputs.raw_input_contracts`.
- The same scope now drives:
  - Coder authority text;
  - deterministic pre-execution gate;
  - early and final executed-receipt gates;
  - concept-audit cache identity;
  - resume revalidation.
- A non-empty scope creates the obligation even if generated source omits the
  range lookup. Generated code cannot opt out of host policy.
- Receipt coverage is exact: missing, unexpected, and duplicate columns block.
- An empty scope rejects any non-empty or malformed plausibility receipt claim.
- Resume derives the scope from the immutable selected step capsule's verified
  resolved-input bytes. A mutable checkpoint projection is observability only.
- Legacy records without a capsule are narrowed to exact raw Planner inputs;
  there is no fallback to all ranged context variables.
- Generic prompts no longer publish an unconditional receipt shape. The
  host-owned attachment lists the exact columns, or explicitly says the scope
  is empty and the receipt must be omitted.

## Real run3 offline replay

No model or Provider was called.

```json
{
  "step_id": "01_define_analysis_cohort",
  "planner_inputs": [],
  "raw_contract_columns": ["sep3_sofa2_max"],
  "expected_columns": [],
  "scope_sha256": "6e431034362b4528d0d9a4f460954a412cc9764fd71b9fb83a73c9b8ea333b04",
  "static_reasons": [],
  "current_receipt_reasons": [
    "plausibility_audit_without_step_authority"
  ],
  "without_unauthorized_receipt_reasons": []
}
```

This is the intended result: the step owes no range receipt, its old induced
receipt is an unsupported policy claim, and the same artifact passes after that
claim is absent.

## Verification

Focused suites passed:

- plausibility scope, static/receipt gates, Coder guidance, and owner-boundary
  contracts: `113 passed`;
- resume, sealed-capsule revalidation, and deterministic runner isolation:
  `110 passed`;
- architecture measurement contracts: `31 passed`.

Static architecture checks:

- Ruff: passed for all touched Agent source and tests.
- Import Linter: 7 contracts kept, 0 broken.
- Deptry: no dependency issues.
- Research-agent module graph diff: passed without moving the baseline;
  `execution/phase.py` is 9 lines below its baseline and both measured
  orchestration functions have zero growth.
- `git diff --check`: passed for the scoped patch.

The approximately 38-minute full suite was not rerun at this stage. The policy
compiler and resume recovery live in the new authority owner module; the
ResearchContext owner resolves raw contracts, and phase passes one immutable
compiled value to the Coder, static gate, receipt gate, cache, and replay.

## E1 launch boundary

The old runner image corresponds to old source and must not be reused. After
the scoped commits and all concurrent Web/test work leave a clean tree:

1. build an immutable exact-source image with the final 40-character revision;
2. require readiness `ready`, runtime `docker`, and network `none`;
3. run a fresh aware-only Step 01 canary stopping at
   `@product:table:cohort_flow`;
4. inspect the real plan, scope, Provider ledger, step summary, and downstream
   non-execution before deciding whether to start a second fresh run through
   `@product:table:adjusted_association_estimates`.

Both are development diagnostics with `paper_authorized=false`. Do not resume
the exhausted run3, raise its Provider budget, run the historical naive arm, or
promote either canary into paper evidence.
