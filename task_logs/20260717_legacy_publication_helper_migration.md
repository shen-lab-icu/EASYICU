# Legacy publication-helper adapter migration

Date: 2026-07-17 EDT

Branch / starting HEAD: `refactor/agent-control-plane@be6738c`

Task: `AGENT-E3-FIGURE-LEGACY-HELPER-MIGRATION`

## Decision

The E3 figure child exposed a case-neutral host API compatibility defect, not a
new scientific capability gap. An already-sealed candidate used runtime
reflection to guess the signature of EasyICU's publication helper. The helper
has a stable keyword API, while the legacy adapter interpreted its variadic
compatibility parameters as required, duplicated `fig`, raised `TypeError`, and
silently fell back to manual exports. The same `inspect` namespace also made the
mechanical coercion proof conservatively fail.

New code is now instructed to call:

```python
save_publication_figure(
    fig=fig,
    out_dir=out_dir,
    stem=stem,
    contract=contract,
)
```

The shared engine contains one removable, versioned compatibility migration for
the exact obsolete adapter. It does not select an exposure, outcome, cohort,
method, estimand, data column, figure claim, or result.

## Safety boundary

- Preflight emits the typed host-owned reason
  `host_helper_runtime_introspection`; common `inspect` and helper import aliases
  and direct `__signature__` access are blocked.
- Automatic migration requires that exact structured finding. The broader
  `INVALID_HELPER_SIGNATURE` repair class alone cannot authorize it.
- The migration accepts one exact frozen source block, unique trusted imports,
  exact authority-name counts, and an executable `inspect.signature(...)` AST
  call inside that block. Text inside a string or any modified/ambiguous adapter
  is a no-op.
- The replacement preserves the existing narrow `TypeError` fallback and all
  manual exports. It changes only the host helper invocation protocol.
- The repair is registered as `SYNTACTIC`: it introduces no numbers and changes
  no scientific choice. This classification does not claim byte-for-byte
  runtime equivalence—the old helper call always failed under the current
  variadic signature, while the direct call can now execute. That is the intended
  host API compatibility correction.
- New generated code must use the stable API directly. The migration is
  compatibility debt and can be removed after every sealed candidate containing
  the obsolete adapter has completed or retired.

Two broader approaches were rejected and reverted before this patch: relaxing
the dynamic-namespace coercion proof, and implementing a general AST interpreter
for reflection adapters. Both expanded the control plane beyond the defect.

## Archived E3 read-only probe

Candidate:

`research_output/_diagnostic_e3_8317_fresh_ceb00f2_20260716T072600Z/E3_kdigo_gradient/aware/run_20260716T072721_7fd5c5/steps/02_exposure_derivation_and_qc_figure/.quarantine/concept_draft.py`

Result without provider or sandbox execution:

```json
{
  "before_reasons": [
    "host_helper_runtime_introspection",
    "lossy_numeric_coercion"
  ],
  "repair_names": [
    "legacy_publication_helper_adapter_v1"
  ],
  "after_reasons": [],
  "changed": true
}
```

This probe did not mutate the archived run, consume a provider call, reset a
receipt, or spend the final logical repair. The remaining scientific-semantic
figure issue is intentionally left for the Agent's last authorized repair.

## Verification

- focused migration/alias/inert-text tests: `19 passed`;
- preflight, coercion, repair registry and meta-generalization shard:
  `412 passed`;
- typed repair-reason tests: `17 passed`;
- Ruff and Black: passed;
- no API call and no run mutation.

The prompt change intentionally changes the prompt-pack/engine authority digest.
Any later resume must start in a fresh process and revalidate the failed step;
completed steps and old monotonic provider receipts must not be reset.

## Next action

Submit this small batch for independent Claude review. Only after ACCEPT should
the same E3 run resume its current figure child. The exact legacy migration must
run first with zero provider calls; the final LLM repair remains reserved for
the distinct semantic availability issue.
