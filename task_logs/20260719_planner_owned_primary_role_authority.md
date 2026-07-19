# Planner-owned primary-role authority bundle

Date: 2026-07-19

## Objective

Make the Planner's explicit scientific role the only authority for headline
results.  The host may validate, render, and register evidence, but must not
infer a primary result from method names, filenames, aliases, or prose.

## Implemented

- Added the typed `planned_analysis_role` contract (`primary`, `secondary`,
  `sensitivity`, `auxiliary`) to plans and step records.
- Required every LLM-planned step to declare its role explicitly; host-created
  support steps remain auxiliary.
- Enforced at most one primary step and required that primary step to declare a
  typed non-rendering scientific product.
- Added strict outer-record plus embedded-request role verification.  Resume
  rejects role drift, scientific-signature drift, and forged completed records.
- Preserved late primary steps through plan capping without method-family
  keyword inference.
- Bound primary-effect selection, deterministic robustness, article headline
  roles, and publication figures to the verified primary lineage.
- Required auxiliary figure credit to follow a unique typed product edge and,
  at runtime, a digest-bound resolved-input receipt.
- Removed publication-figure filenames and aliases as scientific ownership
  signals.  A sensitivity table named `primary_association` cannot become the
  headline result.
- Made article and figure readiness follow the Planner's final `analysis_type`;
  question-text inference remains only the pre-plan prompt prior.
- Kept noncausal treatment-response characterization in the descriptive family.

## Fail-closed regressions

- Multiple primary owners are rejected.
- Empty, untyped, figure-only, log/report/code/test-only primary steps are
  rejected.
- Outer/embedded role mismatch invalidates resume authority.
- A sensitivity step cannot cover a headline article role.
- A declaration-only auxiliary figure cannot inherit primary credit.
- A global decoy figure contract cannot satisfy primary article coverage.
- Misleading sensitivity aliases are excluded from publication-figure input.
- Association, prediction, survival, phenotyping, and causal headline roles all
  require the primary lineage.

## Validation

- Focused role/study/figure/plan authority: 109 passed.
- Main pipeline: 267 passed.
- Resume: 82 passed.
- Other modified-contract suites: 316 passed.
- Meta, capability, characterization, package boundaries, module graph, and
  architecture measurement: 147 passed after the intentional golden refresh.
- Golden characterization ran green twice consecutively after refreshing only
  the validator-finding count/hash.  The added warning is intentional: the
  final phenotyping plan is now audited against the phenotyping contract rather
  than silently falling back to a descriptive contract inferred from prose.
- Ruff, Black, and `git diff --check`: clean.

## Remaining before fresh experiments

- Finish the display/mock protocol cleanup so fixture conveniences cannot leak
  into canonical scientific authority.
- Finish the bounded package/legacy-shim inventory and delete only proven
  retired compatibility surfaces.
- Run the segmented release regression and record the clean freeze manifest.
- Freeze the shared engine, then run fresh E3/H2/E2 and held-out tasks.  Do not
  resume or preserve the old diagnostic runs merely for compatibility.
