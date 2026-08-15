# Web E1 data-package execution readiness

- Date: 2026-08-14
- Branch: `fix/pi-workspace-review-20260809`
- Code checkpoint: `d9c9ca9`
- Scope: development Web E1 only; no formal Canonical9 provider batch was started.

## Outcome

The registered-package review now emits a path-free, result-blind execution-readiness receipt before Plan generation. The receipt proves only host-owned capabilities and aggregate counts:

- the typed adult eligibility denominator;
- the exact source-bound patient-grouping authority required for cluster-robust analysis;
- outcome event-time availability and the governed `death_time` materialization;
- exposure canonical-time availability;
- ICU observation-duration units for a 24-hour landmark analysis;
- the readmission indicator needed by the prespecified non-readmission sensitivity.

Missing required coordinates block Plan generation with stable reason codes. The projection contains no patient rows, private mapping paths, event counts, rates, comparisons, or effect estimates.

## Real E1 evidence

The current ordinary Web E1 StudyContext `study_adf40bd3133d3490` at revision 4 was rebuilt from the registered MIMIC-IV export:

- review status: `ready_for_plan`;
- eligible adult ICU-stay denominator: `94,458`;
- runtime readiness: `ready`;
- required findings: none.

This closes the prior false Web blocker that treated available host authority as absent. It does not authorize a scientific Plan or analysis by itself.

## Focused verification

- data-package review and Pi workflow: 18 passed;
- package dependency directions: 7 passed;
- Pi tool, historical snapshot, and replay contracts: 4 passed;
- Ruff, `git diff --check`, and architecture ratchet: passed with no lower-is-better regression.

One unrelated pre-existing route snapshot mismatch remains: the live router includes presentation-pin and child-job archive routes that the static route snapshot has not yet registered. It was not introduced or hidden by this checkpoint.

## Next action

Restart the Web server at this exact checkpoint, finish the E1 StudyContext in the ordinary Pi conversation, rerun the revision-bound PubMed search, generate a reviewable Plan, and execute only after explicit Plan approval.
