# CAP-PREDVAL-V3.1 semantic-lineage hardening

Date: 2026-08-17
Task: `CAP-PREDVAL-V3.1`
Branch: `codex/cap-predval-v3-1-semantic-lineage-20260817`
Worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-cap-predval-v1-20260817`
Base HEAD: `4cc0a79da4a9abe587e09a5ef09c0c8f5da1d094`

## Decision

Harden V3 before adding another capability slice. The public EvidenceStore
registration route must recompute from current stored bytes and must not accept
a caller-supplied receipt or host seal. The cohort and subject-disjoint split
artifacts must also reconcile semantically with the prediction table rather
than merely exist under matching ids and digests.

The exact-source parser, deterministic result/receipt recomputation, semantic
lineage checks and seal construction now live in one dependency-neutral
prediction-validation owner. The execution incubator remains a compatibility
adapter, and the authority bridge imports the shared owner rather than
reverse-importing execution or copying parser policy.

## Reproduced failing baseline

Three failure-first tests were run against V3 before implementation:

```text
test_public_bridge_does_not_accept_caller_receipt_or_seal FAILED
test_bridge_rejects_cohort_unrelated_to_prediction_subjects FAILED
test_bridge_rejects_split_unrelated_to_prediction_subjects FAILED
```

The first failure showed both `receipt` and `validation_seal` in the public
signature. The other two showed that a cohort or split artifact containing
only `subject-1` was accepted for a 16-subject prediction table.

## Closed behavior

- `register_prediction_validation_analysis_artifact` now accepts only the
  EvidenceStore, spec, closed lineage and validation step id.
- It resolves all seven current EvidenceStore artifacts, recomputes the full
  receipt from the exact prediction-table bytes, reconciles the other semantic
  tables and constructs the runtime-bound seal internally.
- Cohort authority requires one canonical row per subject and an exact subject
  set match to the prediction table.
- Split authority requires one canonical assignment per subject, the exact
  prediction subject set and the same split label for every subject.
- Missing columns, missing/extra subjects, duplicates and wrong assignments
  fail with stable `prediction_validation_lineage_cohort_mismatch` or
  `prediction_validation_lineage_split_mismatch` diagnostics.
- Reload validation repeats the same current-byte recomputation and rejects a
  receipt or seal that differs from the host result.
- The analysis bundle remains `analysis_only`: no alias, numeric claim,
  scientific claim, Planner selection or paper authority is added.

## Verification

- Failure-first baseline: 3 expected failures reproduced.
- Prediction owner, exact-source provenance and V3.1 bridge: 56 passed.
- Focused owner/provenance/bridge, EvidenceStore, scientific-claim authority,
  capability inventory, package-direction and module-graph checks: 157 passed.
- Capability inventory audit: OK.
- Research-agent module graph: 541 modules, 2,121 edges, 0 cyclic SCCs.
- Targeted Ruff format/lint and `git diff --check`: passed.
- No full exact-head CI was run because this remains an isolated experimental
  hardening slice, not a freeze, merge, release or formal-experiment checkpoint.
- The active Figure 2 worktree was not edited or used for these checks.

## Remaining gates

V3.1 does not prove how the model or preprocessing was fit. The model artifact,
source snapshot and environment lock are still exact byte-bound records but
have no governed training-only fit/preprocessing receipt. Cohort reconciliation
currently proves the exact subject set, not attrition reasons or encounter-level
row provenance. No production workflow creates the seven upstream records,
and there is no independent human review or release/full-CI evidence. Planner,
runtime selection, aliases and paper-facing claims remain forbidden.

The next capability slice should therefore be a typed model-fit/preprocessing
producer receipt, not Planner activation or additional metric families.
