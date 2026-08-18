# CAP-PREDVAL-V5 sealed-fit EvidenceStore bridge

Date: 2026-08-17
Task: `CAP-PREDVAL-V5`
Branch: `codex/cap-predval-v5-fit-evidence-20260817`
Worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-cap-predval-v1-20260817`
Base HEAD: `6db2cc44787282b29645de2d2f1054e05fb4d236`

## Decision

Connect the V4 train-only fit owner to the existing V3.1 analysis-only
EvidenceStore bridge without exposing loose model, prediction, receipt, seal or
lineage inputs. The new composite owner accepts the original immutable typed
input, the sealed V4 bundle, matching fit and validation declarations, and an
exact three-record runtime-authority subset already present in EvidenceStore.

The bridge revalidates the complete V4 fit before any fit-owned artifact is
written. It then materializes the four fit-owned roles, constructs the closed
seven-role V3.1 lineage internally, and delegates metric recomputation and
analysis registration to the existing prediction-validation authority owner.

## Reproduced failing baseline

The focused V5 suite was added before implementation and run against the V4
base. Collection failed as expected:

```text
ModuleNotFoundError: No module named
'easyicu.research_agent.authority.prediction_model_fit_evidence'
```

This established that V4 could seal a fit but had no governed route to
materialize that fit into the V3.1 EvidenceStore lineage.

## Closed behavior

- The public API accepts only `evidence_store`, the host-issued
  `LoadedTypedInput`, fit declaration, sealed V4 bundle, matching validation
  declaration, the runtime-authority subset, and fit/validation step ids.
- It accepts no caller-supplied prediction bytes, model bytes, fit or validation
  receipt, host seal, artifact bindings, seven-role lineage, alias or claim.
- Fit and validation coordinates must agree exactly for unit, subject, split,
  outcome, probability, evaluation split and analysis unit.
- Complete V4 recomputation runs against the same immutable typed input before
  materialization. Source, contract, prediction, model, payload or receipt drift
  therefore fails before the bridge writes a fit-owned record.
- Runtime authority is a closed, ordered subset of exactly code snapshot,
  environment lock and runtime receipt. All three records must already exist,
  match their declared digests and producer run, remain current in the store,
  and reconcile to the runtime identity.
- The bridge derives canonical source-projection, subject/split, prediction and
  model-envelope bytes. The model envelope carries the exact fit declaration,
  fit receipt, typed-input consumption receipt and reconstructable model state.
- Evidence identifiers are deterministic over producer run, fit receipt and
  fit step. An exact retry is idempotent without conflating distinct fit steps.
- The internally built lineage uses exactly the canonical seven roles. The
  existing V3.1 owner rereads current stored bytes, verifies cohort and split
  semantics, recomputes metrics and registers one `analysis_only` bundle.
- Reloading EvidenceStore preserves validation. Later byte drift in any of the
  four fit-owned roles fails current-store validation.
- The route publishes no aliases, numeric claims or scientific claims and
  grants neither Planner selection nor paper authority.

## Verification

- Failure-first baseline: missing V5 authority module reproduced at collection.
- V5 composite owner/adversarial suite: 10 passed.
- V5 + V4 + V3.1 vertical contract suite: 50 passed.
- Typed-input SDK, fit owner, prediction-validation owner/provenance/bridge,
  inventory governance, package directions, module graph and static
  architecture: 143 passed in the final exact focused matrix.
- Prediction/dynamic registry, scientific-action catalog, analysis-pattern
  leakage auditor and dynamic prediction kernel: 60 passed.
- Capability inventory audit: OK.
- Research-agent module graph: 544 modules, 2,135 edges, 0 cyclic modules and
  0 cyclic strongly connected components.
- Targeted Ruff format/lint and `git diff --check`: passed.
- No full exact-head CI was run because this remains an isolated experimental
  capability slice, not an E1 completion, freeze, merge, release or formal
  experiment checkpoint.
- The active Figure 2 integration worktree was not edited.

## Remaining gates

This bridge does not capture or approve a live clean source tree, environment
lock or runtime receipt. It consumes three exact records that another host
authority has already registered and verifies their closed identity before any
fit materialization.

The complete fit is recomputed at initial registration. After persistence, the
existing EvidenceStore immutability and V3.1 validator protect current bytes,
closed lineage, cohort/split semantics and recomputed metrics; V5 does not yet
refit the estimator from persisted source/model evidence during reload.

The evidence is synthetic and single-estimator only. There is no real ICU data
run, external estimator oracle, tuning, cross-validation, categorical support,
temporal landmark integration, uncertainty estimate or transport validation.
The route remains outside Planner/runtime selection and cannot support a paper
claim. The next capability gate is independent review plus either persisted
full-fit revalidation/runtime capture or an external-oracle and real-data
validation slice; Planner wiring is deliberately later.
