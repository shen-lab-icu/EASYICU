# CAP-PREDVAL-V4 train-only model-fit receipt

Date: 2026-08-17
Task: `CAP-PREDVAL-V4`
Branch: `codex/cap-predval-v4-fit-receipt-20260817`
Worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-cap-predval-v1-20260817`
Base HEAD: `5d265892e7aa2e2bd90a1293add68a2453259bb1`

## Decision

Add one narrow experimental producer before connecting prediction fitting to
Planner or EvidenceStore. The producer accepts only a host-issued immutable
`LoadedTypedInput` and a frozen declaration. It owns one deterministic v1
pipeline: numeric median imputation, standardization and L2 logistic
regression, with all three fitted on the declared training subjects only.

The owner chooses no cohort, subject split, outcome, feature roster, model
identifier, regularization coordinate or threshold. Those coordinates must be
predeclared. It performs no feature selection, hyperparameter search,
cross-validation, Planner selection, EvidenceStore registration, alias
publication, numeric-claim registration or paper authorization.

## Reproduced failing baseline

The new focused suite was added before implementation and run against the V3.1
base. Collection failed as expected:

```text
ModuleNotFoundError: No module named
'easyicu.research_agent.contracts.prediction_model_fit'
```

This established that the typed fit contract and governed producer did not
exist at the base commit.

## Closed behavior

- The public fit API accepts only `source_input` and `spec`; it accepts no path,
  DataFrame, caller receipt, model artifact or prediction table.
- The declared unit id must equal the typed input's verified row-identity
  column. Units are unique, v1 permits one row per subject, and a subject may
  not cross train/evaluation splits.
- V1 accepts exactly the declared training and evaluation labels. Empty
  partitions, invalid binary outcomes and single-class training data fail with
  stable owner diagnostics.
- Features must be real numeric columns. Missing values are allowed for median
  imputation, but infinity, boolean/string features and a feature entirely
  missing in training fail closed.
- Median imputation, scaling and L2 logistic regression are fitted only on the
  training matrix, then the frozen state is applied to all rows.
- The immutable bundle joins canonical model JSON, canonical prediction CSV,
  an Arrow payload and a self-digesting `analysis_only` receipt. It binds the
  exact typed-input receipt/artifact/frame/row identity, fit contract, train and
  evaluation subject sets, split assignment, model bytes, prediction bytes and
  installed package versions.
- The model JSON contains all medians, means, scales, coefficients and
  intercept needed to reconstruct every probability without hidden estimator
  state.
- Revalidation repeats the complete fit and rejects payload, model, receipt,
  source or contract drift.
- The generated prediction table passes the existing deterministic V3
  prediction-validation owner, but no production route or EvidenceStore bridge
  was added in this slice.

## Verification

- Failure-first baseline: missing typed contract/owner reproduced at
  collection.
- V4 owner and adversarial suite: 22 passed.
- Typed-input SDK, V3/V3.1 prediction validation/provenance/EvidenceStore
  bridge, capability inventory, package directions, module graph and static
  architecture: 133 passed in the final exact focused matrix.
- Prediction/dynamic capability registry, scientific action catalog, static
  analysis-pattern leakage checks and dynamic prediction kernel: 60 passed.
- Capability inventory audit: OK.
- Research-agent module graph: 543 modules, 2,126 edges, 0 cycles.
- Targeted Ruff format/lint and `git diff --check`: passed.
- No full exact-head CI was run because this is an isolated experimental
  capability slice, not an E1 completion, freeze, merge, release or formal
  experiment checkpoint.
- The active Figure 2 integration worktree was not edited.

## Remaining gates

This is a synthetic, single-estimator host contract, not clinical or external
validation. It supports only subject-level, one-row-per-subject binary models
with numeric median imputation, standardization and one predeclared L2 logistic
fit. There is no external estimator oracle, real ICU dataset run, tuning,
cross-validation, categorical encoding, interaction handling, temporal
landmark fit, uncertainty estimate or transport validation.

The bundle is not yet a production artifact producer. It does not materialize
the V3.1 seven-role EvidenceStore lineage, bind a clean Git/source-tree and
environment-lock runtime identity, or let the existing authority bridge derive
its model/prediction records directly from the sealed bundle. Those are the
next prerequisites before any Planner or paper-facing discussion.
