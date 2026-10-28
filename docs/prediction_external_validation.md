# Frozen-model external validation

`prediction.external_validation` is a deterministic, analysis-only supporting
action for `prediction_model` studies. It accepts a fitted **numeric logistic
model** with its development-fitted median imputation and standardization.
It does not refit the development model, select predictors or deserialize
executable model objects. Other estimator/preprocessing families must supply a
separately reviewed implementation; this action cannot approximate them.

The development and validation databases use the existing canonical keys:
`miiv`, `mimic`, `eicu`, `aumc`, `hirid`, `sic`. A declaration names one
development database and one or more distinct validation databases.

## Before reviewing the validation step

1. Finish development without consulting validation-cohort data. Supply the
   exact fitted coefficients, intercept, imputation medians, scaling centers
   and scales as `FrozenLogisticModel` / `FrozenPredictor` from
   `research_agent.contracts.prediction_validation`. Include the development
   source digest, outcome meaning, and observed development predictor means,
   standard deviations and missing fractions for case-mix comparison.
2. Obtain predictor and outcome meanings from
   `replication.metrics.transport_concept_meaning` using the development
   cohort's verified column bindings and source capability profile. This
   representation includes units, aggregation, observation window,
   representation and database-specific mortality meaning.
3. Serialize the complete model to JSON. The UTF-8 bytes of that JSON are the
   `frozen_model_sha256` reviewed in `ExternalValidationSpec`. Preserve those
   exact bytes in the single `model_json` cell of a one-row typed table
   `table:frozen_prediction_model`. The table binding separately verifies its
   physical file digest. The existing development runner does not silently
   create or choose this supplied portable model.
4. Bind an independently materialized cohort per target database as
   `dataset:external_validation_<database>`. Each cohort needs its existing
   verified materialized authority and column metadata. Provide the model's
   declared subject identifier column with exactly one row per subject.
   Independent-subject DeLong intervals cannot be substituted for clustered
   repeated-stay uncertainty.

The frozen payload and its digest must exist before the external step is
approved. A development run may therefore precede a separately reviewed
external-validation plan. `planning.prediction_timing.build_external_validation_step`
constructs the exact step from a reviewed spec; ordinary plan review and typed
artifact binding remain required. The optional hub graph shows the supplied
model/cohort prerequisites explicitly and is metadata, not execution approval.

Progressive materialization carries `external_validation_spec` on this action's
secondary step only, with a compact JSON-text wire encoding. The compiler
validates it and places its canonical JSON inside
`method="prediction_external_validation(<JSON spec>)"` on the reviewed
`AnalysisStep`, following the existing bootstrap-option serialization pattern.
Classic plans use that exact method serialization. The runtime parser recovers
the typed declaration; the sealed schema remains byte-for-byte unchanged, and
the complete declaration participates in the existing method seal/resume identity.

## Execution boundary

`execution.runners.prediction_validation_executor.run_external_validation_step`
loads the aggregate model table first. `methods.prediction_validation.run_external_validation`
verifies the exact model digest and reviewed roster, then records `model_frozen`
before invoking intake. This order matters: materialized-authority verification
itself may read validation file bytes to verify their digest. Validation reads
carry the frozen model digest, source digest and sequence in the aggregate
provenance record; the first event binds the complete spec digest.

The source gate asks the existing database capability profile and verifies the
materialized cohort's source database. Physical columns cannot override an
unavailable dictionary recipe. An exact semantic match selects each predictor
or outcome among auxiliary columns; unresolved or duplicate matches refuse.
Harmonization uses those bindings, never column-name guessing. Missing or entirely unobserved predictors refuse with
`predictor_unavailable`; outcomes have a separate typed refusal. No predictor is
dropped and no validation-fitted imputer or scaler is substituted.

Changed predictor or outcome meanings refuse with
`meaning_change_requires_confirmation`. The researcher must review a
`confirmed_changes` entry containing the exact database, concept, development
meaning, validation meaning and explanation. A different outcome or predictor
meaning is marked `changed_estimand` in the result; confirmation never restores
an equivalence that the source does not support.

## Aggregate outputs and interpretation

- `external_validation.csv`: one untouched transport row per database, plus a
  separately identified updating row when requested. Discrimination uses a 95%
  DeLong logit interval; calibration-in-the-large uses an offset logistic fit
  with slope one, separately from the jointly fitted calibration slope/intercept.
  Brier scores, counts, event rates and estimation statuses are retained.
- `external_calibration.csv` and `external_calibration*.png`: a four-df cubic
  logistic spline over a fixed 41-point grid. Sparse/constant predictions keep
  an explicit non-estimable status table and labeled figure rather than a
  fabricated curve. There is one declared composite figure and one figure per
  database.
- `external_decision_curve.csv`: net benefit from thresholds 0.05 to 0.50 in
  steps of 0.05, with treat-all and treat-none references. This fixed sweep does
  not select a clinical operating threshold.
- `external_case_mix.csv`: signed validation-minus-development standardized
  mean differences using pooled observed variances, plus source-specific
  missing fractions. A nonzero shift with zero pooled variance is explicitly
  non-estimable.
- `external_validation_provenance.json` and `external_validation_tripod.md`:
  aggregate digest/order evidence, exact confirmed changes, claim boundary and
  TRIPOD+AI item-to-output mapping. The checklist is a reporting map, not a
  declaration that every manuscript requirement has been met.

Optional `intercept_only` and `logistic` recalibration fit only an updating
intercept or intercept/slope. They do not change the transport result.
Their performance is **apparent on the recalibration cohort**, not independently
validated performance after updating. All artifacts are aggregate-only, and
the action grants no paper authorization or transport claim beyond the named
cohorts and their declared meanings.
