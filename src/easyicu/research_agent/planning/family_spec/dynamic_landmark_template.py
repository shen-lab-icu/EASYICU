"""Analysis-only dynamic landmark composition using the registered kernels.

Features, labels and metrics have reviewed method owners. Patient splitting,
the exact sklearn Pipeline and visualization remain Coder work, as declared
by analysis_method_suite.dynamic_prediction. This template grants no static
prediction executor authority and promises no CIs, bootstrap validation,
decision curve or subgroup product without a dynamic result owner.
"""

from __future__ import annotations

from ..progressive_contract import ProgressiveSkeletonStep
from .contract import DYNAMIC_LANDMARK_FAMILY_ID, FamilyPlanSpec, FamilySpecRequest
from .landmark_categorical_template import FamilySkeletonDraft
from .longitudinal_template import OutlineBinder, append_report, checked_request, draft, outputs, preparation, ref


def build_dynamic_landmark_skeleton(
    request: FamilySpecRequest, spec: FamilyPlanSpec, *, bind_outline: OutlineBinder | None = None,
) -> FamilySkeletonDraft:
    request = checked_request(request, spec, DYNAMIC_LANDMARK_FAMILY_ID)
    coordinates = request.dynamic_landmark
    assert coordinates is not None
    steps = preparation(request, [request.identity_column, request.outcome, *coordinates.source_columns])
    steps.extend([
        ProgressiveSkeletonStep(
            step_id="landmark_features", module_id="custom_analysis", planned_analysis_role="auxiliary",
            objective=(
                f"Use build_landmark_feature_matrix on the bound long trajectory: landmarks {coordinates.landmark_hours} h, "
                f"lookback {coordinates.lookback_hours:g} h, concepts {coordinates.feature_concepts}, aggregations {coordinates.aggregations}. "
                "Use [landmark-lookback, landmark); retain missing windows and audit post-landmark leakage."
            ),
            depends_on=["cohort_accounting", "measurement_audit"],
            raw_inputs=list(coordinates.feature_concepts),
            product_inputs=[ref("cohort_accounting", "artifact:analysis_cohort")],
            custom_method="build_landmark_feature_matrix",
            outputs=outputs(["table:landmark_features"]),
        ),
        ProgressiveSkeletonStep(
            step_id="horizon_labels", module_id="custom_analysis", planned_analysis_role="auxiliary",
            objective=(
                f"Use attach_landmark_outcomes with {coordinates.event_time_column}, {coordinates.followup_end_column} "
                f"and future horizons {coordinates.horizon_hours} h. Preserve eligible_at_landmark and horizon_observed; "
                "prevalent events and unobserved horizons have missing labels, never a non-event label."
            ),
            depends_on=["landmark_features"], raw_inputs=[coordinates.event_time_column, coordinates.followup_end_column],
            product_inputs=[ref("landmark_features", "table:landmark_features")],
            custom_method="attach_landmark_outcomes", outputs=outputs(["table:landmark_labels"]),
        ),
        ProgressiveSkeletonStep(
            step_id="patient_split", module_id="custom_analysis", planned_analysis_role="auxiliary",
            objective=(
                f"Coder: use sklearn GroupShuffleSplit(n_splits=1, test_size={coordinates.validation_fraction}, "
                f"random_state={coordinates.seed}), groups={coordinates.patient_column}; keep every stay, landmark "
                "and horizon of a patient in the same partition and verify no patient overlap before fitting."
            ),
            depends_on=["horizon_labels"], raw_inputs=[coordinates.patient_column],
            product_inputs=[ref("horizon_labels", "table:landmark_labels")],
            custom_method="patient_grouped_landmark_split", outputs=outputs(["table:landmark_split"]),
        ),
        ProgressiveSkeletonStep(
            step_id="model_fit", module_id="custom_analysis", planned_analysis_role="primary",
            objective=(
                "Coder: fit a separate sklearn Pipeline(SimpleImputer(strategy='median'), StandardScaler(), "
                f"LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state={coordinates.seed})) "
                "for each landmark/horizon on observed development labels only. Fit preprocessing inside training; "
                "retain held-out probabilities, stay/patient ids, prediction time and target horizon; stop on one-class or nonconverged fits."
            ),
            depends_on=["patient_split"], product_inputs=[ref("patient_split", "table:landmark_split")],
            scientific_action_id="prediction.dynamic_prediction",
            custom_method="dynamic_prediction", outputs=outputs(["table:landmark_predictions"]),
        ),
        ProgressiveSkeletonStep(
            step_id="landmark_performance", module_id="custom_analysis", planned_analysis_role="secondary",
            objective=(
                f"Use evaluate_landmark_probabilities(calibration_bins={coordinates.calibration_bins}) on held-out "
                "observed labels and probabilities separately by prediction time and target horizon. Report AUROC, "
                "Brier score, observed risk and calibration bins; this kernel supplies no confidence intervals."
            ),
            depends_on=["model_fit"], product_inputs=[ref("model_fit", "table:landmark_predictions")],
            custom_method="evaluate_landmark_probabilities",
            outputs=outputs(["table:horizon_performance", "table:landmark_calibration"]),
        ),
        ProgressiveSkeletonStep(
            step_id="visualization", module_id="visualization", planned_analysis_role="auxiliary",
            objective="Coder: plot discrimination, Brier and calibration by landmark and horizon from the evaluation tables without inventing intervals.",
            depends_on=["landmark_performance"],
            product_inputs=[ref("landmark_performance", product) for product in ("table:horizon_performance", "table:landmark_calibration")],
            outputs=outputs(["figure:time_varying_discrimination"], "figure"),
        ),
    ])
    limitations = "Held-out evaluation only; CIs, resampling validation, decision curves and subgroup products are absent. Coder fitting has no typed host fit/result validator; analysis only."
    append_report(steps, limitations=limitations)
    return draft(
        request, spec, steps,
        estimand=f"Held-out future {request.outcome} risk at the prespecified landmarks and horizons, evaluated by discrimination, calibration and Brier score.",
        time_zero=f"Each prediction landmark {coordinates.landmark_hours} hours after ICU admission.",
        observation_window=f"Look back {coordinates.lookback_hours:g} hours before each landmark; predict the next {coordinates.horizon_hours} hours.",
        primary_method="Patient-grouped holdout with a prespecified median-imputation, scaling and logistic-regression Pipeline per landmark/horizon.",
        assumptions=["Only measurements strictly before the prediction time enter features.", "Patient identity is complete; unobserved horizons are excluded from evaluation, not coded as non-events."],
        cannot_prove="No causal effect, external validity, clinical benefit or publication readiness; no performance confidence intervals.",
        reviewable_plan=[
            "Account for the bound stays and audit measurement availability before constructing the landmark grid.",
            "Separate prediction times, lookback windows and future horizons; retain missing feature windows and prevent post-landmark leakage.",
            "Attach observable horizon labels and retain risk-set and censoring indicators from the reviewed kernels.",
            "Split by patient once across all stays, landmarks and horizons; fit preprocessing and the prespecified Pipeline on development data only.",
            "Evaluate held-out AUROC, calibration and Brier separately by landmark/horizon; no CIs or resampling-validation output is promised.",
            "Plot the owned metric tables and report the analysis-only limit, including absent decision-curve and subgroup analyses.",
        ], bind_outline=bind_outline,
    )


__all__ = ["build_dynamic_landmark_skeleton"]
