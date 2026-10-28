"""Target-trial skeleton projected from the signed clone-censor-weight owner.

Protocol/mapping, eligibility, Table 1, weight models, diagnostics, estimates
and sensitivities are stages of ONE executor transaction. Splitting them into
independent executable steps would promise owners that do not exist. The
reviewable design spells out that sequence; products retain their real owner.
E-values are in the suite's reporting summary, not a separate table.
"""

from __future__ import annotations

from ...authority.target_trial_runtime import TARGET_TRIAL_FIGURE_METHOD
from ..progressive_contract import ProgressiveSkeletonStep
from .contract import TARGET_TRIAL_FAMILY_ID, FamilyPlanSpec, FamilySpecRequest
from .landmark_categorical_template import FamilySkeletonDraft
from .longitudinal_template import OutlineBinder, append_report, checked_request, draft, outputs, preparation, ref


def build_target_trial_skeleton(
    request: FamilySpecRequest, spec: FamilyPlanSpec, *, bind_outline: OutlineBinder | None = None,
) -> FamilySkeletonDraft:
    request = checked_request(request, spec, TARGET_TRIAL_FAMILY_ID)
    owner = request.sealed_target_trial
    assert owner is not None
    steps = preparation(request, list(owner.required_columns))
    steps.append(ProgressiveSkeletonStep(
        step_id="target_trial_suite", module_id="custom_analysis", planned_analysis_role="primary",
        objective=(
            "Apply the signed protocol and source mapping, eligibility flow and baseline table by observed "
            "strategy adherence. Clone, censor and weight; audit positivity, weights, ESS and covariate balance. "
            "Report weighted risk curves, horizon RD/RR, bootstrap intervals, truncation and unweighted "
            "sensitivities, and the E-value in the reporting summary."
        ),
        depends_on=["cohort_accounting", "measurement_audit"], raw_inputs=list(owner.required_columns),
        product_inputs=[ref("cohort_accounting", "artifact:analysis_cohort")],
        custom_method=owner.plan_method, outputs=outputs(owner.analysis_plan_outputs),
    ))
    steps.append(ProgressiveSkeletonStep(
        # This is the suite's six-source renderer, not the generic four-source
        # visualization module. bind_plan supplies its signed consumption rules.
        step_id="visualization", module_id="custom_analysis", planned_analysis_role="auxiliary",
        objective="Render the signed protocol, eligibility, weighted risk curves, balance and sensitivity panels from their exact suite tables.",
        depends_on=["target_trial_suite"],
        product_inputs=[ref("target_trial_suite", product) for product in owner.figure_input_products],
        custom_method=TARGET_TRIAL_FIGURE_METHOD, outputs=outputs([owner.figure_product]),
    ))
    limitations = "Causal interpretation requires consistency, exchangeability, positivity and correctly specified weight models; analysis only."
    append_report(steps, limitations=limitations)
    return draft(
        request, spec, steps,
        estimand=f"Per-protocol risk difference and risk ratio for {owner.outcome_label} by day {owner.endpoint_horizon_days}, comparing {owner.initiate_label} with {owner.defer_label}.",
        time_zero=f"{owner.time_zero_hours} hours after ICU admission; eligibility precedes cloning.",
        observation_window=f"Start treatment within {owner.grace_period_hours} hours after time zero; follow to ICU-admission day {owner.endpoint_horizon_days}.",
        primary_method="Signed clone-censor-weight suite with patient/stay bootstrap as declared by its authority.",
        assumptions=[limitations, "Confounder coding and pre-time-zero timing are fixed by the approved protocol."],
        cannot_prove="The data cannot establish exchangeability or eliminate residual confounding; no publication authorization.",
        reviewable_plan=[
            f"Specify eligibility ({owner.eligibility_label}), strategies, grace period, time zero, outcome, causal contrast and confounder rationale in the approved protocol.",
            "Map protocol concepts to the signed source columns and audit measurement completeness before execution.",
            "Apply eligibility at time zero and report flow and baseline characteristics by observed initiation in the grace period.",
            "Fit treatment and ICU-exit censoring weight models; inspect positivity, weight distributions, ESS and covariate balance.",
            "Estimate weighted risk curves and horizon RD/RR with the declared bootstrap; compare truncation and unweighted sensitivity estimates and report the E-value.",
            "Render the suite's exact figure tables and report the emulation assumptions and analysis-only boundary.",
        ], bind_outline=bind_outline,
    )


__all__ = ["build_target_trial_skeleton"]
