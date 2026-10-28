"""Time-varying Cox skeleton with the existing analysis-only owner boundary.

Acquisition constructs the counting-process panel before execution. The
signed executor verifies that receipt and publishes estimates and input audit
together. It has no PH, absolute-risk, competing-event or figure product.
Those analyses are omitted, never substituted with a static survival suite.
"""

from __future__ import annotations

from ..progressive_contract import ProgressiveSkeletonStep, ProgressiveTableOneVariable
from .contract import TIME_VARYING_SURVIVAL_FAMILY_ID, FamilyPlanSpec, FamilySpecRequest
from .landmark_categorical_template import FamilySkeletonDraft
from .longitudinal_template import OutlineBinder, append_report, checked_request, draft, outputs, preparation, ref


def build_time_varying_survival_skeleton(
    request: FamilySpecRequest, spec: FamilyPlanSpec, *, bind_outline: OutlineBinder | None = None,
) -> FamilySkeletonDraft:
    request = checked_request(request, spec, TIME_VARYING_SURVIVAL_FAMILY_ID)
    owner = request.sealed_time_varying
    assert owner is not None
    steps = preparation(request, list(owner.required_columns))
    if owner.specification.baseline_columns:
        steps.insert(1, ProgressiveSkeletonStep(
            step_id="baseline_context", module_id="table_one", planned_analysis_role="auxiliary",
            objective="Describe the declared baseline covariates by hospital outcome with standardized differences only; no grouping by a future exposure maximum.",
            depends_on=["cohort_accounting"],
            raw_inputs=[owner.outcome_column, *owner.specification.baseline_columns],
            table_one_group_by=owner.outcome_column, table_one_mode="descriptive_smd_only",
            table_one_variables=[ProgressiveTableOneVariable(
                name=name, summary="count_percent" if name in owner.specification.baseline_categorical_encodings else "median_iqr",
            ) for name in owner.specification.baseline_columns],
        ))
    steps.append(ProgressiveSkeletonStep(
        step_id="time_varying_cox", module_id="custom_analysis", planned_analysis_role="primary",
        objective=(
            "Verify the source-constructed counting-process intervals and input audit, retaining early events "
            "and the unmeasured state. Update the running maximum only from preceding direct measurements "
            "in hours 0–24, then freeze it. Fit R survival Cox with Efron ties and patient-cluster-robust variance."
        ),
        depends_on=["cohort_accounting", "measurement_audit"], raw_inputs=list(owner.required_columns),
        product_inputs=[ref("cohort_accounting", "artifact:analysis_cohort")],
        custom_method=owner.plan_method, outputs=outputs(owner.plan_outputs),
    ))
    limitations = "PH validation, absolute risk, competing events and visualization are absent: this signed owner produces only estimates, input counts and its receipt. Analysis only."
    append_report(steps, limitations=limitations)
    return draft(
        request, spec, steps,
        estimand="Descriptive hospital-death hazard association with the time-updated observed exposure and an unmeasured-state indicator.",
        time_zero="ICU admission, with early events retained.",
        observation_window="Direct measurements during ICU hours 0–24; hospital follow-up until death or discharge.",
        primary_method="Source-verified counting-process Cox, Efron ties and patient-cluster-robust variance.",
        assumptions=["No future measurements enter a risk interval; baseline coding and patient grouping are verified.", "PH validation remains outstanding; informative measurement and residual confounding remain."],
        cannot_prove="No causal effect, validated proportional hazards or publication readiness.",
        reviewable_plan=[
            "Account for every bound stay, including early events, and describe baseline covariates by hospital outcome.",
            "Audit source measurements and missingness; no future-value filling is permitted.",
            "Acquire counting-process intervals with hospital follow-up and the unmeasured-state term; verify the construction receipt against the bound cohort.",
            "Report interval and event counts from the input audit; the receipt records exposure timing rules rather than a separate timing-summary table.",
            "Fit the declared time-varying Cox model with patient-cluster-robust inference and stop on invalid input or fit failure.",
            "Report estimates and input audit with the PH, absolute-risk, competing-event and figure omissions explicitly stated.",
        ], bind_outline=bind_outline,
    )


__all__ = ["build_time_varying_survival_skeleton"]
