"""Digest-bound fixed-landmark survival suite for one continuous exposure.

The binary suite (``LandmarkSurvivalRuntimeAuthority``) contrasts an incident
exposure group with its comparator.  A continuous exposure has no groups to
contrast: this suite models the value one window summary recorded by the
landmark, per one readable step of the source's own scale.  A caller-reviewed protocol
supplies the window, its summary, the endpoint and the adjustment set; the
host then runs one deterministic risk-set build, a descriptive Table 1 and
Kaplan-Meier curves by exposure tertile, the Cox fit per exposure step, a
restricted cubic spline check of that linear term whose rejection makes the
spline's percentile contrasts the result, the proportional-hazards audit
with its interval model, and the composite figure.  None of those
mechanical operations goes through a Coder.

The authority chooses no case science and imports no other authority: the
current-case union imports it.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256
from ..contracts.cohort_product_keys import sole_typed_cohort_input
from ..contracts.figure_plan import DeterministicFigurePanelTemplate
from ..contracts.runtime_outcomes import RuntimeOutcomeContract
from ..schema import AnalysisPlan, AnalysisStep, EndpointSpec

CONTINUOUS_SURVIVAL_AUTHORITY_KIND = "landmark_continuous_survival_suite"
CONTINUOUS_SURVIVAL_PLAN_METHOD = "signed_landmark_continuous_survival_suite"
CONTINUOUS_SURVIVAL_FIGURE_METHOD = "signed_landmark_continuous_survival_figure"
#: The marker line the planning disclosure prints before its JSON coordinates.
#: It does not contain the binary suite's marker, so neither reads the other.
CONTINUOUS_SURVIVAL_SUITE_MARKER = (
    "CALLER-BOUND LANDMARK CONTINUOUS-EXPOSURE SURVIVAL SUITE:"
)
#: The window summaries a materialized numeric exposure is published under.
CONTINUOUS_EXPOSURE_WINDOW_SUMMARIES = ("max", "min", "mean", "first")

_TABLE = r"^table:[a-z][a-z0-9_]{0,79}$"
_DERIVED = r"^[a-z][a-z0-9_]{0,79}$"
_COHORT_INPUT = "table:analysis_cohort"
_COHORT_OWNER_METHOD = "host_materialized_locked_cohort"


class ContinuousSurvivalAuthorityError(ValueError):
    """A plan drifted from the signed continuous-exposure survival suite."""


class LandmarkContinuousSurvivalRuntimeAuthority(BaseModel):
    """Closed fixed-landmark survival suite over one continuous exposure.

    Every field is stated in the signed body: the digest is computed over the
    body as the host wrote it, so a field with a default would digest
    differently once validated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.landmark_continuous_survival_runtime_authority/2"]
    authority_kind: Literal["landmark_continuous_survival_suite"]
    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_method: Literal["signed_landmark_continuous_survival_suite"]
    plan_intent: str = Field(min_length=1)
    plan_outputs: tuple[str, ...]
    development_execution_only_allowed: bool
    exposure_column: str = Field(min_length=1)
    #: Reader words for the exposure and the unit of its scale.
    exposure_label: str = Field(min_length=1, max_length=120)
    exposure_unit: str | None = Field(min_length=1, max_length=40)
    exposure_window_summary: Literal["max", "min", "mean", "first"]
    #: Hours from the time origin the summary read, ending by the landmark.
    exposure_window_hours: tuple[float, float]
    #: The hazard ratio is reported per one step of the source scale, read
    #: from the modelled exposure alone before any model is fitted: the
    #: largest one, two or five times a power of ten within its
    #: interquartile range, or, for a heaped exposure whose interquartile
    #: range is zero, within its 10th-90th percentile range, then its range.
    exposure_increment_rule: Literal["largest_round_step_within_interquartile_range"]
    event_column: str = Field(min_length=1)
    followup_time_column: str = Field(min_length=1)
    endpoint_time_origin: str = Field(min_length=1)
    endpoint_censoring_rule: str = Field(min_length=1)
    landmark_hours: float = Field(gt=0)
    endpoint_horizon_days: float = Field(gt=0)
    analysis_unit_label: str = Field(min_length=1)
    derived_event_column: str = Field(pattern=_DERIVED)
    derived_time_column: str = Field(pattern=_DERIVED)
    adjustment_columns: tuple[str, ...]
    categorical_adjustment_columns: tuple[str, ...]
    table_one_columns: tuple[str, ...]
    estimator: Literal["cox_ph_lifelines_efron"]
    effect_measure: Literal["hazard_ratio_per_exposure_step"]
    uncertainty_method: Literal["wald_95_ci"]
    proportional_hazards_diagnostic: Literal["schoenfeld_residual_test"]
    proportional_hazards_alpha: float = Field(gt=0, lt=1)
    proportional_hazards_policy: Literal["report_only", "block_paper_authorization"]
    #: Fitted in every run; its interval estimates replace the constant one
    #: when the PH test rejects.  The suite therefore needs one cutpoint.
    time_varying_effect_method: Literal["piecewise_time_varying_cox"]
    time_varying_interval_cutpoints_days: tuple[float, ...]
    #: The functional-form check of the linear term: Harrell's three knots at
    #: fixed percentiles of the analysed exposure and a likelihood-ratio test
    #: judged at ``functional_form_alpha``.  When it rejects linearity and the
    #: PH test does not reject, the spline's hazard ratios at the two ends of
    #: ``curve_quantile_range`` relative to the median replace the per-step
    #: estimate as the result.
    spline_knot_quantiles: tuple[float, float, float]
    spline_reference: Literal["median_in_model_population"]
    curve_quantile_range: tuple[float, float]
    curve_points: int = Field(ge=5, le=201)
    functional_form_alpha: float = Field(gt=0, lt=1)
    functional_form_policy: Literal["spline_contrasts_replace_linear_estimate"]
    #: Table 1 and the Kaplan-Meier curves describe value tertiles of the risk
    #: set; no estimate is computed between them.
    descriptive_grouping: Literal["value_tertiles"]
    interpretation: Literal["descriptive_prognostic_association_not_causal"]
    table_one_product: str = Field(pattern=_TABLE)
    risk_set_product: str = Field(pattern=_TABLE)
    km_product: str = Field(pattern=_TABLE)
    cox_product: str = Field(pattern=_TABLE)
    ph_product: str = Field(pattern=_TABLE)
    time_varying_cox_product: str = Field(pattern=_TABLE)
    spline_product: str = Field(pattern=_TABLE)
    measurement_audit_product: str = Field(pattern=_TABLE)
    receipt_product: str = Field(pattern=r"^log:[a-z][a-z0-9_]{0,79}$")
    figure_product: str = Field(pattern=r"^figure:[a-z][a-z0-9_]{0,79}$")

    @model_validator(mode="after")
    def _closed_contract(self) -> "LandmarkContinuousSurvivalRuntimeAuthority":
        window_start, window_end = self.exposure_window_hours
        if not 0.0 <= window_start < window_end <= self.landmark_hours:
            raise ValueError(
                "continuous survival exposure window must close by the landmark"
            )
        if self.landmark_hours / 24.0 >= self.endpoint_horizon_days:
            raise ValueError(
                "continuous survival landmark must precede the endpoint horizon"
            )
        followup_days = self.endpoint_horizon_days - self.landmark_hours / 24.0
        cutpoints = self.time_varying_interval_cutpoints_days
        if (
            not cutpoints
            or tuple(sorted(set(cutpoints))) != cutpoints
            or cutpoints[0] <= 0
            or cutpoints[-1] >= followup_days
        ):
            raise ValueError(
                "continuous survival intervals must be increasing inside follow-up"
            )
        if self.spline_knot_quantiles != (0.10, 0.50, 0.90):
            raise ValueError("continuous survival spline requires frozen 10/50/90 knots")
        if self.curve_quantile_range != (0.10, 0.90):
            raise ValueError(
                "continuous survival curve must span the frozen boundary knots"
            )
        source_columns = (
            self.exposure_column,
            self.event_column,
            self.followup_time_column,
            *self.adjustment_columns,
        )
        if len(source_columns) != len(set(source_columns)):
            raise ValueError("continuous survival source columns must be unique")
        if not set(self.categorical_adjustment_columns).issubset(
            self.adjustment_columns
        ):
            raise ValueError(
                "continuous survival categorical adjustments must be adjusted columns"
            )
        if not set(self.table_one_columns).issubset(self.adjustment_columns):
            raise ValueError(
                "continuous survival Table 1 columns must come from the adjustment set"
            )
        derived = {self.derived_event_column, self.derived_time_column}
        if len(derived) != 2 or derived & set(source_columns):
            raise ValueError(
                "continuous survival derived columns must be distinct from source columns"
            )
        products = self.owned_products
        if len(products) != len(set(products)):
            raise ValueError("continuous survival output products must be unique")
        if self.plan_outputs != products:
            raise ValueError(
                "continuous survival plan outputs must equal the owned products"
            )
        body = self.model_dump(mode="json", exclude={"execution_contract_sha256"})
        if canonical_sha256(body) != self.execution_contract_sha256:
            raise ValueError("continuous survival authority digest mismatch")
        return self

    @property
    def owned_products(self) -> tuple[str, ...]:
        return (
            self.table_one_product,
            self.risk_set_product,
            self.km_product,
            self.cox_product,
            self.ph_product,
            self.time_varying_cox_product,
            self.spline_product,
            self.measurement_audit_product,
            self.receipt_product,
            self.figure_product,
        )

    @property
    def plan_rule_ref(self) -> str:
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.exposure_column,
            self.event_column,
            self.followup_time_column,
            *self.adjustment_columns,
        )

    @property
    def analysis_plan_outputs(self) -> tuple[str, ...]:
        return tuple(
            value for value in self.plan_outputs if value != self.figure_product
        )

    @property
    def figure_input_products(self) -> tuple[str, ...]:
        return (
            self.km_product,
            self.cox_product,
            self.spline_product,
            self.time_varying_cox_product,
            self.risk_set_product,
            self.ph_product,
        )

    def _require_rule_ref(self, step: AnalysisStep) -> None:
        if self.plan_rule_ref not in step.icu_rule_refs:
            raise ContinuousSurvivalAuthorityError(
                "continuous survival step lacks its runtime contract reference"
            )

    def bind_plan(self, plan: AnalysisPlan) -> AnalysisPlan:
        """Compile the signed survival analysis and its source-bound renderer."""

        primary = [
            step for step in plan.steps if step.planned_analysis_role == "primary"
        ]
        if len(primary) != 1:
            raise ContinuousSurvivalAuthorityError(
                "continuous survival authority requires exactly one primary step"
            )
        candidate = primary[0]
        # The run input is already a digest-sealed cohort; the generic host
        # materializer publishes it as the typed input the suite consumes.
        cohort_owner = AnalysisStep.model_validate(
            {
                "step_id": "00_host_bound_analysis_cohort",
                "planned_analysis_role": "auxiliary",
                "intent": "Bind the locked run cohort as the analysis row authority.",
                "inputs": [],
                "expected_outputs": [_COHORT_INPUT],
                "method": _COHORT_OWNER_METHOD,
            }
        )
        bound = candidate.model_copy(
            update={
                "planned_analysis_role": "primary",
                "method": self.plan_method,
                "intent": self.plan_intent,
                "inputs": [_COHORT_INPUT, *self.required_columns],
                "expected_outputs": list(self.analysis_plan_outputs),
                "scientific_capability": None,
                "model_requirements": [],
                "family_primary_result_requirement": None,
                "table_one_spec": None,
                "input_consumption_contracts": [],
                "icu_rule_refs": list(
                    dict.fromkeys([*candidate.icu_rule_refs, self.plan_rule_ref])
                ),
                "runtime_outcome_contract": RuntimeOutcomeContract(
                    owner_ref=self.plan_rule_ref,
                    outcomes=(self.event_column,),
                ),
            }
        )
        figure_owner = AnalysisStep.model_validate(
            {
                "step_id": "02_authority_compiled_survival_figure",
                "planned_analysis_role": "auxiliary",
                "intent": (
                    "Render the signed continuous-exposure survival suite from its "
                    "exact result tables."
                ),
                "inputs": list(self.figure_input_products),
                "expected_outputs": [self.figure_product],
                "method": CONTINUOUS_SURVIVAL_FIGURE_METHOD,
                "input_consumption_contracts": [
                    {"input_key": value, "mode": "all_rows"}
                    for value in self.figure_input_products
                ],
                "icu_rule_refs": [self.plan_rule_ref],
                "figure_panels": [
                    panel.bind(figure_output=self.figure_product).model_dump(
                        mode="json"
                    )
                    for panel in self.figure_panel_templates()
                ],
            }
        )
        return plan.model_copy(update={"steps": [cohort_owner, bound, figure_owner]})

    def figure_panel_templates(self) -> tuple[DeterministicFigurePanelTemplate, ...]:
        """Exact panels of the signed composite figure, by article role.

        Panel b's grammar follows the sealed PH policy at execution: the
        hazard-ratio curve of a model with a constant effect is
        withheld when the assumption is rejected, and the per-unit hazard
        ratios of the prespecified interval model replace it.
        """

        return (
            DeterministicFigurePanelTemplate(
                panel_id="a",
                article_role="temporal_absolute_risk",
                chart_type="kaplan_meier_curve",
                source_products=(self.km_product,),
            ),
            DeterministicFigurePanelTemplate(
                panel_id="b",
                article_role="survival_effect",
                chart_type="hazard_ratio_curve",
                source_products=(
                    self.cox_product,
                    self.spline_product,
                    self.time_varying_cox_product,
                ),
                policy_alternative_chart_types=("time_varying_hazard_ratio_forest",),
            ),
            DeterministicFigurePanelTemplate(
                panel_id="c",
                article_role="cohort_accounting",
                chart_type="cohort_flow",
                source_products=(self.risk_set_product,),
            ),
            DeterministicFigurePanelTemplate(
                panel_id="d",
                article_role="diagnostics",
                chart_type="schoenfeld_plot",
                source_products=(self.ph_product,),
            ),
        )

    def planning_contract_context(self) -> str:
        """Disclose the sealed suite so a planner can name its owner, not re-derive it."""

        coordinates = {
            "sealed_primary_owner": self.plan_method,
            "exposure_column": self.exposure_column,
            "exposure_window_summary": self.exposure_window_summary,
            "exposure_window_hours": list(self.exposure_window_hours),
            "exposure_increment_rule": self.exposure_increment_rule,
            "exposure_unit": self.exposure_unit,
            "event_column": self.event_column,
            "followup_time_column": self.followup_time_column,
            "landmark_hours": self.landmark_hours,
            "endpoint_horizon_days": self.endpoint_horizon_days,
            "adjustment_columns": list(self.adjustment_columns),
            "plan_outputs": list(self.plan_outputs),
        }
        return (
            f"{CONTINUOUS_SURVIVAL_SUITE_MARKER} the single primary step is "
            "owned by the sealed host suite named in sealed_primary_owner; it "
            "must declare exactly the listed source columns and outputs and "
            "carry no model requirement of its own. The host compiles risk-set "
            "accounting, Table 1 and Kaplan-Meier curves by exposure tertile, "
            "the Cox model per exposure step, its spline check, the PH audit with "
            "its interval model and the composite figure from this contract.\n"
            + json.dumps(coordinates, ensure_ascii=False, sort_keys=True)
        )

    def development_execution_only_plan(
        self,
        *,
        research_question: str,
    ) -> AnalysisPlan:
        """Mechanically project this complete authority into one dev-only plan."""

        if not self.development_execution_only_allowed:
            raise ContinuousSurvivalAuthorityError(
                "continuous survival authority does not allow execution-only development"
            )
        draft = AnalysisPlan.model_validate(
            {
                "research_question": str(research_question),
                "analysis_type": "survival",
                "endpoint": self.research_context_endpoint().model_dump(mode="json"),
                "steps": [
                    {
                        "step_id": "01_authority_compiled_survival_suite",
                        "planned_analysis_role": "primary",
                        "intent": self.plan_intent,
                        "inputs": [_COHORT_INPUT],
                        "expected_outputs": list(self.plan_outputs),
                        "method": self.plan_method,
                        "icu_rule_refs": [self.plan_rule_ref],
                    }
                ],
            }
        )
        return self.bind_plan(draft)

    def research_context_endpoint(self) -> EndpointSpec:
        """Project the sealed event/time pair into the shared endpoint contract."""

        return EndpointSpec(
            name=self.event_column,
            kind="time_to_event",
            absence_semantics="absent_row_is_unmeasured",
            levels=[0, 1],
            event_column=self.event_column,
            time_column=self.followup_time_column,
            time_origin=self.endpoint_time_origin,
            censoring_rule=self.endpoint_censoring_rule,
        )

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        """Return the deterministic analysis owner after validating the suite."""

        if len(plan.steps) != 3:
            raise ContinuousSurvivalAuthorityError(
                "continuous survival suite must have cohort, analysis and figure owners"
            )
        cohort_owners = [
            step for step in plan.steps if step.method == _COHORT_OWNER_METHOD
        ]
        suite_owners = [step for step in plan.steps if step.method == self.plan_method]
        figure_owners = [
            step
            for step in plan.steps
            if step.method == CONTINUOUS_SURVIVAL_FIGURE_METHOD
        ]
        if len(cohort_owners) != 1 or len(suite_owners) != 1 or len(figure_owners) != 1:
            raise ContinuousSurvivalAuthorityError(
                "continuous survival plan lacks a unique cohort, analysis or figure owner"
            )
        cohort_owner = cohort_owners[0]
        if cohort_owner.inputs or tuple(cohort_owner.expected_outputs) != (
            _COHORT_INPUT,
        ):
            raise ContinuousSurvivalAuthorityError(
                "continuous survival cohort owner drifted from host materialization"
            )
        step = suite_owners[0]
        issues: list[str] = []
        if step.planned_analysis_role != "primary":
            issues.append("planned_analysis_role")
        if step.intent != self.plan_intent:
            issues.append("intent")
        if tuple(step.expected_outputs) != self.analysis_plan_outputs:
            issues.append("expected_outputs")
        if not set(self.required_columns).issubset(step.inputs):
            issues.append("required_inputs")
        if not sole_typed_cohort_input(step):
            issues.append("typed_cohort_input")
        if (
            step.model_requirements
            or step.family_primary_result_requirement is not None
        ):
            issues.append("nested_model_contract")
        if step.table_one_spec is not None:
            issues.append("nested_table_one_contract")
        if issues:
            raise ContinuousSurvivalAuthorityError(
                "continuous survival plan drifted from signed authority: "
                + ", ".join(issues)
            )
        self._require_rule_ref(step)
        figure = figure_owners[0]
        expected_contracts = {
            (value, "all_rows") for value in self.figure_input_products
        }
        observed_contracts = {
            (item.input_key, item.mode) for item in figure.input_consumption_contracts
        }
        if (
            tuple(figure.inputs) != self.figure_input_products
            or tuple(figure.expected_outputs) != (self.figure_product,)
            or observed_contracts != expected_contracts
        ):
            raise ContinuousSurvivalAuthorityError(
                "continuous survival figure owner drifted from its signed sources"
            )
        self._require_rule_ref(figure)
        return step

    def governed_figure_step(self, plan: AnalysisPlan) -> AnalysisStep:
        self.governed_step(plan)
        return next(
            step
            for step in plan.steps
            if step.method == CONTINUOUS_SURVIVAL_FIGURE_METHOD
        )

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


__all__ = [
    "CONTINUOUS_EXPOSURE_WINDOW_SUMMARIES",
    "CONTINUOUS_SURVIVAL_AUTHORITY_KIND",
    "CONTINUOUS_SURVIVAL_FIGURE_METHOD",
    "CONTINUOUS_SURVIVAL_PLAN_METHOD",
    "CONTINUOUS_SURVIVAL_SUITE_MARKER",
    "ContinuousSurvivalAuthorityError",
    "LandmarkContinuousSurvivalRuntimeAuthority",
]
