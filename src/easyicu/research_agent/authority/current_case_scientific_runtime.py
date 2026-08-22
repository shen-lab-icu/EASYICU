"""Digest-bound current-run scientific authority for caller-reviewed cases.

The shared engine does not know E1, E2, H1 or H2.  It knows execution shapes:
one composed grid of already-supported adjusted-association fits, one frozen
landmark spline association, one fixed-landmark survival suite, and one
source-feasibility result that must fail closed without estimating an effect.
A benchmark-local compiler chooses the columns and scientific constants, seals
them, and passes the immutable value object here. Plan validation and
deterministic executors then consume the same object, so signed rules cannot
stop at prompt prose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Dict, Literal, Mapping, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from ..contracts.association_execution import (
    association_execution_verdict,
    sole_primary_model_requirement,
)
from ..contracts.cohort_product_keys import sole_typed_cohort_input
from ..schema import AnalysisPlan, AnalysisStep, ArtifactConsumptionContract


class CurrentCaseScientificAuthorityError(ValueError):
    """The plan drifted from a caller-reviewed current-run contract."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _normalise(value: Any) -> str:
    return "_".join(
        part
        for part in "".join(
            char.lower() if char.isalnum() else " " for char in str(value or "")
        ).split()
        if part
    )


class _AuthorityBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def plan_rule_ref(self) -> str:
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    def _verify_digest(self) -> None:
        body = self.model_dump(mode="json", exclude={"execution_contract_sha256"})
        observed = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        if observed != self.execution_contract_sha256:
            raise ValueError("current-run scientific execution-contract digest mismatch")

    def _require_rule_ref(self, step: AnalysisStep) -> None:
        if self.plan_rule_ref not in set(step.icu_rule_refs):
            raise CurrentCaseScientificAuthorityError(
                "governed plan step lacks the signed scientific runtime digest"
            )


class AssociationModelGridLandmarkFilter(BaseModel):
    """Retain rows alive at one declared landmark."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    filter_kind: Literal["alive_at_landmark"]
    outcome_column: str = Field(min_length=1)
    event_time_column: str = Field(min_length=1)
    landmark_hours: float = Field(gt=0)
    exclude_negative_event_times: Literal[True]


class AssociationModelGridLevelFilter(BaseModel):
    """Retain declared levels of one categorical/binary source column."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    filter_kind: Literal["level_in"]
    column: str = Field(min_length=1)
    declared_levels: Tuple[str, ...] = Field(min_length=2)
    retained_levels: Tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _closed_levels(self) -> "AssociationModelGridLevelFilter":
        if len(self.declared_levels) != len(set(self.declared_levels)):
            raise ValueError("model-grid declared filter levels must be unique")
        if len(self.retained_levels) != len(set(self.retained_levels)):
            raise ValueError("model-grid retained filter levels must be unique")
        if not set(self.retained_levels).issubset(self.declared_levels):
            raise ValueError("model-grid retained levels must be declared levels")
        return self


AssociationModelGridFilter = Annotated[
    Union[
        AssociationModelGridLandmarkFilter,
        AssociationModelGridLevelFilter,
    ],
    Field(discriminator="filter_kind"),
]


class AssociationModelGridNonlinearTerm(BaseModel):
    """One stable host-generated basis replacing a declared linear covariate."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_column: str = Field(min_length=1)
    basis: Literal["natural_cubic_spline"]
    degrees_of_freedom: int = Field(ge=3, le=8)
    center_before_basis: Literal[True]


ModelGridMetadataValue = Union[str, float, int, bool, None]


class AssociationModelGridVariant(BaseModel):
    """One prespecified eligibility/model-form variant in a composed grid."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    analysis_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    filters: Tuple[AssociationModelGridFilter, ...] = ()
    nonlinear_terms: Tuple[AssociationModelGridNonlinearTerm, ...] = ()
    metadata: Dict[str, ModelGridMetadataValue]

    @model_validator(mode="after")
    def _unique_owned_coordinates(self) -> "AssociationModelGridVariant":
        level_columns = [
            item.column
            for item in self.filters
            if isinstance(item, AssociationModelGridLevelFilter)
        ]
        if len(level_columns) != len(set(level_columns)):
            raise ValueError("model-grid level filters must target unique columns")
        landmarks = [
            item
            for item in self.filters
            if isinstance(item, AssociationModelGridLandmarkFilter)
        ]
        if len(landmarks) > 1:
            raise ValueError("one model-grid variant may declare at most one landmark")
        nonlinear_sources = [item.source_column for item in self.nonlinear_terms]
        if len(nonlinear_sources) != len(set(nonlinear_sources)):
            raise ValueError("model-grid nonlinear sources must be unique")
        return self


class AssociationModelGridRuntimeAuthority(_AuthorityBase):
    """Compose a closed variant grid from the existing verified model adapter.

    This authority owns no regression algorithm.  It binds one already-verified
    adjusted-association parent, declares row filters and stable basis
    transformations, and requires every variant to be executed through the same
    host adapter as the primary model.
    """

    schema_version: Literal["easyicu.association_model_grid_runtime_authority/1"]
    authority_kind: Literal["association_model_grid"]
    plan_method: Literal["verified_association_model_grid"]
    plan_intent: str = Field(min_length=1)
    cohort_product: Literal["artifact:analysis_cohort"]
    parent_product: Literal["table:adjusted_association_estimates"]
    output_product: str = Field(pattern=r"^table:[a-z][a-z0-9_]{0,79}$")
    reference_variant_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    metadata_columns: Tuple[str, ...]
    output_aliases: Dict[str, str] = Field(default_factory=dict)
    variants: Tuple[AssociationModelGridVariant, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _closed_contract(self) -> "AssociationModelGridRuntimeAuthority":
        ids = [item.analysis_id for item in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("association model-grid analysis ids must be unique")
        if self.reference_variant_id not in ids:
            raise ValueError("model-grid reference variant must be declared")
        if len(self.metadata_columns) != len(set(self.metadata_columns)):
            raise ValueError("model-grid metadata columns must be unique")
        if any(
            not value or not value.replace("_", "a").isalnum()
            for value in self.metadata_columns
        ):
            raise ValueError("model-grid metadata columns must be stable identifiers")
        for variant in self.variants:
            if set(variant.metadata) != set(self.metadata_columns):
                raise ValueError(
                    "every model-grid variant must populate the exact metadata columns"
                )
        allowed_aliases = {"n_events", "estimate"}
        if not set(self.output_aliases).issubset(allowed_aliases):
            raise ValueError("model-grid output aliases may rename only core results")
        alias_values = list(self.output_aliases.values())
        if len(alias_values) != len(set(alias_values)) or any(
            not value or not value.replace("_", "a").isalnum()
            for value in alias_values
        ):
            raise ValueError("model-grid output aliases must be unique identifiers")
        reserved = {
            "analysis_id",
            "n_stays",
            "n_events",
            "estimate",
            "ci_low",
            "ci_high",
            "effect_measure",
            "fit_n",
            "fit_events",
            "standard_error",
            "converged",
            "separation_detected",
        }
        if set(self.metadata_columns) & (reserved | set(alias_values)):
            raise ValueError("model-grid metadata columns collide with result columns")
        self._verify_digest()
        return self

    @property
    def sensitivity_ids(self) -> Tuple[str, ...]:
        return tuple(item.analysis_id for item in self.variants)

    def _parent(self, plan: AnalysisPlan) -> AnalysisStep:
        parents = [
            step
            for step in plan.steps
            if self.parent_product in set(step.expected_outputs)
        ]
        if len(parents) != 1:
            raise CurrentCaseScientificAuthorityError(
                "association model-grid requires one adjusted-association parent"
            )
        parent = parents[0]
        verdict = association_execution_verdict(parent)
        requirement = sole_primary_model_requirement(parent)
        if (
            parent.planned_analysis_role != "primary"
            or not verdict.claimed
            or requirement is None
            or requirement.outcome_type != "binary"
        ):
            raise CurrentCaseScientificAuthorityError(
                "association model-grid parent is not one host-owned binary model: "
                + verdict.reason
            )
        return parent

    def _candidate(self, plan: AnalysisPlan) -> AnalysisStep:
        by_output = [
            step for step in plan.steps if self.output_product in step.expected_outputs
        ]
        candidates = by_output or [
            step
            for step in plan.steps
            if step.planned_analysis_role == "sensitivity"
            and tuple(step.sensitivity_spec_ids) == self.sensitivity_ids
        ]
        if len(candidates) != 1:
            raise CurrentCaseScientificAuthorityError(
                "association model-grid requires one exact sensitivity step"
            )
        return candidates[0]

    def required_columns_from_requirement(self, requirement: Any) -> Tuple[str, ...]:
        """Return exact cohort columns for one already-validated parent model."""

        values = [
            requirement.exposure_source,
            requirement.outcome,
            *(item.name for item in requirement.model_terms or ()),
        ]
        if requirement.dependence is not None:
            values.append(requirement.dependence.group_source)
        for variant in self.variants:
            for item in variant.filters:
                if isinstance(item, AssociationModelGridLandmarkFilter):
                    values.extend((item.outcome_column, item.event_time_column))
                else:
                    values.append(item.column)
            values.extend(item.source_column for item in variant.nonlinear_terms)
        return tuple(dict.fromkeys(str(value) for value in values if value))

    def required_columns(self, plan: AnalysisPlan) -> Tuple[str, ...]:
        parent = self._parent(plan)
        requirement = sole_primary_model_requirement(parent)
        assert requirement is not None
        return self.required_columns_from_requirement(requirement)

    def bind_plan(self, plan: AnalysisPlan) -> AnalysisPlan:
        """Compile host-owned products and exact inputs into the draft plan."""

        parent = self._parent(plan)
        candidate = self._candidate(plan)
        parent_index = next(
            index for index, step in enumerate(plan.steps) if step is parent
        )
        candidate_index = next(
            index for index, step in enumerate(plan.steps) if step is candidate
        )
        if parent_index >= candidate_index:
            raise CurrentCaseScientificAuthorityError(
                "association model-grid parent must precede its sensitivity child"
            )
        inputs = [
            *self.required_columns(plan),
            self.cohort_product,
            self.parent_product,
        ]
        bound = candidate.model_copy(
            update={
                "planned_analysis_role": "sensitivity",
                "intent": self.plan_intent,
                "inputs": list(dict.fromkeys(inputs)),
                "expected_outputs": [self.output_product],
                "method": self.plan_method,
                "scientific_capability": "association_adjusted_v1",
                "icu_rule_refs": list(
                    dict.fromkeys([*candidate.icu_rule_refs, self.plan_rule_ref])
                ),
                "sensitivity_spec_ids": list(self.sensitivity_ids),
                "model_requirements": [],
                "family_primary_result_requirement": None,
                "input_consumption_contracts": [
                    ArtifactConsumptionContract(
                        input_key=self.parent_product,
                        mode="all_rows",
                    )
                ],
            }
        )
        steps = [bound if step is candidate else step for step in plan.steps]
        return plan.model_copy(update={"steps": steps})

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        parent = self._parent(plan)
        step = self._candidate(plan)
        parent_index = next(index for index, item in enumerate(plan.steps) if item is parent)
        step_index = next(index for index, item in enumerate(plan.steps) if item is step)
        issues: list[str] = []
        if parent_index >= step_index:
            issues.append("parent_order")
        if step.planned_analysis_role != "sensitivity":
            issues.append("planned_analysis_role")
        if step.method != self.plan_method:
            issues.append("method")
        if step.intent != self.plan_intent:
            issues.append("intent")
        if step.scientific_capability != "association_adjusted_v1":
            issues.append("scientific_capability")
        if tuple(step.expected_outputs) != (self.output_product,):
            issues.append("expected_outputs")
        if tuple(step.sensitivity_spec_ids) != self.sensitivity_ids:
            issues.append("sensitivity_spec_ids")
        required_inputs = set(
            (*self.required_columns(plan), self.cohort_product, self.parent_product)
        )
        if set(step.inputs) != required_inputs:
            issues.append("inputs")
        contracts = {
            item.input_key: item for item in step.input_consumption_contracts
        }
        if (
            set(contracts) != {self.parent_product}
            or contracts[self.parent_product].mode != "all_rows"
        ):
            issues.append("input_consumption_contracts")
        if step.model_requirements or step.family_primary_result_requirement is not None:
            issues.append("nested_model_contract")
        if issues:
            raise CurrentCaseScientificAuthorityError(
                "association model-grid plan drifted from signed authority: "
                + ", ".join(issues)
            )
        self._require_rule_ref(step)
        return step

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


class LandmarkSplineRuntimeAuthority(_AuthorityBase):
    schema_version: Literal["easyicu.landmark_spline_runtime_authority/1"]
    authority_kind: Literal["landmark_spline_association"]
    plan_method: Literal["signed_landmark_restricted_cubic_spline"]
    plan_intent: str
    plan_outputs: tuple[str, ...]
    exposure_column: str
    outcome_column: str
    outcome_time_column: str
    observation_duration_column: str
    observation_duration_unit: Literal["days"]
    landmark_hours: Literal[24]
    required_adjustment_columns: tuple[str, ...]
    categorical_adjustment_columns: tuple[str, ...]
    spline_knot_quantiles: tuple[float, float, float]
    spline_reference: Literal["median_in_primary_population"]
    curve_quantile_range: tuple[float, float]
    curve_points: int = Field(ge=5, le=201)
    linear_sensitivity_per_unit: Literal[1.0]
    interpretation: Literal["descriptive_prognostic_association_not_causal"]

    @model_validator(mode="after")
    def _closed_contract(self) -> "LandmarkSplineRuntimeAuthority":
        if self.spline_knot_quantiles != (0.10, 0.50, 0.90):
            raise ValueError("landmark spline authority requires frozen 10/50/90 knots")
        if self.curve_quantile_range != (0.10, 0.90):
            raise ValueError("landmark spline curve must span the frozen boundary knots")
        if len(self.required_adjustment_columns) != len(
            set(self.required_adjustment_columns)
        ):
            raise ValueError("landmark adjustment columns must be unique")
        if not set(self.categorical_adjustment_columns).issubset(
            self.required_adjustment_columns
        ):
            raise ValueError(
                "categorical adjustments must be members of the adjustment set"
            )
        contrast_products = [
            value
            for value in self.plan_outputs
            if value.startswith("table:")
            and "contrast" in value.partition(":")[2]
        ]
        if len(contrast_products) != 1:
            raise ValueError(
                "landmark spline authority requires one typed contrasts table"
            )
        if len(self._table_products_containing("curve")) != 1:
            raise ValueError("landmark spline authority requires one typed curve table")
        if len(self._table_products_containing("linear", "sensitivity")) != 1:
            raise ValueError(
                "landmark spline authority requires one typed linear-sensitivity table"
            )
        self._verify_digest()
        return self

    def _table_products_containing(self, *tokens: str) -> tuple[str, ...]:
        return tuple(
            value
            for value in self.plan_outputs
            if value.startswith("table:")
            and all(token in value.partition(":")[2] for token in tokens)
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.exposure_column,
                    self.outcome_column,
                    self.outcome_time_column,
                    self.observation_duration_column,
                    *self.required_adjustment_columns,
                )
            )
        )

    @property
    def downstream_parent_product(self) -> str:
        return next(
            value
            for value in self.plan_outputs
            if value.startswith("table:")
            and "contrast" in value.partition(":")[2]
        )

    @property
    def curve_product(self) -> str:
        return self._table_products_containing("curve")[0]

    @property
    def linear_sensitivity_product(self) -> str:
        return self._table_products_containing("linear", "sensitivity")[0]

    def bind_plan(self, plan: AnalysisPlan) -> AnalysisPlan:
        """Compile the signed deterministic primary into one draft plan.

        The Planner owns the study-specific surrounding design. The caller's
        digest-bound authority owns the exact landmark estimator, columns, and
        products, so copying those coordinates is mechanical wiring rather than
        a new scientific decision.
        """

        primary = [
            step for step in plan.steps if step.planned_analysis_role == "primary"
        ]
        if len(primary) != 1:
            raise CurrentCaseScientificAuthorityError(
                "landmark spline authority requires exactly one primary step"
            )
        candidate = primary[0]
        cohort_input = sole_typed_cohort_input(candidate)
        if not cohort_input:
            raise CurrentCaseScientificAuthorityError(
                "landmark spline primary requires one typed cohort input"
            )
        bound = candidate.model_copy(
            update={
                "method": self.plan_method,
                "intent": self.plan_intent,
                "scientific_capability": "association_freeform_v1",
                "expected_outputs": list(self.plan_outputs),
                "inputs": [cohort_input, *self.required_columns],
                "model_requirements": [],
                "family_primary_result_requirement": None,
                "icu_rule_refs": list(
                    dict.fromkeys([*candidate.icu_rule_refs, self.plan_rule_ref])
                ),
            }
        )
        generic_parent = "table:adjusted_association_estimates"
        replacement = self.downstream_parent_product
        steps: list[AnalysisStep] = []
        for step in plan.steps:
            if step is candidate:
                steps.append(bound)
                continue
            inputs = [
                replacement if value == generic_parent else value
                for value in step.inputs
            ]
            if (
                step.planned_analysis_role == "sensitivity"
                and replacement in inputs
                and self.linear_sensitivity_product not in inputs
            ):
                inputs.append(self.linear_sensitivity_product)
            contracts = [
                item.model_copy(update={"input_key": replacement})
                if item.input_key == generic_parent
                else item
                for item in step.input_consumption_contracts
            ]
            if step.planned_analysis_role == "sensitivity" and replacement in inputs:
                contracted = {item.input_key for item in contracts}
                contracts.extend(
                    ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
                    for input_key in (
                        replacement,
                        self.linear_sensitivity_product,
                    )
                    if input_key not in contracted
                )
            steps.append(
                step.model_copy(
                    update={
                        "inputs": list(dict.fromkeys(inputs)),
                        "input_consumption_contracts": contracts,
                    }
                )
            )
        return plan.model_copy(update={"steps": steps})

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        primary = [
            step for step in plan.steps if step.planned_analysis_role == "primary"
        ]
        if len(primary) != 1:
            raise CurrentCaseScientificAuthorityError(
                "landmark spline authority requires exactly one primary step"
            )
        step = primary[0]
        issues: list[str] = []
        if step.method != self.plan_method:
            issues.append("method")
        if step.intent != self.plan_intent:
            issues.append("intent")
        if step.scientific_capability != "association_freeform_v1":
            issues.append("scientific_capability")
        if tuple(step.expected_outputs) != self.plan_outputs:
            issues.append("expected_outputs")
        if not set(self.required_columns).issubset(step.inputs):
            issues.append("required_inputs")
        if not sole_typed_cohort_input(step):
            issues.append("typed_cohort_input")
        if step.model_requirements:
            issues.append("model_requirements")
        if issues:
            raise CurrentCaseScientificAuthorityError(
                "landmark spline plan drifted from signed authority: "
                + ", ".join(issues)
            )
        self._require_rule_ref(step)
        return step

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


class LandmarkSurvivalRuntimeAuthority(_AuthorityBase):
    """Closed fixed-landmark survival suite over one typed cohort.

    The authority chooses no case science. A caller-reviewed protocol supplies
    the time origin, exposure timing rule, endpoint columns, adjustment set and
    output products. The host then performs one deterministic risk-set build,
    Table 1, Kaplan-Meier summary, adjusted Cox fit, PH audit and composite
    figure without routing any of those mechanical operations through a Coder.
    """

    schema_version: Literal["easyicu.landmark_survival_runtime_authority/1"]
    authority_kind: Literal["landmark_survival_suite"]
    plan_method: Literal["signed_landmark_survival_suite"]
    plan_intent: str = Field(min_length=1)
    plan_outputs: tuple[str, ...]
    exposure_status_column: str = Field(min_length=1)
    exposure_onset_column: str = Field(min_length=1)
    event_column: str = Field(min_length=1)
    followup_time_column: str = Field(min_length=1)
    landmark_hours: float = Field(gt=0)
    endpoint_horizon_days: float = Field(gt=0)
    exposure_window_hours: tuple[float, float]
    prevalent_exposure_cutoff_hours: float
    prevalent_exposure_action: Literal["exclude"]
    exposed_group_label: str = Field(min_length=1)
    comparator_group_label: str = Field(min_length=1)
    analysis_unit_label: str = Field(min_length=1)
    derived_exposure_column: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    derived_event_column: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    derived_time_column: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    adjustment_columns: tuple[str, ...]
    categorical_adjustment_columns: tuple[str, ...]
    table_one_columns: tuple[str, ...]
    estimator: Literal["cox_ph_lifelines_efron"]
    effect_measure: Literal["hazard_ratio"]
    uncertainty_method: Literal["wald_95_ci"]
    proportional_hazards_diagnostic: Literal["schoenfeld_residual_test"]
    proportional_hazards_alpha: float = Field(gt=0, lt=1)
    proportional_hazards_policy: Literal[
        "report_only", "block_paper_authorization"
    ]
    interpretation: Literal["descriptive_prognostic_association_not_causal"]
    table_one_product: str = Field(pattern=r"^table:[a-z][a-z0-9_]{0,79}$")
    risk_set_product: str = Field(pattern=r"^table:[a-z][a-z0-9_]{0,79}$")
    km_product: str = Field(pattern=r"^table:[a-z][a-z0-9_]{0,79}$")
    cox_product: str = Field(pattern=r"^table:[a-z][a-z0-9_]{0,79}$")
    ph_product: str = Field(pattern=r"^table:[a-z][a-z0-9_]{0,79}$")
    receipt_product: str = Field(pattern=r"^log:[a-z][a-z0-9_]{0,79}$")
    figure_product: str = Field(pattern=r"^figure:[a-z][a-z0-9_]{0,79}$")

    @model_validator(mode="after")
    def _closed_contract(self) -> "LandmarkSurvivalRuntimeAuthority":
        window_start, window_end = self.exposure_window_hours
        if not (
            window_start
            <= self.prevalent_exposure_cutoff_hours
            < window_end
        ):
            raise ValueError(
                "landmark survival prevalent cutoff must fall inside the exposure window"
            )
        if window_end != self.landmark_hours or window_start >= window_end:
            raise ValueError(
                "landmark survival exposure window must end at the landmark"
            )
        if self.landmark_hours / 24.0 >= self.endpoint_horizon_days:
            raise ValueError(
                "landmark survival landmark must precede the endpoint horizon"
            )
        source_columns = (
            self.exposure_status_column,
            self.exposure_onset_column,
            self.event_column,
            self.followup_time_column,
            *self.adjustment_columns,
        )
        if len(source_columns) != len(set(source_columns)):
            raise ValueError("landmark survival source columns must be unique")
        if len(self.adjustment_columns) != len(set(self.adjustment_columns)):
            raise ValueError("landmark survival adjustment columns must be unique")
        if not set(self.categorical_adjustment_columns).issubset(
            self.adjustment_columns
        ):
            raise ValueError(
                "landmark survival categorical adjustments must be adjusted columns"
            )
        if not set(self.table_one_columns).issubset(self.adjustment_columns):
            raise ValueError(
                "landmark survival Table 1 columns must come from the adjustment set"
            )
        derived = {
            self.derived_exposure_column,
            self.derived_event_column,
            self.derived_time_column,
        }
        if len(derived) != 3 or derived & set(source_columns):
            raise ValueError(
                "landmark survival derived columns must be distinct from source columns"
            )
        products = (
            self.table_one_product,
            self.risk_set_product,
            self.km_product,
            self.cox_product,
            self.ph_product,
            self.receipt_product,
            self.figure_product,
        )
        if len(products) != len(set(products)):
            raise ValueError("landmark survival output products must be unique")
        if self.plan_outputs != products:
            raise ValueError(
                "landmark survival plan outputs must equal the seven owned products"
            )
        self._verify_digest()
        return self

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.exposure_status_column,
                    self.exposure_onset_column,
                    self.event_column,
                    self.followup_time_column,
                    *self.adjustment_columns,
                )
            )
        )

    def bind_plan(self, plan: AnalysisPlan) -> AnalysisPlan:
        """Collapse the signed survival suite into its one deterministic owner."""

        primary = [
            step for step in plan.steps if step.planned_analysis_role == "primary"
        ]
        if len(primary) != 1:
            raise CurrentCaseScientificAuthorityError(
                "landmark survival authority requires exactly one primary step"
            )
        candidate = primary[0]
        cohort_input = sole_typed_cohort_input(candidate)
        if not cohort_input:
            cohort_inputs = {
                value
                for step in plan.steps
                for value in step.inputs
                if value in {"artifact:analysis_cohort", "table:analysis_cohort", "dataset:analysis_cohort"}
                or value.startswith("cohort:")
            }
            if len(cohort_inputs) != 1:
                raise CurrentCaseScientificAuthorityError(
                    "landmark survival suite requires one typed cohort input"
                )
            cohort_input = next(iter(cohort_inputs))
        bound = candidate.model_copy(
            update={
                "planned_analysis_role": "primary",
                "method": self.plan_method,
                "intent": self.plan_intent,
                "inputs": [cohort_input, *self.required_columns],
                "expected_outputs": list(self.plan_outputs),
                "scientific_capability": None,
                "model_requirements": [],
                "family_primary_result_requirement": None,
                "table_one_spec": None,
                "input_consumption_contracts": [],
                "icu_rule_refs": list(
                    dict.fromkeys([*candidate.icu_rule_refs, self.plan_rule_ref])
                ),
            }
        )
        return plan.model_copy(update={"steps": [bound]})

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        if len(plan.steps) != 1:
            raise CurrentCaseScientificAuthorityError(
                "landmark survival suite must have exactly one owner step"
            )
        step = plan.steps[0]
        issues: list[str] = []
        if step.planned_analysis_role != "primary":
            issues.append("planned_analysis_role")
        if step.method != self.plan_method:
            issues.append("method")
        if step.intent != self.plan_intent:
            issues.append("intent")
        if tuple(step.expected_outputs) != self.plan_outputs:
            issues.append("expected_outputs")
        if not set(self.required_columns).issubset(step.inputs):
            issues.append("required_inputs")
        if not sole_typed_cohort_input(step):
            issues.append("typed_cohort_input")
        if step.model_requirements or step.family_primary_result_requirement is not None:
            issues.append("nested_model_contract")
        if step.table_one_spec is not None:
            issues.append("nested_table_one_contract")
        if issues:
            raise CurrentCaseScientificAuthorityError(
                "landmark survival plan drifted from signed authority: "
                + ", ".join(issues)
            )
        self._require_rule_ref(step)
        return step

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


class SourceFeasibilityRuntimeAuthority(_AuthorityBase):
    schema_version: Literal["easyicu.source_feasibility_runtime_authority/1"]
    authority_kind: Literal["source_feasibility_fail_closed"]
    plan_method: Literal["signed_source_feasibility_fail_closed"]
    plan_intent: str
    plan_outputs: tuple[str, ...]
    source: str
    audited_window_hours: tuple[Literal[0], Literal[24]]
    decision: Literal["fail_closed"]
    reason_code: Literal["H2_VERIFIED_NON_USE_UNAVAILABLE"]
    verified_non_use_available: Literal[False]
    binary_control_arm_authorized: Literal[False]
    causal_contrast_authorized: Literal[False]
    forbidden_plan_tokens: tuple[str, ...]
    future_design_authorized: Literal[False]

    @model_validator(mode="after")
    def _closed_contract(self) -> "SourceFeasibilityRuntimeAuthority":
        if not self.forbidden_plan_tokens:
            raise ValueError("source feasibility authority needs forbidden plan tokens")
        if len(self.forbidden_plan_tokens) != len(set(self.forbidden_plan_tokens)):
            raise ValueError("source feasibility forbidden tokens must be unique")
        self._verify_digest()
        return self

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        if len(plan.steps) != 1:
            raise CurrentCaseScientificAuthorityError(
                "source feasibility-only plan contains forbidden additional steps"
            )
        owners = [step for step in plan.steps if step.method == self.plan_method]
        if len(owners) != 1:
            raise CurrentCaseScientificAuthorityError(
                "source feasibility authority requires exactly one fail-closed owner"
            )
        if any(step.planned_analysis_role == "primary" for step in plan.steps):
            raise CurrentCaseScientificAuthorityError(
                "source feasibility-only plan must not declare a primary effect step"
            )
        step = owners[0]
        issues: list[str] = []
        if step.intent != self.plan_intent:
            issues.append("intent")
        if tuple(step.expected_outputs) != self.plan_outputs:
            issues.append("expected_outputs")
        if step.model_requirements or step.family_primary_result_requirement is not None:
            issues.append("effect_model_contract")
        if issues:
            raise CurrentCaseScientificAuthorityError(
                "source feasibility plan drifted from signed authority: "
                + ", ".join(issues)
            )
        self._require_rule_ref(step)
        forbidden = set(self.forbidden_plan_tokens)
        for candidate in plan.steps:
            if candidate is step:
                # The signed owner uses negated phrases such as "no effect estimate".
                # Its fields were already checked for exact equality above, so scan
                # only additional steps for prohibited work.
                continue
            fields = (
                candidate.method,
                candidate.intent,
                *candidate.expected_outputs,
            )
            observed = {_normalise(value) for value in fields if value}
            for value in observed:
                if any(
                    token == value
                    or value.startswith(token + "_")
                    or value.endswith("_" + token)
                    or ("_" + token + "_") in ("_" + value + "_")
                    for token in forbidden
                ):
                    raise CurrentCaseScientificAuthorityError(
                        "source feasibility-only plan requested a forbidden current-run "
                        f"action: {value}"
                    )
        return step

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


CurrentCaseScientificRuntimeAuthority = Annotated[
    Union[
        AssociationModelGridRuntimeAuthority,
        LandmarkSplineRuntimeAuthority,
        LandmarkSurvivalRuntimeAuthority,
        SourceFeasibilityRuntimeAuthority,
    ],
    Field(discriminator="authority_kind"),
]
_AUTHORITY_ADAPTER = TypeAdapter(CurrentCaseScientificRuntimeAuthority)


def load_current_case_scientific_runtime_authority(
    value: CurrentCaseScientificRuntimeAuthority | Mapping[str, Any],
) -> CurrentCaseScientificRuntimeAuthority:
    if isinstance(
        value,
        (
            AssociationModelGridRuntimeAuthority,
            LandmarkSplineRuntimeAuthority,
            LandmarkSurvivalRuntimeAuthority,
            SourceFeasibilityRuntimeAuthority,
        ),
    ):
        return value
    return _AUTHORITY_ADAPTER.validate_json(
        json.dumps(_plain(value), sort_keys=True), strict=True
    )


def build_current_case_scientific_runtime_authority(
    value: Mapping[str, Any],
) -> CurrentCaseScientificRuntimeAuthority:
    body = dict(value)
    if "execution_contract_sha256" in body:
        raise ValueError("execution contract digest is host-generated")
    body["execution_contract_sha256"] = hashlib.sha256(
        _canonical_bytes(body)
    ).hexdigest()
    return load_current_case_scientific_runtime_authority(body)


__all__ = [
    "AssociationModelGridFilter",
    "AssociationModelGridLandmarkFilter",
    "AssociationModelGridLevelFilter",
    "AssociationModelGridNonlinearTerm",
    "AssociationModelGridRuntimeAuthority",
    "AssociationModelGridVariant",
    "CurrentCaseScientificAuthorityError",
    "CurrentCaseScientificRuntimeAuthority",
    "LandmarkSplineRuntimeAuthority",
    "LandmarkSurvivalRuntimeAuthority",
    "SourceFeasibilityRuntimeAuthority",
    "build_current_case_scientific_runtime_authority",
    "load_current_case_scientific_runtime_authority",
]
