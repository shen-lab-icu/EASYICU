"""Digest-bound target trial emulated by clone, censor and weight.

A causal question about when to start a treatment is planned as the emulation
of a target trial (``planning.target_trial_spec``): the ICU stays eligible at
time zero ``T0``, two strategies -- start the treatment within a grace period
of ``G`` hours, or do not start it then -- and the risk of death by a fixed
horizon under each.  The study setup states the trial, the host compiles it
(``planning.target_trial_compile``) and the researcher confirms what the
approval card lists.  This authority seals the result: the compile record the
plan was approved from and the researcher's own confirmation of it, the
column the input holds for each element, the reader words, the products, and
the digest of the host policy (``contracts.target_trial_design``) the
estimate runs under.  The host then runs one deterministic suite --
eligibility, the clone-censor-weight estimate with its diagnostics and stops,
the bootstrap -- and the composite figure.  None of it goes through a Coder.

The estimate is an analysis under the emulation's assumptions: its evidence
ceiling is ``analysis_only`` and its causal sentences are fixed templates.

The authority chooses no case science and imports no other authority: the
current-case union imports it.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...outcome_availability import fixed_horizon_mortality_endpoint
from ...utils.death_time_semantics import DEATH_STATUS, DEATH_TIME_COMPANION
from ..canonical_json import canonical_sha256
from ..contracts.cohort_product_keys import sole_typed_cohort_input
from ..contracts.dependence import PlannedDependenceRequirement
from ..contracts.figure_plan import DeterministicFigurePanelTemplate
from ..contracts.model_terms import ModelTermSpec
from ..contracts.runtime_outcomes import RuntimeOutcomeContract
from ..contracts.target_trial_design import (
    MAX_GRACE_PERIOD_HOURS,
    MAX_TIME_ZERO_HOURS,
    MIN_TIME_ZERO_HOURS,
    target_trial_host_policy_sha256,
)
from ..research_context.stay_events import ICU_LENGTH_OF_STAY_CONCEPT
from ..schema import AnalysisPlan, AnalysisStep, EndpointSpec

TARGET_TRIAL_AUTHORITY_KIND = "target_trial_suite"
TARGET_TRIAL_PLAN_METHOD = "signed_target_trial_suite"
TARGET_TRIAL_FIGURE_METHOD = "signed_target_trial_figure"
#: The marker line the planning disclosure prints before its JSON coordinates.
#: It contains no other suite's marker, so no suite reads another's.
TARGET_TRIAL_SUITE_MARKER = "CALLER-BOUND TARGET TRIAL EMULATION SUITE:"
#: The most treatment concepts and confounders a trial states
#: (``planning.target_trial_spec``).
MAX_TRIAL_TREATMENT_COLUMNS = 4
MAX_TRIAL_COVARIATES = 24

_SHA256 = r"^[0-9a-f]{64}$"
_TABLE = r"^table:[a-z][a-z0-9_]{0,79}$"
#: Reader words: no Markdown, placeholder or table syntax.
_READER = r"^[^{}\[\]<>`\\|*_#\n]{1,160}$"
#: Words a result sentence states, which carry no digit: every number such a
#: sentence prints binds to a result value, and a label's number would not.
_SENTENCE_WORDS = r"^[^{}\[\]<>`\\|*_#\n0-9]{1,80}$"
_COHORT_INPUT = "table:analysis_cohort"
_COHORT_OWNER_METHOD = "host_materialized_locked_cohort"


class TargetTrialAuthorityError(ValueError):
    """A plan drifted from the signed target trial suite."""


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TargetTrialCovariate(_Closed):
    """One confounder the weights adjust for, as its column enters the models.

    ``unmeasured_state`` keeps the stays it was not measured for as their own
    state under the model-retention rule; otherwise those stays need it
    measured.  Every field is stated: the authority digest is computed over
    the body as the host wrote it.
    """

    column: str = Field(min_length=1, max_length=128)
    label: str = Field(pattern=_READER)
    coding: Literal["continuous", "binary", "categorical"]
    levels: tuple[str, ...]
    reference_level: Optional[str]
    unmeasured_state: bool

    @model_validator(mode="after")
    def _one_exact_coding(self) -> "TargetTrialCovariate":
        if self.coding == "continuous":
            if self.levels or self.reference_level is not None:
                raise ValueError("a continuous covariate declares no levels")
            return self
        if any(not level.strip() or level != level.strip() for level in self.levels):
            raise ValueError("covariate levels must be normalized and non-empty")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("covariate levels must be unique")
        if self.coding == "binary" and len(self.levels) != 2:
            raise ValueError("a binary covariate has exactly two levels")
        if len(self.levels) < 2 or self.reference_level not in self.levels:
            raise ValueError("a coded covariate names its reference among its levels")
        return self

    def model_term(self) -> ModelTermSpec:
        """The covariate as the shared model-term compiler reads it."""

        coded = self.coding != "continuous"
        return ModelTermSpec(
            name=self.column,
            role="covariate",
            coding=self.coding,
            levels=list(self.levels) if coded else None,
            reference_level=self.reference_level,
            transform="treatment_contrast" if coded else "identity",
        )


class TargetTrialConfirmation(_Closed):
    """The researcher's own confirmation of what the approval card listed.

    The card lists the compile record's confirmation lines: a capture reading
    or a class composition development assumed, a coordinate the study did not
    state, the adjustment set with the no-unmeasured-confounding assumption it
    carries, each element the spec could not type.  A click records an
    approval event; a revision the system generated records none, so it cannot
    stand in for one and the trial does not execute.
    """

    confirmed_by: Literal["researcher"]
    approval_event_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}$")
    #: The compile record the card showed; it lists every line confirmed.
    confirmed_compile_sha256: str = Field(pattern=_SHA256)
    #: The lines the card showed and the researcher confirmed.
    n_lines_confirmed: int = Field(ge=1)


class TargetTrialRuntimeAuthority(_Closed):
    """Closed clone-censor-weight emulation of one grace-period target trial.

    Every field is stated in the signed body: the digest is computed over the
    body as the host wrote it, so a field with a default would digest
    differently once validated.
    """

    schema_version: Literal["easyicu.target_trial_runtime_authority/1"]
    authority_kind: Literal["target_trial_suite"]
    protocol_content_sha256: str = Field(pattern=_SHA256)
    execution_contract_sha256: str = Field(pattern=_SHA256)
    #: ``CompiledTargetTrial.sha256()`` of the record the plan was approved from.
    target_trial_compile_sha256: str = Field(pattern=_SHA256)
    #: How many confirmation lines that record lists; the researcher confirms
    #: every one, so a card that showed fewer cannot stand in for it.
    target_trial_compile_confirmation_lines: int = Field(ge=1)
    #: ``target_trial_host_policy_sha256()`` when the trial was signed.
    host_policy_sha256: str = Field(pattern=_SHA256)
    confirmation: TargetTrialConfirmation
    plan_method: Literal["signed_target_trial_suite"]
    plan_intent: str = Field(min_length=1)
    plan_outputs: tuple[str, ...]
    development_execution_only_allowed: bool
    database: str = Field(min_length=1)
    #: Reader words: the analysis unit, the population at time zero, the
    #: treatment, the two strategies and the outcome.  All but the population,
    #: which only the protocol table shows, enter result sentences.
    analysis_unit_label: str = Field(pattern=_SENTENCE_WORDS)
    eligibility_label: str = Field(pattern=_READER)
    treatment_label: str = Field(pattern=_SENTENCE_WORDS)
    initiate_label: str = Field(pattern=_SENTENCE_WORDS)
    defer_label: str = Field(pattern=_SENTENCE_WORDS)
    outcome_label: str = Field(pattern=_SENTENCE_WORDS)
    unit_id_column: str = Field(min_length=1)
    #: A bootstrap resamples stays, or patients with all their stays when a
    #: patient may contribute several; the group is then derived exactly as
    #: the dependence owner declares it (``contracts.dependence``).
    resampling_unit: Literal["icu_stay", "patient"]
    patient_group_column: Optional[str]
    patient_group_derivation: Optional[Literal["identity", "prefix_before_delimiter"]]
    patient_group_delimiter: Optional[str]
    #: Event-status onset columns read together: the start is the earliest.
    treatment_onset_columns: tuple[str, ...]
    #: ``[start, end)`` hours after ICU admission the onsets were captured over.
    treatment_onset_window_hours: tuple[float, float]
    time_zero_hours: int
    grace_period_hours: int
    event_column: str = Field(min_length=1)
    followup_time_column: str = Field(min_length=1)
    endpoint_horizon_days: int = Field(gt=0)
    endpoint_time_origin: Literal["icu_admission"]
    death_status_column: str = Field(min_length=1)
    death_time_column: str = Field(min_length=1)
    icu_length_of_stay_column: str = Field(min_length=1)
    covariates: tuple[TargetTrialCovariate, ...]
    estimator: Literal["clone_censor_weight"]
    interpretation: Literal["per_protocol_effect_under_emulation_assumptions"]
    evidence_ceiling: Literal["analysis_only"]
    protocol_product: str = Field(pattern=_TABLE)
    eligibility_product: str = Field(pattern=_TABLE)
    table_one_product: str = Field(pattern=_TABLE)
    risk_curve_product: str = Field(pattern=_TABLE)
    effect_product: str = Field(pattern=_TABLE)
    weight_model_product: str = Field(pattern=_TABLE)
    weight_product: str = Field(pattern=_TABLE)
    balance_product: str = Field(pattern=_TABLE)
    positivity_product: str = Field(pattern=_TABLE)
    adherence_product: str = Field(pattern=_TABLE)
    bootstrap_product: str = Field(pattern=_TABLE)
    receipt_product: str = Field(pattern=r"^log:[a-z][a-z0-9_]{0,79}$")
    figure_product: str = Field(pattern=r"^figure:[a-z][a-z0-9_]{0,79}$")

    @model_validator(mode="after")
    def _closed_contract(self) -> "TargetTrialRuntimeAuthority":
        if not MIN_TIME_ZERO_HOURS <= self.time_zero_hours <= MAX_TIME_ZERO_HOURS:
            raise ValueError("target trial time zero is outside the host's menu")
        if not 1 <= self.grace_period_hours <= MAX_GRACE_PERIOD_HOURS:
            raise ValueError("target trial grace period is outside the host's menu")
        if self.endpoint_horizon_days * 24 <= self.grace_end_hours:
            raise ValueError("target trial horizon must end after the grace period")
        start, end = self.treatment_onset_window_hours
        if not (start <= 0.0 and end >= self.grace_end_hours):
            raise ValueError(
                "target trial treatment onsets must be captured from ICU admission "
                "through the end of the grace period"
            )
        if self.host_policy_sha256 != target_trial_host_policy_sha256():
            raise ValueError("target trial was signed under another host policy")
        if (
            self.confirmation.confirmed_compile_sha256
            != self.target_trial_compile_sha256
        ):
            raise ValueError("target trial confirmation is for another compile record")
        if (
            self.confirmation.n_lines_confirmed
            != self.target_trial_compile_confirmation_lines
        ):
            raise ValueError(
                "target trial confirmation covers another number of lines than its "
                "compile record lists"
            )
        endpoint = fixed_horizon_mortality_endpoint(self.event_column)
        if (
            endpoint is None
            or endpoint.followup_concept != self.followup_time_column
            or endpoint.horizon_days != self.endpoint_horizon_days
            or endpoint.time_origin != self.endpoint_time_origin
        ):
            raise ValueError(
                "target trial endpoint is not a closed fixed-horizon death"
            )
        if (
            self.death_status_column != DEATH_STATUS
            or self.death_time_column != DEATH_TIME_COMPANION
            or self.icu_length_of_stay_column != ICU_LENGTH_OF_STAY_CONCEPT
        ):
            raise ValueError("target trial host columns drifted from their owners")
        onsets = self.treatment_onset_columns
        if not 1 <= len(onsets) <= MAX_TRIAL_TREATMENT_COLUMNS:
            raise ValueError("target trial names one to four treatment onset columns")
        if not 1 <= len(self.covariates) <= MAX_TRIAL_COVARIATES:
            raise ValueError("target trial weights adjust for one to 24 confounders")
        if len({item.label.casefold() for item in self.covariates}) != len(
            self.covariates
        ):
            raise ValueError("target trial covariate labels must be distinct")
        source = (
            self.unit_id_column,
            *onsets,
            self.event_column,
            self.followup_time_column,
            self.death_status_column,
            self.death_time_column,
            self.icu_length_of_stay_column,
            *(item.column for item in self.covariates),
        )
        if len(source) != len(set(source)):
            raise ValueError("target trial source columns must be unique")
        self._closed_resampling(source)
        if (
            self.initiate_label.strip().casefold()
            == self.defer_label.strip().casefold()
        ):
            raise ValueError("target trial strategies need different labels")
        products = self.owned_products
        if len(products) != len(set(products)):
            raise ValueError("target trial output products must be unique")
        if self.plan_outputs != products:
            raise ValueError("target trial plan outputs must equal the owned products")
        body = self.model_dump(mode="json", exclude={"execution_contract_sha256"})
        if canonical_sha256(body) != self.execution_contract_sha256:
            raise ValueError("target trial authority digest mismatch")
        return self

    def _closed_resampling(self, source: tuple[str, ...]) -> None:
        group = (
            self.patient_group_column,
            self.patient_group_derivation,
            self.patient_group_delimiter,
        )
        if self.resampling_unit == "icu_stay":
            if any(value is not None for value in group):
                raise ValueError("a stay bootstrap declares no patient group")
            return
        if self.patient_group_column is None or self.patient_group_derivation is None:
            raise ValueError("a patient bootstrap declares its patient group")
        # The group may be derived from the stay identity itself, never from
        # a column of another role.
        if (
            self.patient_group_column in source
            and self.patient_group_column != self.unit_id_column
        ):
            raise ValueError("the patient group column has another role")
        self.patient_group_requirement()

    @property
    def grace_end_hours(self) -> int:
        return self.time_zero_hours + self.grace_period_hours

    @property
    def horizon_hours(self) -> int:
        return self.endpoint_horizon_days * 24

    def patient_group_requirement(self) -> Optional[PlannedDependenceRequirement]:
        """The patient grouping a patient bootstrap resamples, else ``None``."""

        if self.resampling_unit != "patient":
            return None
        return PlannedDependenceRequirement(
            group_source=str(self.patient_group_column),
            group_derivation=self.patient_group_derivation,
            delimiter=self.patient_group_delimiter,
        )

    @property
    def owned_products(self) -> tuple[str, ...]:
        return (
            self.protocol_product,
            self.eligibility_product,
            self.table_one_product,
            self.risk_curve_product,
            self.effect_product,
            self.weight_model_product,
            self.weight_product,
            self.balance_product,
            self.positivity_product,
            self.adherence_product,
            self.bootstrap_product,
            self.receipt_product,
            self.figure_product,
        )

    @property
    def plan_rule_ref(self) -> str:
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    @property
    def required_columns(self) -> tuple[str, ...]:
        group = (
            (self.patient_group_column,)
            if self.patient_group_column is not None
            and self.patient_group_column != self.unit_id_column
            else ()
        )
        return (
            self.unit_id_column,
            *group,
            *self.treatment_onset_columns,
            self.event_column,
            self.followup_time_column,
            self.death_status_column,
            self.death_time_column,
            self.icu_length_of_stay_column,
            *(item.column for item in self.covariates),
        )

    @property
    def analysis_plan_outputs(self) -> tuple[str, ...]:
        return tuple(
            value for value in self.plan_outputs if value != self.figure_product
        )

    @property
    def figure_input_products(self) -> tuple[str, ...]:
        return (
            self.protocol_product,
            self.eligibility_product,
            self.risk_curve_product,
            self.effect_product,
            self.balance_product,
            self.weight_product,
        )

    def _require_rule_ref(self, step: AnalysisStep) -> None:
        if self.plan_rule_ref not in step.icu_rule_refs:
            raise TargetTrialAuthorityError(
                "target trial step lacks its runtime contract reference"
            )

    def bind_plan(self, plan: AnalysisPlan) -> AnalysisPlan:
        """Compile the signed emulation and its source-bound renderer."""

        primary = [
            step for step in plan.steps if step.planned_analysis_role == "primary"
        ]
        if len(primary) != 1:
            raise TargetTrialAuthorityError(
                "target trial authority requires exactly one primary step"
            )
        candidate = primary[0]
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
                "step_id": "02_authority_compiled_target_trial_figure",
                "planned_analysis_role": "auxiliary",
                "intent": (
                    "Render the signed target trial emulation from its exact "
                    "result tables."
                ),
                "inputs": list(self.figure_input_products),
                "expected_outputs": [self.figure_product],
                "method": TARGET_TRIAL_FIGURE_METHOD,
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

        The protocol and its timeline lead, then the strategies' risks over
        follow-up, the balance they rest on, and the estimate under each
        weighting the host prespecifies.
        """

        return (
            DeterministicFigurePanelTemplate(
                panel_id="a",
                article_role="causal_protocol",
                chart_type="timeline_diagram",
                source_products=(self.protocol_product, self.eligibility_product),
            ),
            DeterministicFigurePanelTemplate(
                panel_id="b",
                article_role="causal_contrast",
                chart_type="effect_curve",
                source_products=(self.risk_curve_product, self.effect_product),
            ),
            DeterministicFigurePanelTemplate(
                panel_id="c",
                article_role="balance_positivity",
                chart_type="love_plot",
                source_products=(self.balance_product,),
            ),
            DeterministicFigurePanelTemplate(
                panel_id="d",
                article_role="robustness",
                chart_type="trimming_panel",
                source_products=(self.effect_product, self.weight_product),
            ),
        )

    def planning_contract_context(self) -> str:
        """Disclose the sealed suite so a planner can name its owner, not re-derive it."""

        coordinates = {
            "sealed_primary_owner": self.plan_method,
            "treatment_onset_columns": list(self.treatment_onset_columns),
            "time_zero_hours": self.time_zero_hours,
            "grace_period_hours": self.grace_period_hours,
            "event_column": self.event_column,
            "followup_time_column": self.followup_time_column,
            "endpoint_horizon_days": self.endpoint_horizon_days,
            "adjustment_columns": [item.column for item in self.covariates],
            "resampling_unit": self.resampling_unit,
            "plan_outputs": list(self.plan_outputs),
        }
        return (
            f"{TARGET_TRIAL_SUITE_MARKER} the single primary step is owned by "
            "the sealed host suite named in sealed_primary_owner; it must declare "
            "exactly the listed source columns and outputs and carry no model "
            "requirement of its own. The host compiles eligibility at time zero, "
            "cloning into both strategies, artificial censoring with its weights, "
            "the weighted risks, their diagnostics and stops, the bootstrap and "
            "the composite figure from this contract.\n"
            + json.dumps(coordinates, ensure_ascii=False, sort_keys=True)
        )

    def development_execution_only_plan(
        self,
        *,
        research_question: str,
    ) -> AnalysisPlan:
        """Mechanically project this complete authority into one dev-only plan."""

        if not self.development_execution_only_allowed:
            raise TargetTrialAuthorityError(
                "target trial authority does not allow execution-only development"
            )
        draft = AnalysisPlan.model_validate(
            {
                "research_question": str(research_question),
                "analysis_type": "causal_inference",
                "endpoint": self.research_context_endpoint().model_dump(mode="json"),
                "steps": [
                    {
                        "step_id": "01_authority_compiled_target_trial",
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

        endpoint = fixed_horizon_mortality_endpoint(self.event_column)
        assert endpoint is not None  # the closed contract checked it
        return EndpointSpec(
            name=self.event_column,
            kind="time_to_event",
            absence_semantics="absent_row_is_unmeasured",
            levels=[0, 1],
            event_column=self.event_column,
            time_column=self.followup_time_column,
            time_origin=self.endpoint_time_origin,
            censoring_rule=endpoint.censoring_rule,
        )

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        """Return the deterministic analysis owner after validating the suite."""

        if len(plan.steps) != 3:
            raise TargetTrialAuthorityError(
                "target trial suite must have cohort, analysis and figure owners"
            )
        cohort_owners = [
            step for step in plan.steps if step.method == _COHORT_OWNER_METHOD
        ]
        suite_owners = [step for step in plan.steps if step.method == self.plan_method]
        figure_owners = [
            step for step in plan.steps if step.method == TARGET_TRIAL_FIGURE_METHOD
        ]
        if len(cohort_owners) != 1 or len(suite_owners) != 1 or len(figure_owners) != 1:
            raise TargetTrialAuthorityError(
                "target trial plan lacks a unique cohort, analysis or figure owner"
            )
        cohort_owner = cohort_owners[0]
        if cohort_owner.inputs or tuple(cohort_owner.expected_outputs) != (
            _COHORT_INPUT,
        ):
            raise TargetTrialAuthorityError(
                "target trial cohort owner drifted from host materialization"
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
            raise TargetTrialAuthorityError(
                "target trial plan drifted from signed authority: " + ", ".join(issues)
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
            raise TargetTrialAuthorityError(
                "target trial figure owner drifted from its signed sources"
            )
        self._require_rule_ref(figure)
        return step

    def governed_figure_step(self, plan: AnalysisPlan) -> AnalysisStep:
        self.governed_step(plan)
        return next(
            step for step in plan.steps if step.method == TARGET_TRIAL_FIGURE_METHOD
        )

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


def signed_target_trial_plan_claimed(plan: object) -> bool:
    """Whether a plan's primary step names the signed target trial owner."""

    return any(
        getattr(step, "planned_analysis_role", None) == "primary"
        and getattr(step, "method", None) == TARGET_TRIAL_PLAN_METHOD
        for step in tuple(getattr(plan, "steps", ()) or ())
    )


__all__ = [
    "MAX_TRIAL_COVARIATES",
    "MAX_TRIAL_TREATMENT_COLUMNS",
    "TARGET_TRIAL_AUTHORITY_KIND",
    "TARGET_TRIAL_FIGURE_METHOD",
    "TARGET_TRIAL_PLAN_METHOD",
    "TARGET_TRIAL_SUITE_MARKER",
    "TargetTrialAuthorityError",
    "TargetTrialConfirmation",
    "TargetTrialCovariate",
    "TargetTrialRuntimeAuthority",
    "signed_target_trial_plan_claimed",
]
