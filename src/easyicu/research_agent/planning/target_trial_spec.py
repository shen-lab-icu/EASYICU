"""The target trial a causal study emulates, as its study setup states it.

A causal question about when to start a treatment is planned as the emulation
of a target trial: who is eligible at time zero, two strategies -- start the
treatment within a grace period after time zero, or do not start it within
that period -- and the risk of a fixed-horizon outcome under each.  The
protocol comes first: the study setup states it from the host's menus, each
element in the words that state it (``quote``) with where they come from
(``source``), and the host decides how each element reaches the emulation
(:mod:`.target_trial_compile`).

The spec states the trial's own elements and nothing the host owns.
Eligibility is the study's population (:mod:`.population_spec`), compiled at
the trial's time zero; the indication names the population criteria that make
both strategies plausible for every eligible stay.  How the emulation is
estimated -- cloning, artificial censoring, weights, resampling -- is the
host's.  Hours count from ICU admission, and windows are ``[start, end)``.

A coordinate the researcher did not state is written with source
``design_choice``: the approval card asks the researcher to confirm it, and so
does a confounder the study setup proposed.  An element no field expresses is
stated as ``not_typed`` with the reason and is never dropped: the card shows
each one, and one that changes a strategy (a dose, a continued use, a start
rule that follows the patient's course) stops the trial when it is compiled.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..contracts.time_varying_exposure import (
    G_METHOD_OUTPUTS, G_METHOD_SOURCE_KEY, GMethodError, GMethodSpec, g_method_for_step,
)
from ..contracts.cohort_product_keys import sole_typed_cohort_input

TARGET_TRIAL_SPEC_SCHEMA_VERSION = "easyicu.target_trial_spec/1"
MAX_TREATMENT_CONCEPTS = 4
MAX_INDICATION_CRITERIA = 4
MAX_TRIAL_CONFOUNDERS = 24
MAX_NOT_TYPED_ELEMENTS = 6
#: A week of hours: no time zero or grace period of the spec reaches further.
MAX_TRIAL_HOURS = 168

ElementSource = Literal["question", "conversation", "design_choice"]


class _Element(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The words that state the element, verbatim; for a design choice, the
    #: words of the proposal the researcher is asked to confirm.
    quote: str = Field(min_length=2, max_length=160)
    source: ElementSource


def _token(value: str, what: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{what} must be named")
    return cleaned


class TrialTreatment(_Element):
    """The treatment whose start the strategies time.

    ``concepts`` are event-status concepts read together: the treatment
    starts at the first hour any of them is recorded.  ``treatment_class``
    names the class of drug the study means, from the capture registry's
    closed vocabulary, so the host can tell whether the concepts record every
    drug of that class.
    """

    concepts: tuple[str, ...] = Field(min_length=1, max_length=MAX_TREATMENT_CONCEPTS)
    treatment_class: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")

    @field_validator("concepts")
    @classmethod
    def _named_once(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(_token(item, "a treatment concept") for item in value)
        if len(set(names)) != len(names):
            raise ValueError("name each treatment concept once")
        return names


class TrialStrategies(_Element):
    """The reader labels of the two strategies, whose kinds are fixed.

    One strategy starts the treatment within the grace period; the other does
    not start it within that period and leaves it unrestricted afterwards.
    """

    initiate_label: str = Field(min_length=2, max_length=80)
    defer_label: str = Field(min_length=2, max_length=80)

    @model_validator(mode="after")
    def _distinct(self) -> "TrialStrategies":
        if (
            self.initiate_label.strip().casefold()
            == self.defer_label.strip().casefold()
        ):
            raise ValueError("the two strategies need different labels")
        return self


class TrialTimeZero(_Element):
    """The hour after ICU admission at which eligibility and follow-up start."""

    hours_after_icu_admission: int = Field(ge=0, le=MAX_TRIAL_HOURS)


class TrialGracePeriod(_Element):
    """The hours after time zero within which the treatment may start."""

    hours: int = Field(ge=1, le=MAX_TRIAL_HOURS)


class TrialOutcome(_Element):
    """The fixed-horizon endpoint whose risk the strategies are compared on."""

    endpoint: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")


class TrialIndication(_Element):
    """The population criteria that state why either strategy is plausible."""

    criterion_ids: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_INDICATION_CRITERIA
    )

    @field_validator("criterion_ids")
    @classmethod
    def _criterion_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        ids = tuple(_token(item, "an indication criterion") for item in value)
        if len(set(ids)) != len(ids):
            raise ValueError("name each indication criterion once")
        return ids


class TrialConfounder(BaseModel):
    """One baseline covariate with its clinical confounding rationale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    clinical_rationale: str = Field(min_length=16, max_length=500)
    #: Who named it: the question, the conversation, or the setup's proposal.
    source: ElementSource

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _token(value, "a confounder")

    @field_validator("clinical_rationale")
    @classmethod
    def _collapse_whitespace(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class TrialNotTyped(_Element):
    """An element of the study that no field of this spec expresses."""

    why: str = Field(min_length=8, max_length=240)
    #: Whether it changes what a strategy does.
    affects_strategy: bool


class TargetTrialSpec(BaseModel):
    """The protocol elements of one target trial; at most one per study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    treatment: TrialTreatment
    strategies: TrialStrategies
    time_zero: TrialTimeZero
    grace_period: TrialGracePeriod
    outcome: TrialOutcome
    indication: Optional[TrialIndication] = None
    confounders: tuple[TrialConfounder, ...] = Field(
        default=(), max_length=MAX_TRIAL_CONFOUNDERS
    )
    not_typed: tuple[TrialNotTyped, ...] = Field(
        default=(), max_length=MAX_NOT_TYPED_ELEMENTS
    )

    @field_validator("confounders")
    @classmethod
    def _confounders_once(
        cls, value: tuple[TrialConfounder, ...]
    ) -> tuple[TrialConfounder, ...]:
        names = [item.name for item in value]
        if len(set(names)) != len(names):
            raise ValueError("name each confounder once")
        return value


__all__ = [
    "MAX_INDICATION_CRITERIA",
    "MAX_NOT_TYPED_ELEMENTS",
    "MAX_TREATMENT_CONCEPTS",
    "MAX_TRIAL_CONFOUNDERS",
    "MAX_TRIAL_HOURS",
    "TARGET_TRIAL_SPEC_SCHEMA_VERSION",
    "ElementSource",
    "TargetTrialSpec",
    "TrialConfounder",
    "TrialGracePeriod",
    "TrialIndication",
    "TrialNotTyped",
    "TrialOutcome",
    "TrialStrategies",
    "TrialTimeZero",
    "TrialTreatment",
]


def require_g_method_source(spec: GMethodSpec, source: Mapping[str, Any] | None, *, database: str) -> None:
    from easyicu.databases.profiles import get_database_profile

    if not isinstance(source, Mapping):
        raise GMethodError("g_methods_person_period_unavailable", "The selected source has no owner-verified person-period structure.")
    allowed = {"schema_version", "producer", "database", "structure", "identity_unit", "identity_column",
               "columns", "pre_treatment_columns", "history_columns", "interval_hours", "time_zero",
               "source_sha256", "specification_sha256"}
    if set(source) - allowed:
        raise GMethodError("g_methods_source_receipt_mismatch", "Source receipts contain only aggregate construction coordinates, never rows or source paths.")
    try:
        declared = get_database_profile(spec.database).key
        selected = get_database_profile(database).key
        actual = get_database_profile((source or {}).get("database", "")).key
    except (ValueError, KeyError) as exc:
        raise GMethodError("g_methods_source_database_mismatch", "A registered selected database and matching source receipt are required.") from exc
    if declared != selected or actual != selected:
        raise GMethodError("g_methods_source_database_mismatch", "G-method source and selected database differ.")
    source = source or {}
    rosters = ("columns", "history_columns", "pre_treatment_columns")
    if any(not isinstance(source.get(key), (list, tuple))
           or len(source[key]) > 128
           or not all(isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                      for name in source[key]) for key in rosters):
        raise GMethodError("g_methods_source_receipt_mismatch", "Source column rosters must be explicit lists of names.")
    if (source.get("schema_version") != "easyicu.g_methods_source/1"
            or source.get("producer") not in {"target_trial", "time_varying"}
            or source.get("structure") != "person_period"):
        raise GMethodError("g_methods_person_period_unavailable", "The selected source has no owner-verified person-period structure.")
    if source.get("identity_unit") != "person" or source.get("identity_column") != spec.identity_column:
        raise GMethodError("g_methods_person_identity_unavailable", "The history must identify one contiguous treatment stream per person; stay keys alone cannot authorize person-cluster inference.")
    if not set(spec.confounder_history) <= set(source.get("history_columns", ())):
        raise GMethodError("g_methods_confounder_history_unavailable", "The selected source does not provide the declared confounder history.")
    if not {*spec.confounder_history, *spec.baseline_columns} <= set(source.get("pre_treatment_columns", ())):
        raise GMethodError("g_methods_history_timing_unavailable", "Confounder measurements must precede each treatment assignment.")
    if source.get("interval_hours") != spec.interval_hours or source.get("time_zero") != spec.time_zero:
        raise GMethodError("g_methods_history_timing_unavailable", "The source interval grid and time zero differ from the reviewed plan.")
    if not set(spec.required_columns) <= set(source.get("columns", ())):
        raise GMethodError("g_methods_source_columns_unavailable", "The selected source lacks declared g-method columns.")
    if (source.get("specification_sha256") != spec.sha256
            or re.fullmatch(r"[0-9a-f]{64}", str(source.get("source_sha256", ""))) is None):
        raise GMethodError("g_methods_source_receipt_mismatch", "Source construction must bind the exact reviewed specification and upstream input digest.")


def require_g_method_step(step: Any) -> GMethodSpec | None:
    spec = g_method_for_step(step)
    if spec is None:
        return None
    if (tuple(step.expected_outputs) != G_METHOD_OUTPUTS
            or not sole_typed_cohort_input(step)
            or not set(spec.required_columns) <= set(step.inputs)
            or step.planned_analysis_role not in {"primary", "secondary", "sensitivity"}
            or step.scientific_capability not in {None, "causal_g_methods_v1"}
            or any(getattr(step, key, None) for key in (
                "model_requirements", "family_primary_result_requirement", "table_one_spec",
                "cohort_definition_spec", "measurement_audit_spec", "robustness_replay_spec",
            ))):
        raise GMethodError("g_methods_step_contract_invalid", "Bind one typed person-period input, the exact columns, estimates, diagnostics, receipt and checklist.")
    return spec


def require_g_method_plan(context: Any, plan: Any) -> None:
    for step in plan.steps:
        spec = require_g_method_step(step)
        if spec is not None:
            if ((context.primary_exposure and context.primary_exposure != spec.treatment_column)
                    or (context.target_outcome and context.target_outcome != spec.outcome_column)):
                raise GMethodError("g_methods_question_mismatch", "The g-method treatment/outcome differs from the research question.")
            source = context.cohort.provenance.get(G_METHOD_SOURCE_KEY)
            require_g_method_source(spec, source, database=context.cohort.database)
