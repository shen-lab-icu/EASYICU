"""Host/UI quick-answer contracts. None enter the Planner output schema.

Readings contain scientific intent only. Source handles, approval receipts,
plans and numeric evidence are supplied by the host, never by the interpreter.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from easyicu.databases.research_requirements import CapabilityDecision
from easyicu.research_agent.planning.population_spec import PopulationSpec, SpecWindow


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Measure(Contract):
    concept: str = Field(default="", max_length=120, pattern=r"^[A-Za-z0-9_]*$")
    statistic: Literal["count", "proportion", "mean", "median", "iqr"]
    summary: Literal["value", "max", "min", "first", "last", "mean"] = "value"
    window: SpecWindow | None = None
    level: str | int | float | bool | None = None


class QuickReading(Contract):
    question: str = Field(min_length=1, max_length=2000)
    intent: Literal["describe", "compare", "adjusted", "prediction", "survival",
                    "causal", "clustering", "multistep"]
    population: PopulationSpec
    measure: Measure
    grouping: tuple[str, ...] = ()
    analysis_unit: Literal["icu_stay", "patient", "hospitalization"] = "icu_stay"
    adjustments: tuple[str, ...] = ()
    endpoint_kind: Literal["observed", "time_to_event", "fixed_horizon"] = "observed"
    ambiguities: tuple[str, ...] = ()


class StudyHandoff(Contract):
    action: Literal["expand_into_full_study"] = "expand_into_full_study"
    reading: QuickReading
    source: dict[str, str]
    # The exact approved effective reading travels beside the original request.
    effective_reading: QuickReading | None = None


class QuickPlan(Contract):
    schema_version: Literal["easyicu.quick_plan/1"] = "easyicu.quick_plan/1"
    reading: QuickReading
    effective_reading: QuickReading
    source_binding: str
    source_data_sha256: str
    context_sha256: str
    population: dict
    measure_column: str | None
    measure_levels: tuple[str | int | float | bool, ...] = ()
    group_column: str | None = None
    group_levels: tuple[str | int | float | bool, ...] = ()
    test: str | None = None
    qualification: tuple[str, ...]


class QuickCard(Contract):
    schema_version: Literal["easyicu.quick_card/1"] = "easyicu.quick_card/1"
    sha256: str
    source: dict[str, str]
    plan: QuickPlan
    capability: CapabilityDecision
    missing_policy: Literal["observed_values_with_missing_counts"] = "observed_values_with_missing_counts"
    interval_policy: Literal["point_summaries_only"] = "point_summaries_only"
    claim_ceiling: Literal["descriptive_unadjusted_not_causal"] = "descriptive_unadjusted_not_causal"
    approval_action: Literal["approve_quick_answer"] = "approve_quick_answer"


class QuickProposal(Contract):
    status: Literal["approval_required", "full_study", "stopped"]
    reason: str
    expand: StudyHandoff
    card: QuickCard | None = None
    capability: CapabilityDecision | None = None


class QuickApproval(Contract):
    card_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    accept_substitutions: bool = False


class EvidenceCoordinate(Contract):
    evidence_id: str
    sha256: str
    source_field: str
    source: dict[str, str]


class AnswerRow(Contract):
    statistic: str
    variable: str
    group: str
    level: str
    unit: str
    value: int | float
    evidence: EvidenceCoordinate


class AnswerSentence(Contract):
    # The sentence contains one numeric value, supplied by its exact coordinate.
    text: str
    evidence: EvidenceCoordinate


class QuickAnswer(Contract):
    schema_version: Literal["easyicu.quick_answer/1"] = "easyicu.quick_answer/1"
    status: Literal["ok", "stopped"]
    reason: str
    source: dict[str, str]
    rows: tuple[AnswerRow, ...] = ()
    sentences: tuple[AnswerSentence, ...] = ()
    expand: StudyHandoff
    claim_ceiling: Literal["descriptive_unadjusted_not_causal"] = "descriptive_unadjusted_not_causal"


class QuickStop(ValueError):
    """Stable public reason only; never expose input values in diagnostics."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)
