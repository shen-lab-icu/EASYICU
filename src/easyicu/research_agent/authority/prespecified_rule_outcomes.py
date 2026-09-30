"""Host claims for the formal outcome of a prespecified rule.

A deterministic owner applies rules fixed before execution: a class-count
criterion over a candidate grid, a minimum of observed windows, a class
description that needs a frozen solution, a planned analysis the inputs cannot
support.  The rule's outcome is a study result even when the rule selects
nothing.  Without a host claim the strict Results grammar can state none of
it, so a run whose owners all succeed can leave its required Results empty,
and an unqualified count such as the criterion's minimum reads as a solution.

An owner opts in with ``reportable_rule_outcomes``, a list of closed,
versioned envelopes.  Each envelope is typed here, its disposition must follow
from its own numbers, and each sentence is a fixed template over those typed
fields.  No owner, Planner or Writer text enters a sentence.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

RULE_OUTCOMES_KEY = "reportable_rule_outcomes"
RULE_OUTCOME_SCHEMA_VERSION = "easyicu.prespecified_rule_outcome/1"
RULE_OUTCOME_CLAIM_SCHEMA_VERSION = "easyicu.scientific_claim/4"

ReportSection = Literal["cohort", "primary"]


def _reader_count(value: int) -> str:
    return f"{value:,}"


def _reader_proportion(value: float) -> str:
    """Two decimals, keeping two significant figures below that.

    A class size reads as a proportion, never a percentage: the numeric binder
    reads "5.00%" as the class count 5 as readily as the fraction 0.05, and one
    summary carries both.
    """

    places = 2
    if value and math.isfinite(value):
        places = max(2, 1 - math.floor(math.log10(abs(value))))
    return f"{value:.{places}f}"


def _reader_series(terms: list[str]) -> str:
    if len(terms) <= 2:
        return " and ".join(terms)
    return ", ".join(terms[:-1]) + ", and " + terms[-1]


def _reader_anchor(anchor: str) -> str:
    words = anchor.split("_")
    return " ".join("ICU" if word == "icu" else word for word in words)


class _RuleOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.prespecified_rule_outcome/1"]


class ClassCountSelectionOutcome(_RuleOutcome):
    """Minimum information criterion over a prespecified class-count grid."""

    rule: Literal["information_criterion_class_count"]
    criterion: Literal["bic"]
    candidate_class_counts: list[int]
    criterion_minimum_class_count: int
    n_records: int = Field(gt=0)
    smallest_class_fraction: float = Field(gt=0.0, le=1.0)
    minimum_class_fraction: float = Field(gt=0.0, lt=1.0)
    disposition: Literal[
        "minimum_selected",
        "minimum_at_upper_boundary",
        "smallest_class_below_minimum",
    ]

    @model_validator(mode="after")
    def _disposition_follows_from_the_grid(self) -> "ClassCountSelectionOutcome":
        counts = self.candidate_class_counts
        if (
            len(counts) < 2
            or counts[0] < 2
            or any(later <= earlier for earlier, later in zip(counts, counts[1:]))
        ):
            raise ValueError("candidate class counts must be an increasing grid from 2")
        if self.criterion_minimum_class_count not in counts:
            raise ValueError("the criterion minimum must be a candidate")
        if self.criterion_minimum_class_count == counts[-1]:
            expected = "minimum_at_upper_boundary"
        elif self.smallest_class_fraction < self.minimum_class_fraction:
            expected = "smallest_class_below_minimum"
        else:
            expected = "minimum_selected"
        if self.disposition != expected:
            raise ValueError("class-count disposition contradicts its own numbers")
        return self

    @property
    def report_section(self) -> ReportSection:
        return "primary"

    def _candidates(self) -> str:
        counts = self.candidate_class_counts
        if counts == list(range(counts[0], counts[-1] + 1)):
            return f"the prespecified candidate range of {counts[0]} to {counts[-1]} classes"
        return "the prespecified candidates of " + _reader_series(
            [str(count) for count in counts]
        ) + " classes"

    def result_sentence(self) -> str:
        k = self.criterion_minimum_class_count
        opening = (
            f"Among {_reader_count(self.n_records)} records in the class model, the "
            f"Bayesian information criterion across {self._candidates()} was lowest"
        )
        if self.disposition == "minimum_at_upper_boundary":
            return (
                f"{opening} at the upper boundary of {k} classes; under the "
                "prespecified rule this is not an interior solution, and no class "
                "solution was selected."
            )
        smallest = (
            "the smallest class held a proportion of "
            f"{_reader_proportion(self.smallest_class_fraction)} of records"
        )
        if self.disposition == "smallest_class_below_minimum":
            return (
                f"{opening} at {k} classes, but {smallest}, below the prespecified "
                f"minimum of {_reader_proportion(self.minimum_class_fraction)}; under "
                "the prespecified rule no class solution was selected."
            )
        return (
            f"{opening} at {k} classes, and {smallest}; this candidate solution "
            "proceeded to the prespecified stability assessment."
        )

    def conclusion_sentence(self) -> str:
        if self.disposition == "minimum_at_upper_boundary":
            return (
                "The prespecified class-count rule selected no class solution within "
                "the candidate range, so no classes are described or interpreted."
            )
        if self.disposition == "smallest_class_below_minimum":
            return (
                "The prespecified class-count and minimum class-size rules selected no "
                "class solution, so no classes are described or interpreted."
            )
        return (
            f"The prespecified class-count rule selected a "
            f"{self.criterion_minimum_class_count}-class candidate solution for the "
            "prespecified stability assessment."
        )


class ObservedWindowEligibilityOutcome(_RuleOutcome):
    """Records meeting a minimum of observed windows on a fixed time grid."""

    rule: Literal["minimum_observed_windows"]
    anchor: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    window_start_hours: int
    window_end_hours: int
    window_width_hours: int = Field(gt=0)
    n_windows: int = Field(ge=1)
    minimum_observed_windows: int = Field(ge=1)
    input_n: int = Field(gt=0)
    included_n: int = Field(ge=0)
    excluded_n: int = Field(ge=0)

    @model_validator(mode="after")
    def _grid_and_counts_close(self) -> "ObservedWindowEligibilityOutcome":
        if (
            self.window_end_hours <= self.window_start_hours
            or self.window_end_hours - self.window_start_hours
            != self.n_windows * self.window_width_hours
        ):
            raise ValueError("observed-window grid does not tile its window")
        if self.minimum_observed_windows > self.n_windows:
            raise ValueError("the window minimum exceeds the grid")
        if self.included_n + self.excluded_n != self.input_n:
            raise ValueError("observed-window counts do not add up")
        return self

    @property
    def report_section(self) -> ReportSection:
        return "cohort"

    def _span(self) -> str:
        anchor = _reader_anchor(self.anchor)
        relation = "after" if self.window_start_hours >= 0 else "relative to"
        return (
            f"{self.window_start_hours} to {self.window_end_hours} hours {relation} "
            f"{anchor}"
        )

    def result_sentence(self) -> str:
        return (
            f"Of {_reader_count(self.input_n)} records in the longitudinal panel, "
            f"{_reader_count(self.included_n)} had at least "
            f"{self.minimum_observed_windows} of the {self.n_windows} prespecified "
            f"{self.window_width_hours}-hour windows from {self._span()} observed "
            f"and entered the class model; {_reader_count(self.excluded_n)} were "
            "excluded."
        )

    def conclusion_sentence(self) -> str:
        return (
            "Class membership was estimated only for records meeting the "
            "prespecified observed-window rule."
        )


class FrozenClassDescriptionOutcome(_RuleOutcome):
    """A class description that requires a frozen class solution."""

    rule: Literal["frozen_class_description"]
    disposition: Literal["no_frozen_solution"]

    @property
    def report_section(self) -> ReportSection:
        return "primary"

    def result_sentence(self) -> str:
        return (
            "No class solution was frozen under the prespecified rules, so no class "
            "was described and no outcome was compared between classes."
        )

    def conclusion_sentence(self) -> str:
        return (
            "No class-specific characteristics or outcomes are reported because no "
            "class solution was frozen."
        )


class PlannedAnalysisFeasibilityOutcome(_RuleOutcome):
    """A reviewed plan's analysis that the study inputs cannot support."""

    rule: Literal["planned_analysis_feasibility"]
    disposition: Literal["not_executable_from_sealed_inputs"]
    planned_analysis_role: Literal["secondary", "sensitivity"]

    @property
    def report_section(self) -> ReportSection:
        return "primary"

    def result_sentence(self) -> str:
        return (
            f"A prespecified {self.planned_analysis_role} analysis was not executable "
            "from the study inputs, and no estimate was produced for it."
        )

    def conclusion_sentence(self) -> str:
        return (
            f"No result is reported for a prespecified {self.planned_analysis_role} "
            "analysis that the study inputs could not support."
        )


PrespecifiedRuleOutcome = Annotated[
    Union[
        ClassCountSelectionOutcome,
        ObservedWindowEligibilityOutcome,
        FrozenClassDescriptionOutcome,
        PlannedAnalysisFeasibilityOutcome,
    ],
    Field(discriminator="rule"),
]
_OUTCOME_ADAPTER: TypeAdapter[Any] = TypeAdapter(PrespecifiedRuleOutcome)

#: One claim per rule and step; the id names the rule, never the study.
_CLAIM_IDS = {
    "information_criterion_class_count": "class_count_rule",
    "minimum_observed_windows": "observed_window_rule",
    "frozen_class_description": "class_description_rule",
    "planned_analysis_feasibility": "feasibility_rule",
}
#: The claim's subject and measure, as the machine claim names them.
_CLAIM_TERMS = {
    "information_criterion_class_count": (
        "candidate class count", "Bayesian information criterion",
    ),
    "minimum_observed_windows": ("observed windows", "class-model eligibility"),
    "frozen_class_description": ("frozen class solution", "class description"),
    "planned_analysis_feasibility": ("planned analysis", "executability"),
}
_CLAIM_POPULATIONS = {
    "information_criterion_class_count": "the class-model records",
    "minimum_observed_windows": "the representation input records",
    "frozen_class_description": "the analysis cohort",
    "planned_analysis_feasibility": "the study inputs",
}
_CLAIM_ROLES = {
    "information_criterion_class_count": "primary",
    "minimum_observed_windows": "auxiliary",
    "frozen_class_description": "auxiliary",
}


def validate_rule_outcome(payload: object) -> Any:
    """Parse one envelope into its closed rule type (fail closed)."""

    return _OUTCOME_ADAPTER.validate_python(payload)


def rule_outcome_payload(outcome: Any) -> dict[str, Any]:
    return outcome.model_dump(mode="json")


def derive_rule_outcome_claim_payloads(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile claim payloads from an owner's versioned rule-outcome envelopes."""

    raw = summary.get(RULE_OUTCOMES_KEY)
    if not isinstance(raw, list) or not raw:
        raise ValueError("reportable_rule_outcomes must be a non-empty list")
    if summary.get("status") != "ok":
        raise ValueError("rule outcomes require a completed owner summary")
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        outcome = validate_rule_outcome(item)
        if outcome.rule in seen:
            raise ValueError(f"rule outcome {outcome.rule!r} is repeated")
        seen.add(outcome.rule)
        subject, measure = _CLAIM_TERMS[outcome.rule]
        disposition = getattr(outcome, "disposition", "eligibility_applied")
        role = (
            outcome.planned_analysis_role
            if isinstance(outcome, PlannedAnalysisFeasibilityOutcome)
            else _CLAIM_ROLES[outcome.rule]
        )
        payloads.append(
            {
                "schema_version": RULE_OUTCOME_CLAIM_SCHEMA_VERSION,
                "claim_id": _CLAIM_IDS[outcome.rule],
                "claim_type": "prespecified_rule_outcome",
                "exposure": subject,
                "outcome": measure,
                "direction": "descriptive_only",
                "estimand": "prespecified rule outcome: "
                + disposition.replace("_", " "),
                "population": _CLAIM_POPULATIONS[outcome.rule],
                "analysis_role": role,
                "status": "supported",
                "rule_outcome": rule_outcome_payload(outcome),
            }
        )
    return payloads


__all__ = [
    "ClassCountSelectionOutcome",
    "FrozenClassDescriptionOutcome",
    "ObservedWindowEligibilityOutcome",
    "PlannedAnalysisFeasibilityOutcome",
    "PrespecifiedRuleOutcome",
    "RULE_OUTCOMES_KEY",
    "RULE_OUTCOME_CLAIM_SCHEMA_VERSION",
    "RULE_OUTCOME_SCHEMA_VERSION",
    "derive_rule_outcome_claim_payloads",
    "rule_outcome_payload",
    "validate_rule_outcome",
]
