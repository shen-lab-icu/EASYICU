"""Host claims for the formal outcome of a prespecified rule.

A deterministic owner applies rules fixed before execution: a class-count
criterion over a candidate grid, a minimum of observed windows, the resampling
stability of a selected class solution, a class description that needs a
frozen solution, a planned analysis the inputs cannot support, a
proportional-hazards test, the spline check of a linear exposure term.  The rule's outcome is a study result even when the rule selects
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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_serializer,
    model_validator,
)

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
    #: What makes a window count.  The representation owner counts a window
    #: when any SOFA-2 coordinate has an owner-available value in it, directly
    #: observed or carried forward by the SOFA-2 owner; other coordinates do
    #: not count.  An envelope written before this field omits it.
    window_evidence: Literal["any_available_sofa2_score"] | None = None

    @model_serializer(mode="wrap")
    def _preserve_unstated_window_evidence(self, handler):
        payload = handler(self)
        if self.window_evidence is None:
            # A registered claim embeds the envelope as written; an envelope
            # that never stated its window evidence replays without the key.
            payload.pop("window_evidence", None)
        return payload

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
        if self.window_evidence is None:
            return (
                f"Of {_reader_count(self.input_n)} records in the study cohort, "
                f"{_reader_count(self.included_n)} had at least "
                f"{self.minimum_observed_windows} of the {self.n_windows} prespecified "
                f"{self.window_width_hours}-hour windows from {self._span()} observed "
                f"and entered the class model; {_reader_count(self.excluded_n)} were "
                "excluded."
            )
        return (
            f"Of {_reader_count(self.input_n)} records in the study cohort, "
            f"{_reader_count(self.included_n)} had at least one SOFA-2 score "
            f"available in at least {self.minimum_observed_windows} of the "
            f"{self.n_windows} prespecified {self.window_width_hours}-hour windows "
            f"from {self._span()} and entered the class model; "
            f"{_reader_count(self.excluded_n)} were excluded."
        )

    def conclusion_sentence(self) -> str:
        if self.window_evidence is None:
            return (
                "Class membership was estimated only for records meeting the "
                "prespecified observed-window rule."
            )
        return (
            "Class membership was estimated only for records with at least one "
            "SOFA-2 score available in the prespecified minimum number of windows."
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


class ProportionalHazardsTestOutcome(_RuleOutcome):
    """The prespecified proportional-hazards test and the estimate it permits."""

    rule: Literal["proportional_hazards_test"]
    diagnostic: Literal["schoenfeld_residual_test"]
    alpha: float = Field(gt=0.0, lt=1.0)
    # A test statistic far in the tail can underflow its p value to zero.
    global_p_value: float = Field(ge=0.0, le=1.0)
    exposure_p_value: float = Field(ge=0.0, le=1.0)
    disposition: Literal["assumption_rejected", "assumption_not_rejected"]

    @model_validator(mode="after")
    def _disposition_follows_from_the_test(self) -> "ProportionalHazardsTestOutcome":
        rejected = min(self.global_p_value, self.exposure_p_value) < self.alpha
        expected = "assumption_rejected" if rejected else "assumption_not_rejected"
        if self.disposition != expected:
            raise ValueError("proportional-hazards disposition contradicts its own p values")
        return self

    @property
    def report_section(self) -> ReportSection:
        return "primary"

    def result_sentence(self) -> str:
        # The decision and its threshold only: a p value far in the tail has no
        # reader display the numeric binder can trace ("p < 0.001" cites a
        # threshold, not a registered value).  The owner's PH diagnostics keep
        # the exact p values.
        # Whether the model was adjusted is the association claims' to say;
        # this rule decides only between hazard ratios constant over
        # follow-up and interval-specific ones.
        alpha = _reader_proportion(self.alpha)
        if self.disposition == "assumption_rejected":
            return (
                "The Schoenfeld residual test rejected the proportional-hazards "
                f"assumption at the prespecified alpha of {alpha}, so interval-specific "
                "hazard ratios are the primary estimates instead of hazard ratios "
                "constant over follow-up."
            )
        return (
            "The Schoenfeld residual test did not reject the proportional-hazards "
            f"assumption at the prespecified alpha of {alpha}, so the primary "
            "estimates are hazard ratios constant over follow-up."
        )

    def conclusion_sentence(self) -> str:
        if self.disposition == "assumption_rejected":
            return (
                "Because the proportional-hazards assumption was rejected, the "
                "association is described by interval-specific hazard ratios rather "
                "than hazard ratios constant over follow-up."
            )
        return (
            "The proportional-hazards assumption was not rejected, so the "
            "association is summarized by hazard ratios constant over follow-up."
        )


#: Why the spline check of a linear exposure term had no result.
FunctionalFormNotAssessableReason = Literal["tied_knots", "spline_model_not_estimable"]
_NOT_ASSESSABLE_WORDS = {
    "tied_knots": "the exposure had too few distinct values for its three spline knots",
    "spline_model_not_estimable": "the spline model had no estimate",
}


class FunctionalFormTestOutcome(_RuleOutcome):
    """The prespecified spline check of a linear exposure term, and the estimate it left.

    A likelihood-ratio test of a restricted cubic spline against the linear
    term judges linearity at a prespecified alpha.  When it rejects and the
    proportional-hazards test does not, the spline's hazard ratios at two
    percentiles of the exposure relative to its median replace the per-step
    estimate; when the PH test rejected, the interval-specific per-step
    estimates remain the result and describe an average log-linear trend.  A
    check without a result changes nothing and says why.  The sentence names
    no shape: the rule decides linearity, not a form.
    """

    rule: Literal["functional_form_test"]
    diagnostic: Literal["restricted_cubic_spline_likelihood_ratio_test"]
    alpha: float = Field(gt=0.0, lt=1.0)
    #: Absent exactly when the check had no result.
    nonlinearity_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    disposition: Literal[
        "linearity_rejected", "linearity_not_rejected", "not_assessable"
    ]
    not_assessable_reason: FunctionalFormNotAssessableReason | None = None
    #: The estimate the suite's rules left as its result.
    primary_estimate: Literal[
        "per_step_hazard_ratio",
        "spline_percentile_contrasts",
        "interval_per_step_hazard_ratios",
    ]

    @model_serializer(mode="wrap")
    def _omit_unstated_fields(self, handler):
        payload = handler(self)
        for name in ("nonlinearity_p_value", "not_assessable_reason"):
            if getattr(self, name) is None:
                payload.pop(name, None)
        return payload

    @model_validator(mode="after")
    def _disposition_follows_from_the_test(self) -> "FunctionalFormTestOutcome":
        assessed = self.disposition != "not_assessable"
        if assessed != (self.nonlinearity_p_value is not None) or assessed == (
            self.not_assessable_reason is not None
        ):
            raise ValueError(
                "a functional-form test states its p value, or why it had none"
            )
        if assessed:
            rejected = self.nonlinearity_p_value < self.alpha
            expected = "linearity_rejected" if rejected else "linearity_not_rejected"
            if self.disposition != expected:
                raise ValueError(
                    "functional-form disposition contradicts its own p value"
                )
        rejected = self.disposition == "linearity_rejected"
        if (self.primary_estimate == "spline_percentile_contrasts") != (
            rejected and self.primary_estimate != "interval_per_step_hazard_ratios"
        ):
            raise ValueError(
                "the spline contrasts are the result exactly when linearity is rejected "
                "and the per-step estimate is constant over follow-up"
            )
        return self

    @property
    def report_section(self) -> ReportSection:
        return "primary"

    def result_sentence(self) -> str:
        if self.not_assessable_reason is not None:
            return (
                "The restricted cubic spline check of the linear exposure term had no "
                f"result, because {_NOT_ASSESSABLE_WORDS[self.not_assessable_reason]}, "
                "so the per-step estimates stand unchecked."
            )
        alpha = _reader_proportion(self.alpha)
        if self.disposition == "linearity_not_rejected":
            return (
                "The restricted cubic spline check did not reject a linear association "
                f"with the log hazard at the prespecified alpha of {alpha}, so the "
                "per-step estimates stand."
            )
        if self.primary_estimate == "spline_percentile_contrasts":
            return (
                "The restricted cubic spline check rejected a linear association with "
                f"the log hazard at the prespecified alpha of {alpha}, so the spline's "
                "hazard ratios at two prespecified percentiles of the exposure relative "
                "to its median are the primary estimates instead of one per-step "
                "hazard ratio."
            )
        return (
            "The restricted cubic spline check rejected a linear association with the "
            f"log hazard at the prespecified alpha of {alpha}; the interval-specific "
            "per-step hazard ratios remain the primary estimates and describe an "
            "average log-linear trend within each interval."
        )

    def conclusion_sentence(self) -> str:
        if self.not_assessable_reason is not None:
            return (
                "The linearity of the association was not checked, because "
                f"{_NOT_ASSESSABLE_WORDS[self.not_assessable_reason]}."
            )
        if self.disposition == "linearity_not_rejected":
            return (
                "A linear association with the log hazard was not rejected, so the "
                "association is summarized per step of the exposure."
            )
        if self.primary_estimate == "spline_percentile_contrasts":
            return (
                "Because a linear association was rejected, the association is "
                "described at two percentiles of the exposure relative to its median "
                "rather than per step."
            )
        return (
            "A linear association was rejected, so each interval-specific per-step "
            "hazard ratio describes only an average log-linear trend."
        )


def _reader_pair(value: float, minimum: float) -> tuple[str, str]:
    """A mean and its prespecified minimum, with places enough to differ.

    At two places a mean of 0.696 and a minimum of 0.70 would both read
    "0.70" in a sentence saying one is below the other.  Four places at most:
    the reader rounding (``reader_numeric_display``) keeps them as written.
    """

    places = max(
        len(_reader_proportion(number).partition(".")[2]) for number in (value, minimum)
    )
    while (
        value != minimum
        and places < 4
        and f"{value:.{places}f}" == f"{minimum:.{places}f}"
    ):
        places += 1
    return f"{value:.{places}f}", f"{minimum:.{places}f}"


class ClassSolutionStabilityOutcome(_RuleOutcome):
    """The prespecified resampling-stability rule for a selected class solution.

    Every planned subsample refit must realize every class, and the mean
    agreement of the refits with the selected solution must reach the
    planner's minimum, before the solution is frozen.
    """

    rule: Literal["class_solution_stability"]
    metric: Literal["adjusted_rand_index"]
    selected_class_count: int = Field(ge=2)
    planned_resamples: int = Field(ge=2)
    successful_resamples: int = Field(ge=0)
    minimum_successful_resamples: int = Field(ge=2)
    # The mean over the successful refits; with too few of them the
    # prespecified quantity does not exist.  A report-only design has no
    # minimum, so only its refit count can decide.
    mean_stability: float | None = Field(ge=-1.0, le=1.0)
    minimum_mean_stability: float | None = Field(ge=-1.0, le=1.0)
    disposition: Literal[
        "stability_threshold_met",
        "stability_below_threshold",
        "too_few_successful_refits",
    ]

    @model_validator(mode="after")
    def _disposition_follows_from_the_refits(self) -> "ClassSolutionStabilityOutcome":
        if max(
            self.successful_resamples, self.minimum_successful_resamples
        ) > self.planned_resamples:
            raise ValueError("refit counts exceed the planned resamples")
        if self.successful_resamples < self.minimum_successful_resamples:
            if self.mean_stability is not None:
                raise ValueError("an unestablished stability has no mean")
            expected = "too_few_successful_refits"
        elif self.mean_stability is None:
            raise ValueError("an established stability needs its mean")
        elif self.minimum_mean_stability is None:
            raise ValueError("a stability decision needs its prespecified minimum")
        elif self.mean_stability >= self.minimum_mean_stability:
            expected = "stability_threshold_met"
        else:
            expected = "stability_below_threshold"
        if self.disposition != expected:
            raise ValueError("stability disposition contradicts its own numbers")
        return self

    @property
    def report_section(self) -> ReportSection:
        return "primary"

    def result_sentence(self) -> str:
        opening = (
            "In the prespecified resampling stability assessment of the "
            f"{self.selected_class_count}-class candidate solution"
        )
        if self.disposition == "too_few_successful_refits":
            return (
                f"{opening}, {self.successful_resamples} of {self.planned_resamples} "
                "subsample refits converged and realized every class, fewer than the "
                f"{self.minimum_successful_resamples} the rule requires; stability was "
                "not established, and under the prespecified rule no class solution "
                "was frozen."
            )
        assert self.mean_stability is not None
        assert self.minimum_mean_stability is not None
        mean, minimum = _reader_pair(self.mean_stability, self.minimum_mean_stability)
        agreement = (
            f"{opening}, the mean adjusted Rand index across "
            f"{self.successful_resamples} subsample refits was {mean}"
        )
        if self.disposition == "stability_below_threshold":
            return (
                f"{agreement}, below the prespecified minimum of {minimum}; under the "
                "prespecified rule no class solution was frozen."
            )
        return (
            f"{agreement}, at or above the prespecified minimum of {minimum}, and the "
            "candidate solution was frozen."
        )

    def conclusion_sentence(self) -> str:
        if self.disposition == "stability_threshold_met":
            return (
                f"The {self.selected_class_count}-class solution met the prespecified "
                "resampling stability rule and was frozen for class description."
            )
        return (
            f"The {self.selected_class_count}-class candidate solution did not meet "
            "the prespecified resampling stability rule, so no classes are described "
            "or interpreted."
        )


PrespecifiedRuleOutcome = Annotated[
    Union[
        ClassCountSelectionOutcome,
        ObservedWindowEligibilityOutcome,
        FrozenClassDescriptionOutcome,
        PlannedAnalysisFeasibilityOutcome,
        ProportionalHazardsTestOutcome,
        ClassSolutionStabilityOutcome,
        FunctionalFormTestOutcome,
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
    "proportional_hazards_test": "proportional_hazards_rule",
    "class_solution_stability": "class_stability_rule",
    "functional_form_test": "functional_form_rule",
}
#: The claim's subject and measure, as the machine claim names them.
_CLAIM_TERMS = {
    "information_criterion_class_count": (
        "candidate class count", "Bayesian information criterion",
    ),
    "minimum_observed_windows": ("observed windows", "class-model eligibility"),
    "frozen_class_description": ("frozen class solution", "class description"),
    "planned_analysis_feasibility": ("planned analysis", "executability"),
    "proportional_hazards_test": ("proportional hazards", "Schoenfeld residual test"),
    "class_solution_stability": ("selected class solution", "resampling stability"),
    "functional_form_test": (
        "linear exposure term",
        "restricted cubic spline likelihood-ratio test",
    ),
}
_CLAIM_POPULATIONS = {
    "information_criterion_class_count": "the class-model records",
    "minimum_observed_windows": "the representation input records",
    "frozen_class_description": "the analysis cohort",
    "planned_analysis_feasibility": "the study inputs",
    "proportional_hazards_test": "the survival model records",
    "class_solution_stability": "the class-model records",
    "functional_form_test": "the survival model records",
}
_CLAIM_ROLES = {
    "information_criterion_class_count": "primary",
    "minimum_observed_windows": "auxiliary",
    "frozen_class_description": "auxiliary",
    "proportional_hazards_test": "primary",
    "class_solution_stability": "primary",
    "functional_form_test": "primary",
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
    "ClassSolutionStabilityOutcome",
    "FrozenClassDescriptionOutcome",
    "FunctionalFormNotAssessableReason",
    "FunctionalFormTestOutcome",
    "ObservedWindowEligibilityOutcome",
    "PlannedAnalysisFeasibilityOutcome",
    "PrespecifiedRuleOutcome",
    "ProportionalHazardsTestOutcome",
    "RULE_OUTCOMES_KEY",
    "RULE_OUTCOME_CLAIM_SCHEMA_VERSION",
    "RULE_OUTCOME_SCHEMA_VERSION",
    "derive_rule_outcome_claim_payloads",
    "rule_outcome_payload",
    "validate_rule_outcome",
]
