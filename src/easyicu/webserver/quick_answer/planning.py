"""Compile a narrow typed reading without a Planner, extraction or model call."""

from __future__ import annotations

import re

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import (
    StudyRequirements, ThresholdRequirement, assess_study,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.closed_levels import validate_closed_scalar_levels
from easyicu.research_agent.gates.figure_privacy import _is_identifier_name, _is_time_name
from easyicu.research_agent.planning.population_compile import compile_population, population_spec_concepts
from easyicu.research_agent.research_context.materialization_window import context_column_windows
from easyicu.webserver.design_variance_basis import counts_only_request

from .contracts import QuickCard, QuickPlan, QuickProposal, QuickReading, QuickStop, StudyHandoff
from .source import PreparedSource, require_source


# Defence against a typed reader dropping a consequential part of the question.
# These guards only refuse; they never assign a measure or clinical definition.
_COMPLEXITY = (
    ("adjusted", r"adjust|account(?:ing)?\s+for|controll?ing\s+for|independent\s+(?:effect|association)|"
                 r"holding\s+.+?constant|balanc(?:e|ing)|matched|校正|调整|匹配|控制.+?(?:年龄|性别|混杂)|排除.+?影响"),
    ("survival", r"surviv|censor|hazard|kaplan|time.to.event|still\s+alive|"
                 r"(?:by|within|at)\s+(?:day|week|month)\s*\d|\d+[- ]day\s+(?:death|mortality)|"
                 r"(?:death|died|mortality).{0,30}(?:days|weeks)|生存|存活|删失|随访|\d+\s*(?:天|日|周|月).{0,6}死亡"),
    ("prediction", r"predict|forecast|预测"),
    ("causal", r"causal|causation|propensity|effect\s+of|因果|倾向评分|导致"),
    ("clustering", r"cluster|phenotyp|聚类|分型"),
    ("multistep", r"multi.step|nested\s+cohort|landmark|先.+再|多阶段|嵌套队列"),
    ("interval", r"confidence\s+interval|置信区间"),
)
_OUTCOMES = {"hospital_death", "icu_death", "death_within_72h_after_icu_discharge"}


def bind_concept_names(reading: QuickReading) -> QuickReading:
    """Name the reading's measure and groups by their EasyICU concept ids.

    The interpreter sees no concept roster, so it may type a clinical name
    ("lactate", "gender") where the materializer and the prepared source take
    an id (``lact``, ``sex``); pilot1 10-26 stopped a fully typed reading that
    way. An id stays as typed. Any other name binds only when the question
    reader reads it as exactly one concept (``concept_named_by``); otherwise it
    stays as typed and the source lookup refuses it, as before.
    """
    from easyicu.concept.loader import load_concept_dict_cached
    from easyicu.webserver.study_intent import concept_named_by

    known = load_concept_dict_cached().keys() | _OUTCOMES

    def bind(name: str) -> str:
        return name if not name or name in known else concept_named_by(name) or name

    concept = bind(reading.measure.concept)
    grouping = tuple(bind(name) for name in reading.grouping)
    if concept == reading.measure.concept and grouping == reading.grouping:
        return reading
    return reading.model_copy(update={
        "measure": reading.measure.model_copy(update={"concept": concept}), "grouping": grouping,
    })


def qualification_stop(reading: QuickReading) -> str | None:
    from easyicu.outcome_availability import fixed_horizon_mortality_endpoint

    if reading.intent not in {"describe", "compare"}:
        return "quick_requires_" + reading.intent + "_study"
    if reading.adjustments:
        return "quick_requires_adjusted_study"
    if reading.endpoint_kind != "observed":
        return "quick_requires_survival_study"
    if fixed_horizon_mortality_endpoint(reading.measure.concept) is not None:
        return "quick_requires_survival_study"
    from easyicu.webserver.question_named_levels import undistinguished_named_levels

    # A named kind the prepared data do not separate (invasive ventilation)
    # is never answered on every kind, as a full study's launch refuses it.
    if undistinguished_named_levels(reading.question):
        return "quick_named_level_undistinguished"
    for kind, pattern in _COMPLEXITY:
        if kind == "interval" and counts_only_request(reading.question):
            continue
        if re.search(pattern, reading.question, re.IGNORECASE):
            return "quick_requires_" + kind + "_study"
    if reading.ambiguities:
        return "quick_reading_ambiguous"
    if len(reading.grouping) > 1:
        return "quick_grouping_scope_exceeded"
    if len(reading.population.criteria) > 1 or any(
        item.kind not in {"age_years", "measurement", "condition_present", "first_icu_stay",
                          "first_icu_stay_per_hospitalization"}
        for item in reading.population.criteria
    ):
        return "quick_requires_multistep_study"
    if any(item.source not in {"question", "study_wording"} or item.quote not in reading.question
           for item in reading.population.criteria):
        return "quick_reading_ambiguous"
    if reading.intent == "compare" and (not reading.grouping or reading.measure.statistic == "count"):
        return "quick_comparison_ambiguous"
    if not reading.measure.concept and reading.measure.statistic != "count":
        return "quick_reading_ambiguous"
    return None


def _requirements(reading: QuickReading) -> StudyRequirements:
    thresholds = []
    for item in reading.population.criteria:
        if item.kind == "age_years":
            thresholds.extend(ThresholdRequirement(concept="age", operator=operator, value=value)
                              for operator, value in ((">=", item.min_years), ("<=", item.max_years))
                              if value is not None)
        elif item.kind == "measurement" and item.concept in {"age", "weight"}:
            thresholds.append(ThresholdRequirement(concept=item.concept, operator=item.op, value=item.value))
    concept = reading.measure.concept
    return StudyRequirements(
        family="descriptive_epidemiology", analysis_unit=reading.analysis_unit,
        outcome=concept if concept in _OUTCOMES else None,
        concepts=tuple(dict.fromkeys((*([] if concept in _OUTCOMES or not concept else [concept]),
                                     *reading.grouping, *population_spec_concepts(reading.population)))),
        thresholds=tuple(thresholds),
    )


def _effective(reading, decision):
    effective = reading
    for item in decision.substitutions:
        if item.capability == "outcome_meaning":
            effective = effective.model_copy(update={
                "measure": effective.measure.model_copy(update={"concept": item.substitute}),
            })
        elif item.capability == "analysis_unit":
            effective = effective.model_copy(update={"analysis_unit": item.substitute})
        else:
            # Population band changes must pass the normal db6a population
            # approval compiler; a quick host must not invent a new predicate.
            raise QuickStop("quick_substitution_requires_full_study")
    return QuickReading.model_validate(effective.model_dump(mode="json"))


def _variable(source, measure):
    if not measure.concept:
        return None
    candidates = [v for v in source.context.variables
                  if (v.name == measure.concept if measure.summary == "value" else
                      v.name == f"{measure.concept}_{measure.summary}" and v.source_concept == measure.concept)]
    if len(candidates) != 1:
        raise QuickStop("quick_measure_unresolved")
    variable = candidates[0]
    if variable.role.value in {"id", "index", "time", "meta"} or _is_identifier_name(variable.name) or _is_time_name(variable.name):
        raise QuickStop("quick_identifier_forbidden")
    window = context_column_windows(source.context).get(variable.name)
    if measure.summary != "value" and measure.window is None:
        raise QuickStop("quick_window_unresolved")
    if measure.window is not None:
        if window is None or (window.anchor, window.start_hours, window.end_hours) != (
            "icu_admission", measure.window.start_hours, measure.window.end_hours,
        ):
            raise QuickStop("quick_window_unresolved")
    elif window is not None:
        raise QuickStop("quick_window_unresolved")
    return variable


def _levels(variable):
    if variable is None:
        return ()
    levels = (variable.observed_domain or {}).get("levels", [])
    try:
        levels = validate_closed_scalar_levels(levels, label="quick levels")
    except ValueError as exc:
        raise QuickStop("quick_levels_unresolved") from exc
    if len(levels) > 12 or len(set(levels)) != len(levels) or len({str(value) for value in levels}) != len(levels):
        raise QuickStop("quick_grouping_scope_exceeded")
    if any(isinstance(value, str) and (len(value) > 80 or re.search(r"[/\\\n]|\d{6,}", value))
           for value in levels):
        raise QuickStop("quick_identifier_forbidden")
    return tuple(levels)


def population_receipt(population) -> dict:
    """A public decision and private-proof digest, with no source receipt data."""
    return {
        "compile_sha256": population.sha256(),
        "criteria": [{"criterion": item.criterion.model_dump(mode="json"),
                      "disposition": item.disposition, "reason": item.reason}
                     for item in population.criteria],
    }


def _plan(reading, effective, source):
    from .contracts import Measure

    context = source.context.model_copy(update={"research_question": reading.question})
    population = compile_population(effective.population, context)
    if any(not item.applied for item in population.criteria):
        raise QuickStop("quick_population_not_prepared")
    if effective.analysis_unit != source.analysis_unit:
        raise QuickStop("quick_unit_not_prepared")
    variable = _variable(source, effective.measure)
    levels = _levels(variable)
    categorical = effective.measure.statistic in {"count", "proportion"}
    if variable is not None and effective.measure.statistic == "proportion" and not levels:
        raise QuickStop("quick_levels_unresolved")
    if not categorical and levels:
        raise QuickStop("quick_measure_ambiguous")
    if effective.measure.level is not None and effective.measure.level not in levels:
        raise QuickStop("quick_level_unresolved")
    group = (_variable(source, Measure(concept=effective.grouping[0], statistic="count"))
             if effective.grouping else None)
    group_levels = _levels(group)
    if group is not None and (len(group_levels) < 2 or group == variable):
        raise QuickStop("quick_grouping_ambiguous")
    if {str(level) for level in group_levels} & {"Overall", "Missing", "Comparison"}:
        raise QuickStop("quick_grouping_ambiguous")
    test = None
    if effective.intent == "compare":
        if len(group_levels) != 2:
            raise QuickStop("quick_two_groups_required")
        # The shared categorical test compares the complete distribution. It
        # tests a selected proportion only when that distribution is binary.
        if categorical and effective.measure.level is not None and len(levels) != 2:
            raise QuickStop("quick_comparison_ambiguous")
        if source.patient_column is None:
            raise QuickStop("quick_independent_units_unverified")
        test = ("chi_square_with_fisher_exact_for_sparse_2x2" if categorical else
                "welch_t_or_anova" if effective.measure.statistic == "mean" else "mann_whitney_or_kruskal")
    return QuickPlan(
        reading=reading, effective_reading=effective, source_binding=source.binding,
        source_data_sha256=source.data_sha256, context_sha256=canonical_sha256(source.context.model_dump(mode="json")),
        population=population_receipt(population), measure_column=variable.name if variable else None,
        measure_levels=levels, group_column=group.name if group else None, group_levels=group_levels, test=test,
        qualification=("typed_reading_unambiguous", "single_prepared_population", "single_measure",
                       "host_descriptive_kernel", "standard_two_group_test" if test else "point_summaries_only"),
    )


def card_digest(card: QuickCard) -> str:
    return canonical_sha256(card.model_dump(mode="json", exclude={"sha256"}))


def prepare_quick_answer(reading: QuickReading, *, source: PreparedSource, provider: object,
                         question: str | None = None) -> QuickProposal:
    """Host entrypoint; pass the original turn as ``question`` after interpretation.

    Full-study dispositions carry the reading for normal setup/Planner; stops
    retain a precise remedy. Neither disposition executes or calls a provider.
    """
    reading = bind_concept_names(QuickReading.model_validate(reading.model_dump(mode="json")))
    question_mismatch = question is not None and question != reading.question
    if question_mismatch:
        reading = reading.model_copy(update={"question": question, "ambiguities": (*reading.ambiguities, "question_mismatch")})
    handoff = StudyHandoff(reading=reading, source=source.identity.to_dict())
    reason = "quick_reading_ambiguous" if question_mismatch else qualification_stop(reading)
    if reason:
        return QuickProposal(status="full_study", reason=reason, expand=handoff)
    decision = None
    try:
        require_source(source, provider)
        decision = assess_study(research_capability_profile(source.identity.database), _requirements(reading))
        if decision.status == "refusal":
            raise QuickStop("quick_capability_refused")
        if decision.unassessed_concepts:
            raise QuickStop("quick_capability_unassessed")
        effective = _effective(reading, decision)
        plan = _plan(reading, effective, source)
    except QuickStop as exc:
        fallback = exc.code.endswith(("ambiguous", "unresolved", "full_study"))
        return QuickProposal(status="full_study" if fallback else "stopped", reason=exc.code,
                             expand=handoff, capability=decision)
    handoff = handoff.model_copy(update={"effective_reading": effective})
    card = QuickCard(sha256="", source=source.identity.to_dict(), plan=plan, capability=decision)
    card = card.model_copy(update={"sha256": card_digest(card)})
    return QuickProposal(status="approval_required", reason="quick_qualified", expand=handoff,
                         card=card, capability=decision)
