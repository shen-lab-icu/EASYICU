"""A cohort predicate has to separate rows by the values its column takes.

A predicate filters one column the host materialized before planning: its
bare concept, else its ``<concept>_<aggregation>`` summary
(``cohort_eligibility.predicate_context_column``).  When that column's values
form a closed set -- the levels observed in the cohort, else the set its
owner declares (``closed_planning_levels_for``) -- each value is judged with
the comparison the cohort builder applies (``predicate_accepts_closed_level``).
The judgement covers the rows that have a value: a row without one meets no
comparison except ``!=`` and ``not_in``.

An inclusion that no value meets keeps no row that has a value, and an
exclusion that every value meets removes every such row: the cohort is empty,
or holds only rows that lack the column.  An inclusion that every value
meets, or an exclusion that none meets, cannot separate rows by value, so the
criterion it was written for is not applied -- unless the study states a
bound that every value meets.  That second pair is judged only against a
declared set: levels observed in one cohort are not every value a column can
take, and an exclusion that happens to match no one there is not an error.

A planning schema has no rows, so only a declared set can catch a threshold
copied from a concept's description, such as "SOFA >= 2" read off a 0/1
diagnosis flag.  A set describes a predicate only under a summary that keeps
the column's own values (max, min, first, last); ``any`` and ``all``
summarize an event status, so they keep a 0/1 set only.  A mean, a median, a
sum or a count takes other values.  Predicates without a value (``missing``,
``not_missing``) are not judged either, nor is a column without a closed set
(a count, a time, a numeric summary).  An observed set is described by its
size only, never by its values.
"""

from __future__ import annotations

import json
import math
import numbers
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..authority.declared_levels import closed_planning_levels_for, observed_levels_for
from ..cohort.schema import CohortDefinition, ConceptPredicate, TimeWindow, predicate_accepts_closed_level
from ..contracts.exposure_group_rules import (
    EXPOSURE_GROUP_LABELS_KEY,
    EXPOSURE_GROUP_TRANSFORMS,
    UNMEASURED_GROUP_ID,
    ExposureGroupRuleError,
    read_grouping_rules,
)
from ..research_context.materialization_window import context_column_windows, window_anchor
from ..research_context.concept_population import context_data_constraints
from ..research_context.stay_events import ICU_LENGTH_OF_STAY_CONCEPT, column_kind, stay_outcome_columns
from ..schema import AnalysisPlan, ResearchContext
from .cohort_eligibility import predicate_context_column

_VALUE_FREE_OPS = frozenset({"missing", "not_missing"})
_VALUE_KEEPING_AGGREGATIONS = frozenset({"max", "min", "first", "last", "any", "all"})
_EVENT_STATUS_AGGREGATIONS = frozenset({"any", "all"})

Effect = Literal["keeps_no_row", "removes_every_row", "keeps_every_row", "removes_no_row"]

_OUTCOMES: dict[str, str] = {
    "keeps_no_row": "none of them meets it, so it keeps no row that has a value",
    "removes_every_row": "every one of them meets it, so it removes every row that has a value",
    "keeps_every_row": "every one of them meets it, so it keeps every row that has a value",
    "removes_no_row": "none of them meets it, so it removes no row that has a value",
}


@dataclass(frozen=True)
class PredicateOutsideColumnDomain:
    """One cohort predicate whose comparison its column's values cannot honour."""

    side: Literal["inclusion", "exclusion"]
    index: int
    concept_id: str
    aggregation: str
    op: str
    value: Any
    column: str
    effect: Effect
    level_count: int
    #: The owner's declared values; ``None`` when the set was observed.
    declared_levels: tuple[Any, ...] | None

    @property
    def empties_cohort(self) -> bool:
        """No row with a value is left: it keeps none, or removes every one."""

        return self.effect in {"keeps_no_row", "removes_every_row"}

    def message(self) -> str:
        predicate = (
            f"{self.concept_id} ({self.aggregation}) {self.op} {_value_text(self.value)}"
        )
        if self.declared_levels is not None:
            values = (
                f"its owner declares the values "
                f"{json.dumps(list(self.declared_levels), ensure_ascii=False)}"
            )
        else:
            values = f"it holds {self.level_count} observed values"
        return (
            f"The cohort {self.side} {predicate} filters column {self.column!r}: "
            f"{values}, and {_OUTCOMES[self.effect]}"
        )


def cohort_predicates_outside_column_domain(
    context: ResearchContext,
    *,
    inclusion: Iterable[ConceptPredicate],
    exclusion: Iterable[ConceptPredicate],
) -> tuple[PredicateOutsideColumnDomain, ...]:
    """Each predicate that empties the cohort or cannot separate rows by value."""

    variables = {variable.name: variable for variable in context.variables}
    found: list[PredicateOutsideColumnDomain] = []
    for side, predicates in (("inclusion", inclusion), ("exclusion", exclusion)):
        for index, predicate in enumerate(predicates):
            op = str(predicate.op or "").strip().casefold()
            aggregation = str(predicate.aggregation or "").strip().casefold()
            if op in _VALUE_FREE_OPS or aggregation not in _VALUE_KEEPING_AGGREGATIONS:
                continue
            column = predicate_context_column(
                variables, predicate.concept_id, predicate.aggregation
            )
            levels = closed_planning_levels_for(name=column, variables=variables)
            if len(levels) < 2:
                continue
            if aggregation in _EVENT_STATUS_AGGREGATIONS and not _event_status(levels):
                continue
            verdicts = [predicate_accepts_closed_level(predicate, level) for level in levels]
            if any(verdict is None for verdict in verdicts):
                continue
            met = sum(1 for verdict in verdicts if verdict)
            observed = bool(observed_levels_for(name=column, variables=variables))
            effect = _effect(side, met=met, total=len(levels), declared=not observed)
            if effect is None:
                continue
            found.append(
                PredicateOutsideColumnDomain(
                    side=side,
                    index=index,
                    concept_id=str(predicate.concept_id),
                    aggregation=str(predicate.aggregation),
                    op=str(predicate.op),
                    value=predicate.value,
                    column=column,
                    effect=effect,
                    level_count=len(levels),
                    declared_levels=None if observed else tuple(levels),
                )
            )
    return tuple(found)


def _effect(side: str, *, met: int, total: int, declared: bool) -> Effect | None:
    if side == "inclusion":
        if met == 0:
            return "keeps_no_row"
        if met == total and declared:
            return "keeps_every_row"
        return None
    if met == total:
        return "removes_every_row"
    if met == 0 and declared:
        return "removes_no_row"
    return None


def _event_status(levels: Sequence[Any]) -> bool:
    return all(
        isinstance(level, numbers.Real) and level in (0, 1) for level in levels
    )


def _value_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else f"{value}"


@dataclass(frozen=True)
class PredicateEmptiesExposureLevel:
    """A population predicate and the declared level it makes impossible."""

    side: str
    index: int
    predicate: ConceptPredicate
    grouping: str
    level: Any

    def message(self) -> str:
        return (
            f"The cohort {self.side}[{self.index}] "
            f"{json.dumps(self.predicate.to_dict(), ensure_ascii=False, sort_keys=True)} "
            f"empties declared level {self.level!r} of grouping {self.grouping!r} "
            "by its typed rule alone."
        )


# A question that asks what share of the whole population a level is fixes the
# denominator: every declared level must stay in the cohort. Case-neutral
# wording only; "比例风险"/"proportional" (hazards/odds) are not shares.
_POPULATION_SHARE = re.compile(
    r"\b(?:what|which)\s+(?:share|proportion|percentage|fraction|percent)\b(?!al)"
    r"|\b(?:share|proportion|percentage|fraction|prevalence)\s+of\s+(?:all\s+)?(?:the\s+)?"
    r"(?:patients|stays|admissions|population|cohort|icu\s+stays)\b"
    r"|\bhow\s+(?:common|prevalent)\b"
    r"|占[^。？?！!；;，,]{0,8}?(?:比例|比重|百分比|多少)|占比|所占(?:比例|百分比)|构成比",
    re.IGNORECASE,
)


def question_asks_population_share(question: str | None) -> bool:
    """Whether the question asks what share of the whole population a level is."""

    return bool(_POPULATION_SHARE.search(str(question or "")))


def share_resolves_level_conflict(context: ResearchContext, conflict: PredicateEmptiesExposureLevel) -> bool:
    """A share question decides a restriction on a declared grouping's own concept.

    The grouping and the restriction read the same concept; the share needs every
    level in its denominator, so the restriction, not the grouping, yields.
    """

    return population_grouping_resolution(context).absorbs(conflict)


@dataclass(frozen=True)
class PopulationGroupingResolution:
    """Host decision from the study alone, shared by launch, planning and review."""

    grouping_concepts: tuple[tuple[str, str], ...] = ()
    absorbed_fields: tuple[str, ...] = ()
    remaining_fields: tuple[str, ...] = ()
    saved_conflicts: tuple[PredicateEmptiesExposureLevel, ...] = ()

    def absorbs(self, conflict: PredicateEmptiesExposureLevel) -> bool:
        return (conflict.grouping, conflict.predicate.concept_id) in self.grouping_concepts

    def selection_mode(self, required: str | None) -> str | None:
        if required == "predicate_filtered" and self.absorbed_fields and not self.remaining_fields:
            return "all_input_rows"
        return required

    @property
    def note(self) -> str:
        if not self.saved_conflicts:
            return ""
        return (
            "Host-decided population resolution: removed saved population restriction(s) "
            "absorbed by the declared grouping: "
            + "; ".join(dict.fromkeys(
                f"{item.predicate.concept_id} {item.predicate.op} {item.predicate.value:g} "
                f"would empty {item.grouping} level {item.level}"
                for item in self.saved_conflicts
            ))
            + ". The question asks what share of the whole population a level is; "
            "the denominator keeps every declared level, and the grouping retains the "
            "level of interest. "
            + ("The population is all input stays." if not self.remaining_fields else
               "Keep the independent population restrictions on the input stays.")
        )

    def record(self) -> dict[str, Any]:
        if not self.note:
            return {}
        return {
            "owner": "easyicu.planning.cohort_predicate_domain_v1",
            "decision": "host_decided",
            "absorbed_study_fields": list(self.absorbed_fields),
            "remaining_study_fields": list(self.remaining_fields),
            "note": self.note,
        }


def authorized_population_mode(stated_mode: str | None, context: ResearchContext | None) -> str | None:
    """The one judge of which population mode a stated restriction authorizes.

    Planning, launch, candidate materialization and execution all ask this
    function; none re-derives the absorption from the saved study. Without a
    verified context the stated mode stands.
    """
    if context is None:
        return stated_mode
    return population_grouping_resolution(context).selection_mode(stated_mode)


def population_grouping_resolution(context: ResearchContext) -> PopulationGroupingResolution:
    """A share keeps every level; only a proven restriction on its own concept yields.

    Saved typed bounds are tested separately, so a compatible bound on the same
    concept survives. Unknown active cohort settings remain bound. Wording is
    provenance, not an additional executable restriction and is never parsed.
    """

    groupings = tuple(
        (str(item["variable"]), str(item["concept"]))
        for item in context.cohort.provenance.get(EXPOSURE_GROUP_LABELS_KEY, ())
        if isinstance(item, Mapping) and item.get("variable") and item.get("concept")
    ) if question_asks_population_share(context.research_question) else ()
    raw = context_data_constraints(context).get("cohort")
    raw = raw if isinstance(raw, Mapping) else {}
    absorbed, remaining, conflicts = [], [], []
    descriptive = {"label", "review", "exclusion_statement",
                   "max_patients", "sample_size", "sampling", "sampling_seed"}
    for field, value in sorted(raw.items()):
        if field in descriptive or value is None or value is False or value == "" or value == []:
            continue
        if field in {"preset", "mode"} and value == "all_icu":
            continue
        # The minimum-stay owner treats zero as no bound. Explicit age
        # bounds remain restrictions even when compatible with every level.
        if field == "min_icu_los_hours" and value == 0:
            continue
        bound = {"age_min": ("age", ">="), "age_max": ("age", "<=")}.get(field)
        threshold = value
        numeric = isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)
        if groupings and field == "min_icu_los_hours" and numeric and value > 0:
            variable = next((item for item in context.variables if item.name == ICU_LENGTH_OF_STAY_CONCEPT), None)
            unit = str(getattr(variable, "unit", None) or "").strip().casefold()
            # The template reads an unspecified ICU duration in days; explicit
            # hour/day units are the same ones its typed bound can enforce.
            hours_per_unit = {
                **dict.fromkeys(("", "d", "day", "days"), 24.0),
                **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), 1.0),
            }.get(unit) if variable is not None else None
            if hours_per_unit is not None:
                bound = (ICU_LENGTH_OF_STAY_CONCEPT, ">=")
                threshold = value / hours_per_unit
        found = ()
        if groupings and bound and numeric:
            predicate = ConceptPredicate(
                concept_id=bound[0], op=bound[1], value=float(threshold), aggregation="first",
                time_window=TimeWindow(anchor="icu_admission", start_offset_hours=0, end_offset_hours=math.inf),
            )
            found = tuple(item for item in cohort_emptied_exposure_levels(
                context, CohortDefinition(name="saved", inclusion=(predicate,)),
            ) if (item.grouping, item.predicate.concept_id) in groupings)
        if found:
            absorbed.append(field)
            conflicts.extend(found)
        else:
            remaining.append(field)
    return PopulationGroupingResolution(groupings, tuple(absorbed), tuple(remaining), tuple(conflicts))


def exposure_level_conflict_feedback(
    context: ResearchContext, conflicts: Sequence[PredicateEmptiesExposureLevel],
) -> str:
    """Locate a symbolic contradiction without manufacturing observed counts."""

    labels = {
        (item["variable"], level["group"]): level
        for item in context.cohort.provenance.get(EXPOSURE_GROUP_LABELS_KEY, ())
        for level in item.get("levels", ())
    }
    messages = []
    for conflict in conflicts:
        level = labels.get((conflict.grouping, conflict.level), {})
        messages.append(conflict.message() + (
            f" Declared code={level['code']!r}, label={level.get('label')!r}." if level else ""
        ))
    pairs = dict.fromkeys((conflict.predicate.concept_id, conflict.grouping) for conflict in conflicts)
    resolve = "; ".join(
        f"list both {concept!r} and {grouping!r} in the concepts of the estimand or subgroup "
        "question requirement that asks it (coverage \"plan\")"
        for concept, grouping in pairs
    )
    sides = "; ".join(
        f"population restriction on {concept!r} versus the declared grouping {grouping!r}"
        for concept, grouping in pairs
    )
    return " ".join(messages) + (
        " Observed level counts: unavailable in this planning context; this is a "
        f"typed-rule contradiction, not an observed count. Conflicting pair: {sides}. "
        "Resolution options: (a) drop the population restriction and keep the grouping. "
        "This is required when the question needs the whole population, for example what "
        f"share of stays a level is or a comparison between levels: {resolve}; the host then "
        "removes the restriction and records why. (b) keep the restriction and drop the "
        "grouping. The grouping is a stated study choice, so only the user can drop it; the "
        "Planner must not choose (b). Do not restate the restriction in another form, invent "
        "counts, or discard a level."
    )


def population_emptied_exposure_levels(
    context: ResearchContext, plan: AnalysisPlan,
) -> tuple[PredicateEmptiesExposureLevel, ...]:
    """Prove empty levels without rows, question text or observed frequencies.

    Derived groups read their sealed scalar rules, including first-match
    priority. Categorical products declare their literal levels. Only the
    same materialized value, summary and window can establish a contradiction;
    unknown semantics provide no proof. Each reported predicate suffices alone.
    """

    targets: dict[str, list[Any]] = {}
    from ..authority.table_one_binding import table_one_execution_spec

    for step in plan.steps:
        table = table_one_execution_spec(step)
        if table is not None and table.group_by:
            targets.setdefault(table.group_by, []).extend(table.group_levels)
        distribution = step.exposure_outcome_distribution_spec
        if distribution is not None:
            targets.setdefault(distribution.exposure, []).extend(distribution.exposure_levels)
    return cohort_emptied_exposure_levels(context, plan.cohort, targets=targets)


def cohort_emptied_exposure_levels(
    context: ResearchContext, cohort: CohortDefinition | None,
    *, targets: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[PredicateEmptiesExposureLevel, ...]:
    """The same typed proof before executable steps exist, or on a compiled plan."""

    if cohort is None:
        return ()
    targets = targets or {}
    variables = {variable.name: variable for variable in context.variables}
    windows = context_column_windows(context)
    outcomes = stay_outcome_columns(context)
    groupings = context.cohort.provenance.get(EXPOSURE_GROUP_LABELS_KEY) or ()
    found: list[PredicateEmptiesExposureLevel] = []
    for side, predicates in (("inclusion", cohort.inclusion),
                             ("exclusion", cohort.exclusion)):
        for index, predicate in enumerate(predicates):
            column = predicate_context_column(variables, predicate.concept_id, predicate.aggregation)

            def record(grouping: str, level: Any) -> None:
                item = PredicateEmptiesExposureLevel(side, index, predicate, grouping, level)
                if item not in found:
                    found.append(item)

            # Literal levels come from product contracts, not patient rows.
            variable = variables.get(column)
            kind = column_kind(variable, column=column, concept=predicate.concept_id, outcomes=outcomes)
            window = windows.get(column)
            same_value = (
                predicate.aggregation in {"first", "last", "min", "max"}
                if kind in {"admission", "stay_level"}
                else (column.endswith(f"_{predicate.aggregation}") or (
                    getattr(variable, "unit_normalization", None) in EXPOSURE_GROUP_TRANSFORMS
                    and predicate.aggregation in {"first", "last", "min", "max"}
                )) and window is not None
                and window.is_window(predicate.time_window.anchor,
                                     predicate.time_window.start_offset_hours,
                                     predicate.time_window.end_offset_hours)
            )
            if same_value:
                for level in targets.get(column, ()):
                    accepted = predicate_accepts_closed_level(predicate, level)
                    if accepted is not None and accepted == (side == "exclusion"):
                        record(column, level)

            for grouping in groupings:
                if not isinstance(grouping, Mapping) or not isinstance(grouping.get("parameters"), Mapping):
                    continue
                parameters = grouping["parameters"]
                try:
                    rules = read_grouping_rules(parameters)
                except ExposureGroupRuleError:
                    continue  # An unreadable declaration proves no empty level.
                if predicate.concept_id != rules.concept:
                    continue
                summary = "value" if parameters["window"] is None else predicate.aggregation
                if rules.source_columns.get(summary) != column:
                    continue
                if summary == "value":
                    if predicate.aggregation != "first":
                        continue
                else:
                    stated = predicate.time_window
                    if (window_anchor(stated.anchor) != "icu_admission"
                            or stated.start_offset_hours != parameters["window"]["start_hours"]
                            or stated.end_offset_hours != parameters["window"]["end_hours"]):
                        continue
                earlier = []
                for group, rule in rules.groups:
                    # Other summaries are unconstrained: an empty intersection
                    # still proves emptiness without assuming min == first == max.
                    constraints = [*earlier]
                    if rule is not None and rule.summary == summary:
                        constraints.append((rule.op, rule.value))
                    if _scalar_intersection_empty(constraints, predicate, side):
                        record(str(grouping["variable"]), group)
                    if rule is not None and rule.summary == summary:
                        earlier.append((_NEGATED_OP[rule.op], rule.value))
                if rules.unmeasured == "own_group" and set(rules.read_columns) == {column}:
                    accepted = predicate_accepts_closed_level(predicate, float("nan"))
                    if accepted is not None and accepted == (side == "exclusion"):
                        record(str(grouping["variable"]), UNMEASURED_GROUP_ID)
    return tuple(found)


_NEGATED_OP = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}


def _scalar_intersection_empty(
    constraints: Sequence[tuple[str, float]], predicate: ConceptPredicate, side: str,
) -> bool:
    """Exact satisfiability of scalar threshold/equality/set comparisons.

    Truth is constant between adjacent thresholds. Check each endpoint and
    one point in each open interval; rational arithmetic keeps even adjacent
    floating-point endpoints distinct. These are symbolic witnesses, not data.
    """

    values = predicate.value if isinstance(predicate.value, list) else [predicate.value]
    if predicate.op in _VALUE_FREE_OPS:
        values = []
    if any(not isinstance(value, numbers.Real) or not math.isfinite(value) for value in values):
        return False
    cuts = sorted({Fraction(value) for _, value in constraints} | {Fraction(value) for value in values})
    witnesses = ([cuts[0] - 1, *cuts, cuts[-1] + 1,
                  *((left + right) / 2 for left, right in zip(cuts, cuts[1:]))]
                 if cuts else [Fraction(0)])
    for value in witnesses:
        if not all({"<": value < bound, "<=": value <= bound,
                    ">": value > bound, ">=": value >= bound}[op]
                   for op, bound in constraints):
            continue
        accepted = predicate_accepts_closed_level(predicate, value)
        if accepted is None or accepted == (side == "inclusion"):
            return False
    return True


__all__ = [
    "PredicateOutsideColumnDomain",
    "PopulationGroupingResolution",
    "population_grouping_resolution",
    "cohort_predicates_outside_column_domain",
    "cohort_emptied_exposure_levels",
    "population_emptied_exposure_levels",
]
