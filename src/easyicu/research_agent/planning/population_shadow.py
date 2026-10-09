"""The population a Planner's spec compiles to, set beside the plan's cohort.

Owner
-----
This module owns the shadow audit of step 2a of the population spec design:
the Planner states the population as a typed spec beside the cohort it
already writes, the host compiles the spec
(:func:`.population_compile.compile_population`) and compares the compiled
cohort with the plan's own, predicate by predicate.  The plan's cohort is
still the one the run applies; the audit changes nothing and decides
nothing.  It is written as ``population_shadow_audit.json`` so each
difference can be explained before the plan's cohort is ever compiled from
the spec.

Two predicates that differ only where the difference cannot change a row are
reported as equivalent, not as a difference:

* a window on a column that holds one value per stay (the context records no
  window it was summarized over, ``context_column_windows``), where the
  builder reads the same column and value whatever window a predicate states;
* two cohorts with no predicate on either side, whatever selection mode each
  states: both keep every input row.

A criterion over a concept of the study's design (its exposure or its
outcome) is listed apart: a restriction the exposure itself defines belongs
to the design, not to the population.  ``would_block`` marks a spec with an
inclusion the host could not apply: once the plan's cohort is compiled from
the spec (step 2b), such a plan stops before approval, so it is a difference
to explain even when the two cohorts select the same rows.

The spec is read here, not by the plan: the cohort intent keeps it as the
Planner wrote it.  When the owner refuses the spec as a whole, each criterion
is read on its own; the ones it accepts are compiled (``spec_partly_invalid``)
and the others are recorded with the owner's errors.  A spec that is not an
object, or of which no criterion can be read, is ``spec_invalid``.  The time
zero is the one the plan review reads (``scientific_review.plan_time_zero_hours``),
without the signed runtime authority, which planning does not hold.
Compiling and comparing read no patient row and call no model; the writer
never raises, because an audit that fails must not stop planning.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import TypeAdapter, ValidationError

from ..schema import AnalysisPlan, ResearchContext
from .cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    cohort_concept_id_scope,
    sealed_cohort_concept_ids,
)
from .cohort_eligibility import POPULATION_TIME_ZERO_ANCHORS, predicate_context_column
from .population_compile import compile_population
from .population_spec import PopulationCriterion, PopulationSpec
from .scientific_review import plan_time_zero_hours, trajectory_representation_facts
from ..research_context.materialization_window import context_column_windows

__all__ = [
    "POPULATION_SHADOW_AUDIT_FILENAME",
    "POPULATION_SHADOW_AUDIT_SCHEMA_VERSION",
    "criterion_concepts",
    "population_shadow_audit",
    "write_population_shadow_audit",
]

POPULATION_SHADOW_AUDIT_SCHEMA_VERSION = "easyicu.population_shadow_audit/1"
POPULATION_SHADOW_AUDIT_FILENAME = "population_shadow_audit.json"

_SIDES = ("inclusion", "exclusion")
#: The owner's errors kept for a refused spec, without the input it quoted.
_MAX_SPEC_ERRORS = 20
_CRITERION: TypeAdapter[PopulationCriterion] = TypeAdapter(PopulationCriterion)


def criterion_concepts(criterion: PopulationCriterion) -> tuple[str, ...]:
    """The study concepts a criterion names, in the order it names them."""

    named = getattr(criterion, "concepts_all_of", None)
    if named:
        return tuple(str(concept) for concept in named)
    concept = getattr(criterion, "concept", None)
    return (str(concept),) if concept else ()


def population_shadow_audit(
    *,
    spec: Optional[PopulationSpec | Mapping[str, Any]],
    context: ResearchContext,
    plan_cohort: Optional[Mapping[str, Any]],
    time_zero_hours: Optional[float],
    design_concepts: Sequence[str] = (),
) -> dict[str, Any]:
    """Compile ``spec`` on ``context`` and compare it with the plan's cohort."""

    stated = dict(plan_cohort or {})
    audit: dict[str, Any] = {
        "schema_version": POPULATION_SHADOW_AUDIT_SCHEMA_VERSION,
        "time_zero_hours": time_zero_hours,
        "design_concepts": sorted(
            {str(c) for c in design_concepts if str(c or "").strip()}
        ),
        # Whether the plan needed a spec: a caller-bound cohort of every row does
        # not.  A cohort omits the default mode; it is read as the owner reads it.
        "plan_selection_mode": stated.get("selection_mode")
        or ("predicate_filtered" if plan_cohort is not None else None),
        "plan_predicate_count": sum(len(stated.get(side) or ()) for side in _SIDES),
    }
    if spec is None:
        return {**audit, "status": "no_spec"}
    raw = spec
    status = "compiled"
    errors: list[dict[str, str]] = []
    if not isinstance(spec, PopulationSpec):
        spec, errors = _read_spec(raw)
        if spec is None:
            return {**audit, "status": "spec_invalid", "spec": raw, "errors": errors}
        status = "spec_partly_invalid" if errors else "compiled"
    compiled = compile_population(spec, context, time_zero_hours=time_zero_hours)
    # The plan's predicates name this run's sealed concepts, as the compiled ones do.
    with cohort_concept_id_scope(sealed_cohort_concept_ids(context)):
        plan = CohortDefinition.from_dict(dict(plan_cohort or {}))
        comparison = _compare(compiled.cohort_definition(), plan, context)
    design = set(audit["design_concepts"])
    blocking = [item.criterion.id for item in compiled.blocking]
    return {
        **audit,
        "status": status,
        "spec": spec.model_dump(mode="json") if not errors else raw,
        **({"errors": errors} if errors else {}),
        "compiled_sha256": compiled.sha256(),
        "criteria": [_criterion_row(item, design) for item in compiled.criteria],
        "blocking": blocking,
        "would_block": bool(blocking),
        "on_design_concepts": [
            item.criterion.id
            for item in compiled.criteria
            if design.intersection(criterion_concepts(item.criterion))
        ],
        "comparison": comparison,
        "differs": _differs(comparison),
    }


def write_population_shadow_audit(
    run_dir: Path, *, context: ResearchContext, plan: AnalysisPlan, cohort: Any
) -> Optional[Path]:
    """Write the audit of ``plan`` beside it; never raise.

    ``cohort`` is the cohort intent the plan was compiled from; its
    ``population_spec`` is the spec audited.  The plan's cohort is the one
    the run applies, so an audit that fails is recorded as ``audit_failed``,
    an audit that cannot be written is skipped (``None``), and planning goes
    on either way.
    """

    path = Path(run_dir) / POPULATION_SHADOW_AUDIT_FILENAME
    try:
        audit = population_shadow_audit(
            spec=getattr(cohort, "population_spec", None),
            context=context,
            # plan_dict keeps the criteria the plan states but does not apply.
            plan_cohort=plan.cohort.plan_dict() if plan.cohort is not None else None,
            time_zero_hours=plan_time_zero_hours(
                context, trajectory_representation_facts(context, plan), None
            ),
            design_concepts=(
                context.primary_exposure or "",
                context.target_outcome or "",
            ),
        )
    except Exception as exc:  # noqa: BLE001 - the shadow audit never stops planning
        audit = {
            "schema_version": POPULATION_SHADOW_AUDIT_SCHEMA_VERSION,
            "status": "audit_failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    temporary = None
    try:
        raw = json.dumps(
            audit, ensure_ascii=False, indent=1, sort_keys=True, default=str
        )
        handle, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=".shadow-", suffix=".json"
        )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(raw)
        os.replace(temporary, path)
    except Exception:  # noqa: BLE001 - nor does a file it cannot write
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        return None
    return path


def _errors(exc: ValidationError, prefix: str = "") -> list[dict[str, str]]:
    """The owner's errors, located, without the input they quote."""

    head = [prefix] if prefix else []
    return [
        {
            "loc": ".".join([*head, *(str(part) for part in error["loc"])]),
            "type": error["type"],
            "msg": error["msg"],
        }
        for error in exc.errors(include_url=False, include_input=False)
    ]


def _read_spec(raw: Any) -> tuple[Optional[PopulationSpec], list[dict[str, str]]]:
    """The spec the owner reads from ``raw``, and the errors of what it refused.

    The whole spec first; when the owner refuses it, each criterion on its
    own, so one malformed criterion does not hide the others' dispositions.
    """

    try:
        return PopulationSpec.model_validate(raw), []
    except ValidationError as exc:
        whole = _errors(exc)[:_MAX_SPEC_ERRORS]
    criteria = raw.get("criteria") if isinstance(raw, Mapping) else None
    if not isinstance(criteria, list):
        return None, whole
    kept, refused = [], []
    for index, item in enumerate(criteria):
        try:
            kept.append(_CRITERION.validate_python(item))
        except ValidationError as exc:
            refused.extend(_errors(exc, prefix=f"criteria.{index}"))
    try:
        partial = PopulationSpec(criteria=kept) if kept else None
    except ValidationError:
        partial = None
    if partial is None:
        return None, whole
    return partial, (refused or whole)[:_MAX_SPEC_ERRORS]


# -- comparison ----------------------------------------------------------------


def _criterion_row(item: Any, design: set[str]) -> dict[str, Any]:
    concepts = criterion_concepts(item.criterion)
    return {
        "id": item.criterion.id,
        "kind": item.criterion.kind,
        "role": item.criterion.role,
        "source": item.criterion.source,
        "quote": item.criterion.quote,
        "concepts": list(concepts),
        "disposition": item.disposition,
        "reason": item.reason,
        "side": item.side,
        "predicates": [predicate.to_dict() for predicate in item.predicates],
        "on_design_concept": bool(design.intersection(concepts)),
    }


def _number(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return float(value)


def _canonical(predicate: ConceptPredicate) -> dict[str, Any]:
    """The predicate as a row filter reads it: 1 and 1.0 alike, one admission anchor.

    The anchors that name ICU admission are the eligibility owner's
    (``cohort_eligibility.POPULATION_TIME_ZERO_ANCHORS``).
    """

    data = predicate.to_dict()
    window = dict(data["time_window"])
    if str(window.get("anchor") or "").casefold() in POPULATION_TIME_ZERO_ANCHORS:
        window["anchor"] = "icu_admission"
    value = data.get("value")
    return {
        **data,
        "time_window": {key: _number(item) for key, item in window.items()},
        "value": [_number(item) for item in value]
        if isinstance(value, list)
        else _number(value),
    }


def _key(predicate: ConceptPredicate) -> str:
    return json.dumps(_canonical(predicate), sort_keys=True, default=str)


def _compare(
    compiled: CohortDefinition, plan: CohortDefinition, context: ResearchContext
) -> dict[str, Any]:
    variables = {str(variable.name): variable for variable in context.variables}
    windows = context_column_windows(context)
    result: dict[str, Any] = {
        "compiled_selection_mode": compiled.selection_mode,
        "plan_selection_mode": plan.selection_mode,
        "selection_mode": _selection_mode(compiled, plan),
        # The two lists cannot be matched by their words (a spec's quote, a
        # plan's criterion text), so both are kept with their counts.
        "compiled_unapplied": list(compiled.unapplied_population_criteria),
        "plan_unapplied": list(plan.unapplied_population_criteria),
        "unapplied_counts": {
            "compiled": len(compiled.unapplied_population_criteria),
            "plan": len(plan.unapplied_population_criteria),
        },
    }
    for side in _SIDES:
        mine = {_key(p): p for p in getattr(compiled, side)}
        theirs = {_key(p): p for p in getattr(plan, side)}
        compiled_only = [mine[k] for k in sorted(set(mine) - set(theirs))]
        plan_only = [theirs[k] for k in sorted(set(theirs) - set(mine))]
        equivalent = []
        for predicate in list(compiled_only):
            match = next(
                (
                    other
                    for other in plan_only
                    if _same_row_selection(predicate, other, variables, windows)
                ),
                None,
            )
            if match is not None:
                equivalent.append([predicate.to_dict(), match.to_dict()])
                compiled_only.remove(predicate)
                plan_only.remove(match)
        result[side] = {
            "both": [mine[k].to_dict() for k in sorted(set(mine) & set(theirs))],
            "equivalent": equivalent,
            "compiled_only": [p.to_dict() for p in compiled_only],
            "plan_only": [p.to_dict() for p in plan_only],
        }
    return result


def _same_row_selection(
    one: ConceptPredicate,
    other: ConceptPredicate,
    variables: Mapping[str, Any],
    windows: Mapping[str, Any],
) -> bool:
    """Whether two predicates differing only in window select the same rows."""

    mine, theirs = _canonical(one), _canonical(other)
    if any(mine[field] != theirs[field] for field in ("aggregation", "op", "value")):
        return False
    column = predicate_context_column(variables, one.concept_id, one.aggregation)
    return (
        column
        == predicate_context_column(variables, other.concept_id, other.aggregation)
        and column in variables
        and column not in windows
    )


def _selection_mode(compiled: CohortDefinition, plan: CohortDefinition) -> str:
    if compiled.selection_mode == plan.selection_mode:
        return "same"
    empty = not (
        plan.inclusion or plan.exclusion or compiled.inclusion or compiled.exclusion
    )
    return "equivalent" if empty else "differs"


def _differs(comparison: Mapping[str, Any]) -> bool:
    return comparison["selection_mode"] == "differs" or any(
        comparison[side]["compiled_only"] or comparison[side]["plan_only"]
        for side in _SIDES
    )
