"""Synthetic population contexts shared across planner and Web tests."""
from __future__ import annotations

import json
from typing import Any

from easyicu.research_agent.schema import (
    CohortDescriptor, ConceptDescriptor, ObservationSemantics, ResearchContext, UserPreferences, VariableRole,
)


_BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}

def _event_time(status: str, *, unit: str = "h") -> ConceptDescriptor:
    return ConceptDescriptor(
        name=f"{status}_time",
        role=VariableRole.OTHER,
        dtype="float64",
        observation_semantics=ObservationSemantics(
            kind="conditional_event_time",
            event_status_column=status,
            representative_column=f"{status}_time",
            time_origin="icu_admission",
            time_unit=unit,
        ),
    )

_VARIABLES = (
    ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
    ConceptDescriptor(
        name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="years"
    ),
    ConceptDescriptor(
        name="los_icu", role=VariableRole.OUTCOME, dtype="float64", unit="days"
    ),
    ConceptDescriptor(
        name="death", role=VariableRole.OUTCOME, dtype="int64", observed_domain=_BINARY
    ),
    _event_time("death"),
    # A status summarized over its own window.
    ConceptDescriptor(
        name="sep3",
        role=VariableRole.OTHER,
        dtype="int64",
        analysis_window="icu_admission[0,24]h",
        observed_domain=_BINARY,
    ),
    # A status summarized over the host's materialization window.
    ConceptDescriptor(
        name="circ", role=VariableRole.OTHER, dtype="int64", observed_domain=_BINARY
    ),
    # A status read by its own recorded time.
    ConceptDescriptor(
        name="vent",
        role=VariableRole.INTERVENTION,
        dtype="int64",
        observed_domain=_BINARY,
    ),
    _event_time("vent"),
    # An outcome status the stay records over its whole length, with no time.
    ConceptDescriptor(
        name="aki", role=VariableRole.OUTCOME, dtype="int64", observed_domain=_BINARY
    ),
    ConceptDescriptor(
        name="lact_max",
        role=VariableRole.LAB,
        dtype="float64",
        source_concept="lact",
        unit="mmol/L",
    ),
    ConceptDescriptor(
        name="map", role=VariableRole.VITAL, dtype="float64", unit="mmHg"
    ),
    ConceptDescriptor(
        name="organ_max",
        role=VariableRole.ORDINAL_SCORE,
        dtype="int64",
        source_concept="organ",
        is_ordinal=True,
        ordinal_levels=[0, 1, 2, 3, 4],
    ),
)

def _ctx(
    *variables: ConceptDescriptor,
    without: tuple[str, ...] = (),
    constraints: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    n_stays: int = 100,
    materialized: float | None = 24.0,
) -> ResearchContext:
    replaced = {variable.name for variable in variables} | set(without)
    data = dict(constraints or {})
    if materialized is not None:
        data["materialization_window"] = {
            "role": "outer_observation_window",
            "anchor": "ICU admission",
            "hours": materialized,
        }
    return ResearchContext(
        research_question="Which stays does the study include?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="miiv",
            n_stays=n_stays,
            id_columns=["stay_id"],
            outcome_columns=["death"],
            provenance=dict(provenance or {}),
        ),
        variables=[*(v for v in _VARIABLES if v.name not in replaced), *variables],
        target_outcome="death",
        user_preferences=UserPreferences(
            data_constraints=json.dumps(data) if data else None
        ),
    )

def _recorded(
    demographic: tuple[tuple[str, dict, int], ...] = (),
    *,
    icd: tuple[list[str], list[str]] | None = None,
    executed: dict[str, Any] | None = None,
    basis: str = "export_contract",
) -> tuple[dict[str, Any], int]:
    """A source selection whose count report chains, and the stays it leaves."""

    remaining = 1000
    steps = []
    for criterion, parameters, excluded in demographic:
        steps.append(
            {
                "criterion": criterion,
                "parameters": parameters,
                "n_before": remaining,
                "n_excluded": excluded,
                "n_remaining": remaining - excluded,
                "n_excluded_missing": 0,
            }
        )
        remaining -= excluded
    report: dict[str, Any] = {
        "count_unit": "icu_stay",
        "source_total": 1000,
        "demographic_steps": steps,
        "selected_before_concept_prefilter": remaining,
        "concept_matches": None,
        "selected_before_icd": remaining,
    }
    if icd is not None:
        report["icd"] = {
            "enabled": True,
            "include_tokens": icd[0],
            "exclude_tokens": icd[1],
        }
        remaining -= 50
    report["selected_before_cap"] = remaining
    report["selected"] = remaining
    record: dict[str, Any] = {
        "basis": basis,
        "host_applied": [],
        "export_report": report,
    }
    if executed is not None:
        record["executed_cohort"] = executed
    return {"source_selection": record}, remaining

def _recorded_ctx(*args, **kwargs) -> ResearchContext:
    constraints, remaining = _recorded(*args, **kwargs)
    return _ctx(constraints=constraints, n_stays=remaining)
