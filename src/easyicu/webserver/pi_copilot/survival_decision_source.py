"""Metadata-only input binding for a survival decision before a plan exists.

The host's versioned research context supplies resolved coordinates. No model
plan, prose-derived coordinate, cohort table or patient record is read here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from easyicu.outcome_availability import fixed_horizon_mortality_endpoint
from easyicu.research_agent.concept_availability import normalize_database_name
from easyicu.research_agent.research_context.concept_population import context_data_constraints
from easyicu.research_agent.research_context.materialization_window import host_materialization_window_hours
from easyicu.research_agent.research_context.typed import parse_research_context_json
from easyicu.research_agent.schema import ResearchContext
from easyicu.webserver.study_scientific_configuration import ScientificConfiguration


@dataclass(frozen=True)
class SurvivalDecisionSource:
    run_id: str
    context_sha256: str
    context: ResearchContext


def read_survival_decision_source(wrapper: Path | None) -> SurvivalDecisionSource | None:
    """Read the sole host run's context; ambiguous or escaping inputs fail closed."""
    if wrapper is None:
        return None
    try:
        root = wrapper.resolve() / "pipeline"
        if root.is_symlink():
            return None
        # The pipeline keeps its lock directory beside the run; it is not a run.
        runs = [entry for entry in root.iterdir() if not entry.name.startswith(".")]
        if len(runs) != 1:
            return None
        run = runs[0]
        path = run / "research_context.json"
        if run.is_symlink() or not run.is_dir() or path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
            return None
        raw = path.read_bytes()
        return SurvivalDecisionSource(
            run_id=run.name, context_sha256=hashlib.sha256(raw).hexdigest(),
            context=parse_research_context_json(raw),
        )
    except (OSError, ValueError, TypeError):
        return None


def study_with_run_coordinates(
    study: Mapping[str, Any], source: SurvivalDecisionSource,
) -> dict[str, Any] | None:
    """Bind saved intent to resolved columns, retaining a saved exact roster."""
    context = source.context
    database = (study.get("data_source") or {}).get("database")
    if normalize_database_name(context.cohort.database) != normalize_database_name(str(database or "")):
        return None
    endpoint = fixed_horizon_mortality_endpoint(context.target_outcome or "")
    exposure = context.primary_exposure
    hours = host_materialization_window_hours(context)
    window = context_data_constraints(context).get("materialization_window")
    if isinstance(window, Mapping) and window.get("start_hours", 0) != 0:
        return None
    preferences = context.user_preferences
    if endpoint is None or not exposure or hours is None or (preferences is not None and (
        preferences.landmark_hours is not None
        or any(spec.axis == "timing" or spec.strategy == "restricted_mean_survival" for spec in preferences.sensitivity_specs)
    )):
        return None
    configuration = ScientificConfiguration.inspect(study)
    saved = configuration.covariate_selection() == "exact"
    if saved:
        # An already structured study must agree with its host-resolved run.
        if (
            configuration.executable_primary_exposure() != exposure
            or configuration.executable_target_outcome() != endpoint.event_concept
            or configuration.materialization_window()[1] != hours
        ):
            return None
        covariates = configuration.covariates()
    else:
        statements = study.get("stated_adjustments")
        if not isinstance(statements, (list, tuple)) or not statements:
            return None
        covariates = tuple(row.get("concept") if isinstance(row, Mapping) else None for row in statements)
        if any(not isinstance(name, str) or not name for name in covariates) or len(set(covariates)) != len(covariates):
            return None
    available = {variable.name for variable in context.variables}
    if not {exposure, endpoint.event_concept, endpoint.followup_concept, *covariates} <= available:
        return None
    if saved:
        return dict(study)
    return {
        **study,
        "execution_concepts": {
            **dict(study.get("execution_concepts") or {}),
            "primary_exposure": exposure, "outcome": endpoint.event_concept, "covariates": list(covariates),
        },
        "time_window": {"anchor": "icu_admission", "hours": hours},
        "covariates": list(covariates), "covariate_selection": "exact",
        "covariate_authority": "researcher_stated",
        "covariate_rationales": {name: "Stated in the research question." for name in covariates},
        "covariate_operationalizations": {name: name for name in covariates},
    }
