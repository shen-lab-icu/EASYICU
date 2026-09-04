"""Shared bounded StudyContext fixtures for Copilot workflow contracts."""

from __future__ import annotations

import hashlib
from typing import Any

from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot import cohort_eligibility

def confirmed_cohort_decision(
    option_id: str,
    *,
    study_context_id: str,
    study_context_revision: int,
    current_cohort: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = dict(current_cohort or {})
    target = cohort_eligibility.selection_cohort_for_option(
        {"cohort": base}, option_id
    )
    scope = study_context_owner.normalize_primary_cohort_scope(
        {"cohort": target}
    )
    session_id = f"pi-{study_context_id}"
    seed = (
        f"{session_id}:{study_context_id}:{study_context_revision - 1}:"
        f"{option_id}:{scope.sha256}"
    )
    event = cohort_eligibility.build_selection_event(
        option_id=option_id,
        study_context_id=study_context_id,
        expected_revision=study_context_revision - 1,
        session_id=session_id,
        user_turn_id=f"turn-{session_id}",
        event_id=hashlib.sha256(f"event:{seed}".encode()).hexdigest(),
        one_use_grant_id=hashlib.sha256(f"grant:{seed}".encode()).hexdigest(),
        primary_cohort_contract_sha256=scope.sha256,
        actor_id_sha256=hashlib.sha256(f"actor:{session_id}".encode()).hexdigest(),
        selected_at="2026-08-29T12:00:00Z",
    )
    authority = cohort_eligibility.confirmation_authority_for_option(
        option_id,
        study_context_id=study_context_id,
        study_context_revision=study_context_revision,
        current_cohort=base,
        selection_event=event,
        confirmed_at="2026-08-29T12:00:00Z",
    )
    return target, authority


def complete_study() -> dict[str, Any]:
    cohort, authority = confirmed_cohort_decision(
        "no_eligibility_filter",
        study_context_id="study-workflow",
        study_context_revision=4,
        current_cohort={"max_patients": 2000},
    )
    return {
        "id": "study-workflow",
        "revision": 4,
        "question": "Does an aggregate ICU feature predict mortality?",
        "data_source": {
            "path": "/private/prepared/source",
            "database": "mimiciv",
        },
        "cohort": cohort,
        "cohort_eligibility_authority": authority,
        "modules": ["vitals", "outcome"],
        "outcome": "In-hospital mortality",
        "primary_exposure": "heart_rate",
        "covariates": ["age", "sex"],
        "covariate_selection": "exact",
        "covariate_rationales": {
            "age": "Age is a baseline demographic confounder selected before analysis.",
            "sex": "Sex is a baseline demographic confounder selected before analysis.",
        },
        "covariate_temporal_roles": {
            "age": "baseline_static",
            "sex": "baseline_static",
        },
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "heart_rate",
            "covariates": ["age", "sex"],
        },
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "confirmations": {
            "feature_time_window": True,
            "export_format": True,
            "extraction_completed": True,
        },
        "export_format": "parquet",
        "analysis_goal": "Descriptive prognostic association",
    }
