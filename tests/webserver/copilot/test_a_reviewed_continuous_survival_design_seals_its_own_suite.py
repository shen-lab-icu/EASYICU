"""A reviewed continuous-exposure survival proposal ends on the continuous suite.

The proposal's review publishes its coordinates like the binary suite's, and
"apply execution settings" declares the same survival design: the survival
family, the landmark paired with the endpoint's follow-up, the reviewed
roster, and the exposure's window operation.  The next candidate then binds
the continuous suite on its zero-row catalog, and the sealed suite discloses
the coordinates the proposal named, so the replan is the same design.
Synthetic study (the highest total bilirubin in the first 24 hours, 90-day
mortality); zero patient rows.
"""

from __future__ import annotations

import pandas as pd

from easyicu.research_agent.agents.progressive_planner import candidate_analysis_types
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.landmark_continuous_survival_runtime import (
    LandmarkContinuousSurvivalRuntimeAuthority,
)
from easyicu.research_agent.planning.family_spec import (
    LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID,
    family_template_id_for_context,
    sealed_continuous_survival_suite_coordinates,
)
from easyicu.research_agent.planning.family_spec.request import (
    proposed_continuous_survival_suite_coordinates,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.planning.sensitivity_authority import (
    normalize_prespecified_sensitivities,
)
from easyicu.webserver.landmark_survival_runtime_projection import (
    compile_landmark_survival_runtime_projection,
    survival_exposure_onset_column,
    validate_landmark_survival_declaration,
)
from easyicu.webserver.pi_copilot.plan_decisions import compile_agent_plan_configuration
from easyicu.webserver.research_launch_scientific import (
    _metadata_planning_operationalized_columns,
)
from tests.support.continuous_survival import (
    AGE,
    SEX,
    continuous_survival_context,
    continuous_survival_request,
    continuous_survival_spec,
    proposed_continuous_survival_plan,
)

_OWNER = "SURVIVAL_LANDMARK_OWNER_NOT_SEALED"
_TIMING = "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"


def _context():
    # The exposure summarises the first 24 h: it is assessed after ICU admission.
    context = continuous_survival_context()
    return context.model_copy(update={"variables": [
        item.model_copy(update={"analysis_window": "icu_admission[0,24]h"})
        if item.name == "bili_max" else item
        for item in context.variables
    ]})


def _reviewed_facts() -> dict:
    context = _context()
    plan, _llm = proposed_continuous_survival_plan(
        context, continuous_survival_spec(continuous_survival_request(context), [AGE, SEX])
    )
    return dict(build_plan_scientific_review(context=context, plan=plan).facts)


def _study() -> dict:
    return {
        "question": (
            "Is the highest total bilirubin in the first 24 hours associated with "
            "90-day mortality?"
        ),
        "cohort": {"preset": "all_icu"},
        "confirmations": {"feature_time_window": True},
        "sensitivity_specs": [],
    }


def test_the_reviewed_continuous_design_seals_the_continuous_suite(tmp_path) -> None:
    facts = _reviewed_facts()
    assert facts["landmark_survival_suite"]["executable"] is True

    compiled = compile_agent_plan_configuration(
        study=_study(),
        agent_plan={"steps": []},
        runtime_finding_codes=(_OWNER, _TIMING),
        patient_cluster_available=False,
        review_facts=facts,
    )
    patch = compiled.patch
    # The window operation travels with the exposure's concept.
    assert patch["execution_concepts"] == {
        "outcome": "mort_90d",
        "primary_exposure": "bili",
        "primary_exposure_aggregation": "max",
        "covariates": ["age", "sex"],
    }
    assert patch["analysis_design"]["analysis_family"] == "survival"
    [landmark] = patch["sensitivity_specs"]
    assert (landmark["landmark_hours"], landmark["observation_duration_variable"]) == (
        24.0, "followup_days_90d",
    )
    study = {**_study(), **patch}
    assert validate_landmark_survival_declaration(study) == 24.0

    specs = normalize_prespecified_sensitivities(study["sensitivity_specs"])
    columns = _metadata_planning_operationalized_columns(
        primary_exposure_source="bili",
        primary_exposure_aggregation="max",
        covariates=study["covariates"],
        covariate_selection="exact",
        covariate_operationalizations=study["covariate_operationalizations"],
        sensitivity_specs=specs,
        exposure_onset_column=survival_exposure_onset_column(
            study, sensitivity_specs=specs, primary_exposure_source="bili"
        ),
    )
    assert {"bili_max", "followup_days_90d", "age", "sex"} <= set(columns)
    catalog = tmp_path / "planner_catalog.parquet"
    pd.DataFrame(
        {name: pd.Series(dtype="float64") for name in ("stay_id", "mort_90d", *columns)}
    ).to_parquet(catalog, index=False)

    projection = compile_landmark_survival_runtime_projection(
        study=study,
        sensitivity_specs=specs,
        primary_exposure="bili_max",
        primary_exposure_source="bili",
        target_outcome="mort_90d",
        declared_covariates=study["covariates"],
        covariate_operationalizations=study["covariate_operationalizations"],
        target_is_event_status=True,
        universe_path=catalog,
        scientific_configuration_sha256="d" * 64,
    )

    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkContinuousSurvivalRuntimeAuthority)
    assert (authority.exposure_column, authority.exposure_window_summary) == ("bili_max", "max")
    assert authority.exposure_label == "Total Bilirubin, highest in hours 0 to 24"
    assert authority.exposure_unit == "mg/dL"
    assert (authority.event_column, authority.followup_time_column) == ("mort_90d", "followup_days_90d")
    assert authority.time_varying_interval_cutpoints_days == (7.0, 14.0, 28.0)
    assert authority.adjustment_columns == ("age", "sex")

    # The sealed suite discloses the coordinates the proposal named.
    context = _context()
    proposal = proposed_continuous_survival_suite_coordinates(context)
    disclosure = authority.planning_contract_context()
    sealed = sealed_continuous_survival_suite_coordinates(disclosure)
    assert sealed is not None
    keys = (
        "primary_owner", "exposure_column", "exposure_window_summary", "exposure_unit",
        "event_column", "followup_time_column", "landmark_hours", "endpoint_horizon_days",
    )
    assert {key: getattr(sealed, key) for key in keys} == {key: getattr(proposal, key) for key in keys}
    assert sealed.analysis_outputs == proposal.analysis_outputs
    assert sealed.adjustment_columns == ["age", "sex"]
    assert family_template_id_for_context(
        context,
        analysis_types=candidate_analysis_types(context),
        planning_contract_context=disclosure,
    ) == LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID
    replan = continuous_survival_request(context, planning_contract_context=disclosure)
    assert replan.sealed_continuous_suite == sealed and replan.proposed_continuous_suite is None
    assert (replan.adjustment_selection, replan.exact_roster) == ("exact", ["age", "sex"])
