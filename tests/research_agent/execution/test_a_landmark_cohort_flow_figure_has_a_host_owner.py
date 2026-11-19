"""A signed landmark cohort's own flow figure is host-owned before review.

UX leg 3 (2026-11-19): a reviewed raw-Demo plan (first-day lactate groups,
in-hospital death, 24 h landmark) could never prepare its data. The family
template adds a cohort-flow figure over the analysis cohort's
``table:cohort_flow``; the pre-review ownership query accepted that figure
only over the verified host cohort or the trajectory representation, so the
signed fixed-landmark cohort's flow was refused as needing a sandbox
(research_pipeline_plan_step_needs_sandbox), and every repair repeated it.
The landmark cohort executor writes that flow in the figure's sequential
columns; Execute still verifies the produced table. Synthetic data.
"""

from __future__ import annotations

from functools import partial

import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.execution.owner_declaration import (
    HostPlanStepNeedsSandbox,
    _declared_host_figure,
    require_host_rendered_plan,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection
from tests.support.landmark_categorical_fixtures import _dependence, _draft_plan, _frame, _landmark_spec

FIGURE = {
    "step_id": "family_cohort_flow_figure", "planned_analysis_role": "auxiliary",
    "intent": "Render the recorded cohort accounting counts as a participant flow figure.",
    "inputs": ["table:cohort_flow"], "expected_outputs": ["figure:cohort_flow"], "method": "visualization",
    "input_consumption_contracts": [{
        "schema_version": "easyicu.artifact_consumption/1", "input_key": "table:cohort_flow",
        "mode": "all_rows", "role_column": None, "expected_roles": [],
    }],
    "figure_panels": [{
        "schema_version": "easyicu.planned_figure_panel/1", "panel_id": "cohort_accounting",
        "figure_output": "figure:cohort_flow", "article_role": "cohort_accounting", "chart_type": "cohort_flow",
        "placement": "main", "source_products": ["table:cohort_flow"], "presentation": None,
    }],
}


def _authority(tmp_path):
    universe = tmp_path / "universe.parquet"
    _frame().to_parquet(universe, index=False)
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"}, sensitivity_specs=(_landmark_spec(),),
        primary_exposure="aki_stage_strict", primary_exposure_source="aki_stage",
        target_outcome="death", declared_covariates=("age", "sex"), covariate_operationalizations={},
        target_is_event_status=True, universe_path=universe,
        scientific_configuration_sha256="b" * 64, dependence=_dependence(),
    )
    return load_current_case_scientific_runtime_authority(projection.authority)


def _bound(authority, figure=FIGURE):
    plan = _draft_plan().model_dump(mode="json")
    plan["steps"].append(figure)
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    bound, _findings = runtime.bind_plan(AnalysisPlan.model_validate(plan))
    return bound


def _require(plan, authority):
    require_host_rendered_plan(
        plan=plan, run_dir=None, trajectory_scientific_runtime_authority=None,
        current_case_scientific_runtime_authority=authority, scientific_runtime_projection_sha256="c" * 64,
    )


def test_the_landmark_cohorts_flow_figure_needs_no_sandbox(tmp_path):
    authority = _authority(tmp_path)
    _require(_bound(authority), authority)


@pytest.mark.parametrize("change", ["presentation", "second_input"])
def test_any_other_figure_over_that_flow_is_still_refused(tmp_path, change):
    # Each variant is a schema-valid plan, so the refusal is the ownership query's.
    authority = _authority(tmp_path)
    figure = {**FIGURE, "figure_panels": [dict(panel) for panel in FIGURE["figure_panels"]]}
    if change == "presentation":
        figure["figure_panels"][0]["presentation"] = {"layout": "column"}
    else:
        figure["inputs"] = ["table:cohort_flow", "artifact:analysis_cohort"]
    with pytest.raises(HostPlanStepNeedsSandbox) as refused:
        _require(_bound(authority, figure), authority)
    assert refused.value.easyicu_safe_diagnostic["step_id"] == "family_cohort_flow_figure"


def test_without_the_signed_cohort_authority_the_flow_figure_has_no_owner(tmp_path):
    authority = _authority(tmp_path)
    plan = _bound(authority)
    figure = next(step for step in plan.steps if step.step_id == FIGURE["step_id"])
    declared = partial(_declared_host_figure, figure, plan=plan, host_cohort_step=None,
                       trajectory_scientific_runtime_authority=None)
    assert declared(current_case_scientific_runtime_authority=authority)
    assert not declared(current_case_scientific_runtime_authority=None)
