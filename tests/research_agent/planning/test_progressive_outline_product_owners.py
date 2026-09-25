"""An outline knows which step owns a host product.

A step repair cannot change outline-owned coordinates (``depends_on``,
``population_scope``). A product owner the outline gets wrong therefore has
to be bound or rejected while the outline itself can still change.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    _bind_runtime_action_dependencies,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _outline_payload,
)


def test_outline_binds_the_cohort_module_an_action_reads() -> None:
    """An action that reads the analysis cohort depends on the step building it.

    The cohort module's product appears in no action contract, so the edge was
    left to the Planner; an outline that named only the clustering step failed
    compilation three times, and a step repair may not add the edge.
    """

    def outline(*, with_cohort: bool) -> ProgressivePlanOutline:
        return ProgressivePlanOutline(
            analysis_type="trajectory_clustering",
            cohort_objective="Use the host-authorized analysis cohort.",
            steps=[
                *(
                    [
                        ProgressiveOutlineStep(
                            step_id="cohort",
                            planned_analysis_role="auxiliary",
                            module_id="cohort_definition",
                            objective="Build the analysis cohort and its flow.",
                            variable_names=["stay_id", "organ_score"],
                        )
                    ]
                    if with_cohort
                    else []
                ),
                ProgressiveOutlineStep(
                    step_id="trajectory_clusters",
                    planned_analysis_role="primary",
                    module_id="custom_analysis",
                    objective="Cluster aligned organ-score trajectories.",
                    variable_names=["stay_id", "organ_score"],
                    scientific_action_id="phenotyping.trajectory_feature_clustering",
                ),
                ProgressiveOutlineStep(
                    step_id="outcome_by_cluster",
                    planned_analysis_role="secondary",
                    module_id="custom_analysis",
                    objective="Describe outcomes across the frozen clusters.",
                    variable_names=["stay_id", "hospital_death"],
                    scientific_action_id="phenotyping.outcome_by_cluster",
                    depends_on=["trajectory_clusters"],
                ),
            ],
            rationale="Separate scientific choices from host-owned product edges.",
        )

    bound = _bind_runtime_action_dependencies(outline(with_cohort=True))
    assert bound.steps[2].depends_on == ["trajectory_clusters", "cohort"]
    # Without a cohort step there is no owner to bind; validators report it.
    unbound = _bind_runtime_action_dependencies(outline(with_cohort=False))
    assert unbound.steps[1].depends_on == ["trajectory_clusters"]


@pytest.mark.parametrize(
    ("variant", "primary_steps"),
    [
        ({"scope": "primary_model"}, None),
        ({"scope": "analysis_cohort", "primary_module": "custom_analysis"}, None),
        (
            {"scope": "primary_model", "primary_module": "custom_analysis"},
            [{"step_id": "05_primary", "module_id": "custom_analysis", "scientific_action_id": None}],
        ),
        ({"scope": "primary_model", "primary_role": "secondary"}, []),
        (
            {"scope": "primary_model", "before_primary": True},
            [
                {
                    "step_id": "05_primary",
                    "module_id": "adjusted_association",
                    "scientific_action_id": "association.adjusted_association",
                }
            ],
        ),
    ],
)
def test_outline_requires_a_preceding_primary_owner_for_a_primary_model_population(
    variant: dict, primary_steps: list | None
) -> None:
    """A primary-model risk table reuses the rows of the primary association model.

    With a primary that emits no association estimates, step compilation
    rejected the outline-owned scope three times, where no step repair could
    change it; the outline check returns that choice to the Planner.
    """

    payload = _outline_payload()
    steps = payload["steps"]
    primary = next(step for step in steps if step["step_id"] == "05_primary")
    primary["planned_analysis_role"] = variant.get("primary_role", "primary")
    if variant.get("primary_module", "adjusted_association") != "adjusted_association":
        primary["module_id"] = variant["primary_module"]
        primary["scientific_action_id"] = None
    if "primary_module" in variant or "primary_role" in variant:
        # Association sensitivity steps require the association primary.
        steps[:] = [step for step in steps if step["step_id"] != "06_sensitivity"]
    before = bool(variant.get("before_primary"))
    steps.insert(
        steps.index(primary) if before else len(steps),
        {
            "step_id": "08_absolute_risk",
            "planned_analysis_role": "auxiliary",
            "module_id": "absolute_risk_context",
            "objective": "Describe observed outcome risk by exposure level.",
            "depends_on": ["01_cohort"] if before else ["05_primary"],
            "variable_names": ["exposure_flag", "outcome_flag"],
            "literature_citation_keys": [],
            "scientific_action_id": None,
            "population_scope": variant["scope"],
        },
    )

    def validate() -> None:
        ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(payload),
            analysis_types=["association_study"],
            variable_names=[variable.name for variable in _context().variables],
            allowed_literature_citation_keys=[],
        )

    if primary_steps is None:
        validate()
        return
    with pytest.raises(ProgressivePlanCompileError) as caught:
        validate()
    assert caught.value.reason_code == "progressive_outline_primary_population_unsupported"
    assert caught.value.details["path"] == "population_scope"
    assert caught.value.details["step_id"] == "08_absolute_risk"
    assert caught.value.details["findings"] == [
        {
            "required_product": "table:adjusted_association_estimates",
            "preceding_owner_step_ids": (
                ["05_primary"] if variant.get("primary_role") == "secondary" else []
            ),
            "primary_steps": primary_steps,
        }
    ]
    assert "analysis_cohort" in str(caught.value)
