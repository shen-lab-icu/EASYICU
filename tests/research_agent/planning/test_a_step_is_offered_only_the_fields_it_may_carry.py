"""A step is offered only the coordinate-owned fields its outline lets it carry.

The run's action roster lists every action any step may use, but the outline
pins each step to one module, action and role.  A JSON-mode Planner copies the
keys its template shows, so a field the step contract refuses on that step
must not be shown to it, in either transport.
"""

import json

import pytest

from easyicu.research_agent.agents.progressive_payload import (
    progressive_step_materialization_request,
)
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    step_materialization_shape_contract,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.progressive_contract import (
    COORDINATE_OWNED_STEP_FIELDS,
    ProgressiveOutlineStep,
    ProgressiveSkeletonStep,
    coordinate_owned_step_fields,
)

_ROSTER = (
    "phenotyping.cluster_solution",
    "phenotyping.trajectory_feature_clustering",
    "phenotyping.outcome_by_cluster",
    "phenotyping.trajectory_cluster_stability",
)
_VARIABLES = ["stay_key", "marker_a", "marker_b", "marker_c", "event_flag"]
_FEATURES = ["marker_a", "marker_b"]
_COMPARISONS = [{"name": "event_flag", "summary": "count_percent"}]
_FORM = {"target_column": "marker_a", "knot_quantiles": [0.1, 0.5, 0.9]}
_VALUES = {
    "phenotyping_feature_columns": _FEATURES,
    "phenotyping_comparison_variables": _COMPARISONS,
    "functional_form_spec": _FORM,
}
_COORDINATES = {
    "trajectory_primary": ("phenotyping.trajectory_feature_clustering", "primary"),
    "cross_sectional_primary": ("phenotyping.cluster_solution", "primary"),
    "outcome_comparison": ("phenotyping.outcome_by_cluster", "secondary"),
    "stability_sensitivity": ("phenotyping.trajectory_cluster_stability", "sensitivity"),
    "comparison_as_sensitivity": ("phenotyping.outcome_by_cluster", "sensitivity"),
    "cross_sectional_as_secondary": ("phenotyping.cluster_solution", "secondary"),
}


def _outline(action, role):
    return ProgressiveOutlineStep(
        step_id="cluster_fit", module_id="custom_analysis", planned_analysis_role=role,
        objective="Fit the prespecified clusters.", variable_names=_VARIABLES,
        scientific_action_id=action,
    )


def _template(outline):
    text = step_materialization_shape_contract(
        outline_step=outline, outline_step_sha256=canonical_sha256(outline.model_dump(mode="json")),
    )
    head, _, _ = text.partition("\nCopy schema_version")
    return json.loads(head.split("\n", 1)[1])["step"], text


def _schema_branches(outline):
    """Each module-bound shape the strict transport offers for this step."""

    request = progressive_step_materialization_request(
        outline_step=outline, outline_step_sha256=canonical_sha256(outline.model_dump(mode="json")),
        variable_names=outline.variable_names, scientific_action_ids=list(_ROSTER),
    )
    step = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"]
    return [branch["properties"] for branch in step.get("anyOf", [step])]


def _offered(branches, field):
    return any(branch.get(field, {"type": "null"}) != {"type": "null"} for branch in branches)


def test_a_trajectory_fit_is_not_offered_the_cross_sectional_feature_roster():
    outline = _outline("phenotyping.trajectory_feature_clustering", "primary")

    step, text = _template(outline)

    assert "phenotyping_feature_columns" not in step
    assert "set phenotyping_feature_columns" not in text
    assert not _offered(_schema_branches(outline), "phenotyping_feature_columns")


@pytest.mark.parametrize("name", sorted(_COORDINATES))
def test_every_transport_offers_exactly_what_the_contract_accepts(name):
    action, role = _COORDINATES[name]
    outline = _outline(action, role)
    owned = coordinate_owned_step_fields(
        module_id="custom_analysis", scientific_action_id=action, planned_analysis_role=role,
    )

    step, text = _template(outline)
    branches = _schema_branches(outline)

    for field in COORDINATE_OWNED_STEP_FIELDS:
        accepted = _accepts(action, role, field)
        assert (field in owned) is accepted, field
        assert (field in step) is accepted, field
        assert (f"set {field}" in text or f"Set {field}" in text) is accepted, field
        if not accepted:
            assert not _offered(branches, field), field


@pytest.mark.parametrize(("module", "action", "role", "expected"), [
    ("custom_analysis", "phenotyping.cluster_solution", "primary", {"phenotyping_feature_columns"}),
    ("custom_analysis", "phenotyping.cluster_solution", "secondary", set()),
    ("custom_analysis", "phenotyping.trajectory_feature_clustering", "primary", set()),
    ("custom_analysis", "phenotyping.outcome_by_cluster", "secondary", {"phenotyping_comparison_variables"}),
    ("custom_analysis", "phenotyping.outcome_by_cluster", "primary", set()),
    ("custom_analysis", None, "sensitivity", {"functional_form_spec"}),
    ("custom_analysis", None, "primary", set()),
    ("measurement_audit", None, "sensitivity", set()),
])
def test_the_ownership_rule(module, action, role, expected):
    assert coordinate_owned_step_fields(
        module_id=module, scientific_action_id=action, planned_analysis_role=role,
    ) == expected


def test_the_owners_keep_their_fields_in_both_transports():
    for action, role, field in (
        ("phenotyping.cluster_solution", "primary", "phenotyping_feature_columns"),
        ("phenotyping.outcome_by_cluster", "secondary", "phenotyping_comparison_variables"),
    ):
        outline = _outline(action, role)
        step, _ = _template(outline)
        assert step[field] is None
        assert _offered(_schema_branches(outline), field)


def _accepts(action, role, field):
    try:
        ProgressiveSkeletonStep(
            step_id="cluster_fit", planned_analysis_role=role, module_id="custom_analysis",
            objective="Fit the prespecified clusters.", raw_inputs=_VARIABLES,
            outputs=[{"product_id": "table:cluster_fit", "semantic_role": "custom"}],
            scientific_action_id=action, custom_method="prespecified_cluster_fit",
            **{field: _VALUES[field]},
        )
    except ValueError as exc:
        assert "belongs only to" in str(exc), exc
        return False
    return True
