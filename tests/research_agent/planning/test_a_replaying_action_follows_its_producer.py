"""A host action that replays another action's sealed result follows its producer.

Some host actions run on the output another action of the same executor
sealed: a cluster-number selection replays the cluster solution's sealed
standardized matrix, and a prediction validation reuses the fitted model's
scores.  A product of the same name from another step is not that output, and
a step can only name its action's fixed products, so an outline that selects
such an action without its producer can never become an executable plan.  The
outline check refuses it while the Planner can still choose differently.  The
pairs are read from the runtime contracts; outlines are synthetic.
"""

from __future__ import annotations

import dataclasses

import pytest

from easyicu.research_agent.agents import progressive_prompt_contracts
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    outline_shape_contract,
)
from easyicu.research_agent.planning import outline_action_rules
from easyicu.research_agent.planning.analysis_types import list_analysis_types
from easyicu.research_agent.planning.outline_action_rules import (
    replay_producer_rule_text,
    replayed_producer_action_ids,
    validate_outline_action_rules,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
)
from easyicu.research_agent.planning.scientific_action_catalog import (
    ScientificActionRuntimeContract,
    scientific_actions_for_analysis_type,
)

_CODE = "progressive_outline_replay_producer_absent"
_TRAJECTORY_FIT = "phenotyping.trajectory_feature_clustering"
_CLUSTERS = "phenotyping.cluster_solution"
_K = "phenotyping.k_selection"
_STABILITY = "phenotyping.cluster_stability"
_FITTED = "prediction.discrimination_calibration"
_VALIDATION = "prediction.internal_validation"


def _step(step_id, role, action=None, depends_on=()):
    return {
        "step_id": step_id,
        "planned_analysis_role": role,
        "module_id": "custom_analysis",
        "objective": f"Run the prespecified {step_id} on the sealed cohort.",
        "depends_on": list(depends_on),
        "variable_names": ["stay_key", "marker_a", "marker_b"],
        "scientific_action_id": action,
    }


def _outline(analysis_type, *steps):
    return ProgressivePlanOutline.model_validate(
        {
            "analysis_type": analysis_type,
            "cohort_objective": "Describe the sealed synthetic cohort.",
            "steps": [_step("cohort", "auxiliary"), *steps],
            "rationale": "A prespecified synthetic outline.",
        }
    )


def _refused(outline):
    with pytest.raises(ProgressivePlanCompileError) as refused:
        validate_outline_action_rules(outline, requested_outcomes=[])
    return refused.value


def _pairs():
    found = {}
    for spec in list_analysis_types():
        for action in scientific_actions_for_analysis_type(spec.key).actions:
            producers = replayed_producer_action_ids(spec.key, action.action_id)
            if producers:
                found.setdefault(action.action_id, set()).add(producers)
    return found


def test_each_replay_is_read_from_the_runtime_contracts():
    assert _pairs() == {
        _K: {(_CLUSTERS,)},
        _STABILITY: {(_CLUSTERS,)},
        "prediction.calibration_metrics": {(_FITTED,)},
        "prediction.decision_curve": {(_FITTED,)},
        _VALIDATION: {(_FITTED,)},
    }
    # Another executor's consumer, and one that also accepts other owners'
    # products, replay nothing.
    assert (
        replayed_producer_action_ids(
            "trajectory_clustering", "phenotyping.outcome_by_cluster"
        )
        == ()
    )
    assert replayed_producer_action_ids("survival", "time_to_event.rmst") == ()
    assert replayed_producer_action_ids("not_an_analysis_type", _K) == ()


@pytest.mark.parametrize(
    ("analysis_type", "primary", "replay"),
    [
        ("trajectory_clustering", _TRAJECTORY_FIT, _K),
        ("trajectory_clustering", _TRAJECTORY_FIT, _STABILITY),
        ("prediction_model", None, _VALIDATION),
    ],
)
def test_a_replay_without_its_producer_is_refused_at_the_outline(
    analysis_type, primary, replay
):
    found = _refused(
        _outline(
            analysis_type,
            _step("fit", "primary", primary, ["cohort"]),
            _step("check", "sensitivity", replay, ["fit"]),
        )
    )

    assert found.reason_code == _CODE
    assert (found.step_id, found.step_index, found.path) == ("check", 2, "steps")
    (finding,) = found.details["findings"]
    assert finding["action_id"] == replay
    assert finding["producer_action_ids"] == list(
        replayed_producer_action_ids(analysis_type, replay)
    )
    assert "No earlier step of this outline selects" in str(found)


def test_a_replay_after_its_producer_is_accepted():
    validate_outline_action_rules(
        _outline(
            "trajectory_clustering",
            _step("fit", "primary", _CLUSTERS, ["cohort"]),
            _step("k", "secondary", _K, ["fit"]),
            _step("stability", "sensitivity", _STABILITY, ["fit"]),
        ),
        requested_outcomes=[],
    )
    validate_outline_action_rules(
        _outline(
            "prediction_model",
            _step("model", "primary", _FITTED, ["cohort"]),
            _step("validation", "secondary", _VALIDATION, ["model"]),
            _step(
                "calibration", "secondary", "prediction.calibration_metrics", ["model"]
            ),
            _step("utility", "secondary", "prediction.decision_curve", ["model"]),
        ),
        requested_outcomes=[],
    )


def test_a_trajectory_fit_checks_its_stability_with_its_own_action():
    validate_outline_action_rules(
        _outline(
            "trajectory_clustering",
            _step("fit", "primary", _TRAJECTORY_FIT, ["cohort"]),
            _step(
                "stability",
                "sensitivity",
                "phenotyping.trajectory_cluster_stability",
                ["fit"],
            ),
        ),
        requested_outcomes=[],
    )


def test_the_producer_must_come_before_the_replay():
    found = _refused(
        _outline(
            "trajectory_clustering",
            _step("k", "secondary", _K, ["cohort"]),
            _step("fit", "primary", _CLUSTERS, ["cohort"]),
        )
    )

    assert (found.reason_code, found.step_id) == (_CODE, "k")


def _with_action(monkeypatch, action_id, contract):
    """The trajectory catalog plus one injected host action with ``contract``."""

    real = scientific_actions_for_analysis_type

    def catalog(analysis_type):
        built = real(analysis_type)
        template = next(a for a in built.actions if a.action_id == _CLUSTERS)
        injected = dataclasses.replace(
            template, action_id=action_id, runtime_contract=contract
        )
        return dataclasses.replace(built, actions=(*built.actions, injected))

    monkeypatch.setattr(
        outline_action_rules, "scientific_actions_for_analysis_type", catalog
    )


def _lookalike(monkeypatch, executor):
    """A host action of ``executor`` that also writes the cluster assignments."""

    _with_action(
        monkeypatch,
        "phenotyping.lookalike_assignments",
        ScientificActionRuntimeContract(
            outputs=(("table:phenotype_assignments", "custom"),),
            standard_executor=executor,
        ),
    )


def test_a_same_named_product_of_another_executor_is_not_the_producer(monkeypatch):
    _lookalike(monkeypatch, "signed_fixed_window_suite")
    outline = _outline(
        "trajectory_clustering",
        _step("fit", "primary", "phenotyping.lookalike_assignments", ["cohort"]),
        _step("k", "secondary", _K, ["fit"]),
    )

    found = _refused(outline)

    assert (found.reason_code, found.step_id) == (_CODE, "k")
    assert replayed_producer_action_ids("trajectory_clustering", _K) == (_CLUSTERS,)


def test_an_action_of_the_same_executor_writing_that_product_is_a_producer(
    monkeypatch,
):
    _lookalike(monkeypatch, "cross_sectional_phenotyping")

    assert replayed_producer_action_ids("trajectory_clustering", _K) == (
        _CLUSTERS,
        "phenotyping.lookalike_assignments",
    )
    validate_outline_action_rules(
        _outline(
            "trajectory_clustering",
            _step("fit", "primary", "phenotyping.lookalike_assignments", ["cohort"]),
            _step("k", "secondary", _K, ["fit"]),
        ),
        requested_outcomes=[],
    )


def test_an_action_that_also_accepts_another_input_set_replays_nothing(
    monkeypatch,
):
    _with_action(
        monkeypatch,
        "phenotyping.lookalike_description",
        ScientificActionRuntimeContract(
            outputs=(("table:lookalike_description", "custom"),),
            required_product_inputs=("table:phenotype_assignments",),
            alternative_product_inputs=(("table:cluster_assignments",),),
            standard_executor="cross_sectional_phenotyping",
        ),
    )

    assert (
        replayed_producer_action_ids(
            "trajectory_clustering", "phenotyping.lookalike_description"
        )
        == ()
    )
    validate_outline_action_rules(
        _outline(
            "trajectory_clustering",
            _step("fit", "primary", _TRAJECTORY_FIT, ["cohort"]),
            _step(
                "describe", "secondary", "phenotyping.lookalike_description", ["fit"]
            ),
        ),
        requested_outcomes=[],
    )


def test_the_comparison_rule_still_decides_first():
    found = _refused(
        _outline(
            "trajectory_clustering",
            _step("fit", "primary", _TRAJECTORY_FIT, ["cohort"]),
            _step("k", "secondary", _K, ["fit"]),
            _step("outcomes", "secondary", "phenotyping.outcome_by_cluster", ["fit"]),
        )
    )

    assert found.reason_code == "progressive_outline_trajectory_comparison_unowned"


def test_the_planners_outline_check_runs_the_rule():
    outline = _outline(
        "trajectory_clustering",
        _step("fit", "primary", _TRAJECTORY_FIT, ["cohort"]),
        _step("k", "secondary", _K, ["fit"]),
    )

    with pytest.raises(ProgressivePlanCompileError) as refused:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=["trajectory_clustering"],
            variable_names=["stay_key", "marker_a", "marker_b"],
            allowed_literature_citation_keys=[],
        )

    assert refused.value.reason_code == _CODE


def _prompt(analysis_type):
    return outline_shape_contract(
        analysis_types=[analysis_type],
        module_ids_by_analysis_type={analysis_type: ["custom_analysis"]},
    )


def test_the_outline_prompt_names_each_pair_before_the_planner_chooses():
    trajectory = _prompt("trajectory_clustering")
    prediction = _prompt("prediction_model")

    assert f"{_K}, {_STABILITY} after {_CLUSTERS}" in trajectory
    assert (
        "prediction.calibration_metrics, prediction.decision_curve, "
        f"{_VALIDATION} after {_FITTED}"
    ) in prediction
    # Analysis types that share a family name each pair once.
    shared = replay_producer_rule_text(["prediction_model", "validation"])
    assert shared.count(_VALIDATION) == 1


@pytest.mark.parametrize("analysis_type", [spec.key for spec in list_analysis_types()])
def test_the_sentence_is_the_only_change_to_any_outline_prompt(
    analysis_type, monkeypatch
):
    with_rule = _prompt(analysis_type)
    sentence = replay_producer_rule_text([analysis_type])
    monkeypatch.setattr(
        progressive_prompt_contracts, "replay_producer_rule_text", lambda _types: ""
    )
    without_rule = _prompt(analysis_type)

    has_pair = any(
        replayed_producer_action_ids(analysis_type, action.action_id)
        for action in scientific_actions_for_analysis_type(analysis_type).actions
    )
    assert bool(sentence) == has_pair
    if not sentence:
        assert with_rule == without_rule
    else:
        assert with_rule.count(sentence) == 1
        assert with_rule.replace(sentence, "", 1) == without_rule
