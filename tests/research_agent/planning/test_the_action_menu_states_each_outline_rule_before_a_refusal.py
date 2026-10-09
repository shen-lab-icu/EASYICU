"""The action menu states each outline-stage rule the outline check enforces.

A Planner chose actions from rows that did not say where an outline may
select them, and met each rule only as a refusal: a phenotype comparison in
an outline with a model-coded trajectory primary, a design selection that
listed no question anchor, then a cross-sectional stability replay after the
trajectory fit.  Each row now carries the place its rule owner checks
(``outline_position``), and the outline prompt states the anchor rule with
the anchors the refusal names.  Rows, outlines and contexts are synthetic.
"""

from __future__ import annotations

import json
import re

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    select_progressive_variables,
)
from easyicu.research_agent.planning.analysis_types import list_analysis_types
from easyicu.research_agent.planning.outline_action_menu import outline_action_catalog
from easyicu.research_agent.planning.outline_action_rules import (
    outline_action_position,
    replayed_producer_action_ids,
    validate_outline_action_rules,
)
from easyicu.research_agent.planning.outline_design_selection import (
    listable_question_anchors,
    question_anchor_rule_text,
    validate_outline_design_selection,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
)
from easyicu.research_agent.planning.scientific_action_catalog import (
    ScientificActionGapError,
    scientific_action_for_id,
    scientific_actions_for_analysis_type,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _outline_payload,
)

_OUTCOMES = ("outcome_flag",)
_ACTION_ID = re.compile(r"\b[a-z_]+\.[a-z_]+\b")
_ANCHOR_CODE = "progressive_design_selection_question_anchor_missing"


def _analysis_types() -> list[str]:
    return [spec.key for spec in list_analysis_types()]


def _available(analysis_type: str):
    return [
        action
        for action in scientific_actions_for_analysis_type(analysis_type).actions
        if action.execution_mode != "not_available"
    ]


def _step(step_id, action_id, *, role="secondary", depends_on=()):
    return ProgressiveOutlineStep(
        step_id=step_id,
        module_id="custom_analysis",
        planned_analysis_role=role,
        objective="Run the selected scientific action.",
        depends_on=list(depends_on),
        variable_names=["exposure_flag", *_OUTCOMES],
        scientific_action_id=action_id,
    )


def _outline(analysis_type, *steps):
    return ProgressivePlanOutline.model_construct(
        analysis_type=analysis_type, steps=list(steps)
    )


def _refusal(outline, *, requested_outcomes=()):
    """The outline check's refusal, or None.

    Without requested outcomes a cluster primary needs no comparison step, so
    one supporting step after one primary meets every other rule alone.
    """

    try:
        validate_outline_action_rules(outline, requested_outcomes=requested_outcomes)
    except ProgressivePlanCompileError as refused:
        return refused
    return None


def _after(analysis_type, primary_id, action_id):
    """One primary, then one secondary step that depends on it."""

    return _outline(
        analysis_type,
        _step("primary_step", primary_id, role="primary"),
        _step("supporting_step", action_id, depends_on=("primary_step",)),
    )


def test_each_row_carries_the_position_its_rule_owner_states():
    for analysis_type in _analysis_types():
        _ids, rows = outline_action_catalog((analysis_type,))
        for row in rows:
            stated = outline_action_position(analysis_type, row["action_id"])
            assert row.get("outline_position") == (stated or None)


def test_a_replaying_action_is_refused_exactly_where_its_row_says():
    checked = []
    for analysis_type in _analysis_types():
        _ids, rows = outline_action_catalog((analysis_type,))
        for row in rows:
            producers = replayed_producer_action_ids(analysis_type, row["action_id"])
            if not producers:
                continue
            checked.append(row["action_id"])
            assert set(_ACTION_ID.findall(row["outline_position"])) == set(producers)
            alone = _refusal(
                _outline(analysis_type, _step("supporting_step", row["action_id"]))
            )
            assert alone is not None
            assert alone.reason_code == "progressive_outline_replay_producer_absent"
            assert alone.details["findings"][0]["producer_action_ids"] == list(
                producers
            )
            for producer in producers:
                assert (
                    _refusal(_after(analysis_type, producer, row["action_id"])) is None
                )
    assert checked, "no family registers a replaying action"


def test_a_primary_row_names_exactly_the_actions_its_outline_refuses_after_it():
    named_somewhere = False
    for analysis_type in _analysis_types():
        actions = _available(analysis_type)
        for primary in (action for action in actions if action.tier == "primary"):
            named = set(
                _ACTION_ID.findall(
                    outline_action_position(analysis_type, primary.action_id)
                )
            )
            named_somewhere = named_somewhere or bool(named)
            refused = {
                action.action_id
                for action in actions
                if action.action_id != primary.action_id
                and _refusal(_after(analysis_type, primary.action_id, action.action_id))
                is not None
            }
            assert named == refused, (analysis_type, primary.action_id)
    assert named_somewhere, "no primary row names an action it does not admit"


def test_the_comparison_row_states_both_halves_of_its_rule():
    analysis_type = "trajectory_clustering"
    comparison = "phenotyping.outcome_by_cluster"
    stated = outline_action_position(analysis_type, comparison)
    assert "phenotyping.cluster_solution" in stated
    assert (
        "never in an outline with a phenotyping.trajectory_feature_clustering primary"
        in stated
    )

    detached = _outline(
        analysis_type,
        _step("clusters", "phenotyping.cluster_solution", role="primary"),
        _step("comparison", comparison),
    )
    assert _refusal(detached, requested_outcomes=_OUTCOMES).reason_code == (
        "progressive_outline_phenotype_comparison_incomplete"
    )
    attached = _after(analysis_type, "phenotyping.cluster_solution", comparison)
    assert _refusal(attached, requested_outcomes=_OUTCOMES) is None
    unnamed = _refusal(attached, requested_outcomes=(*_OUTCOMES, "another_outcome"))
    assert unnamed.reason_code == "progressive_outline_phenotype_comparison_incomplete"
    with_trajectory = _after(
        analysis_type, "phenotyping.trajectory_feature_clustering", comparison
    )
    assert _refusal(with_trajectory).reason_code == (
        "progressive_outline_trajectory_comparison_unowned"
    )


def test_an_action_no_outline_rule_places_carries_no_position():
    _ids, rows = outline_action_catalog(("trajectory_clustering",))
    unplaced = {row["action_id"] for row in rows if "outline_position" not in row}
    assert {
        "phenotyping.cluster_solution",
        "phenotyping.trajectory_cluster_stability",
    } <= unplaced


def test_the_menu_offers_exactly_the_actions_the_compiler_accepts():
    # Six families list an action the catalog marks not available; the
    # compiler refuses each, so a row for one would only lead to a refusal.
    unavailable = 0
    for analysis_type in _analysis_types():
        ids, rows = outline_action_catalog((analysis_type,))
        offered = {row["action_id"] for row in rows}
        assert offered == set(ids)
        assert offered == {action.action_id for action in _available(analysis_type)}
        for action in scientific_actions_for_analysis_type(analysis_type).actions:
            if action.execution_mode != "not_available":
                continue
            unavailable += 1
            assert action.action_id not in offered
            with pytest.raises(ScientificActionGapError):
                scientific_action_for_id(
                    analysis_type=analysis_type, action_id=action.action_id
                )
    assert unavailable


@pytest.mark.parametrize(
    "analysis_type", ["trajectory_clustering", "dynamic_prediction"]
)
def test_the_outline_prompt_shows_each_row_with_its_position(analysis_type):
    context = _context()
    variables = select_progressive_variables(context)
    _ids, rows = outline_action_catalog((analysis_type,))
    prompt = ProgressivePlannerAgent._user_prompt(
        context, analysis_types=(analysis_type,), variables=variables, action_rows=rows
    )
    block = prompt.split(
        "Retrieved scientific actions (only these may be selected):\n", 1
    )[1]
    shown = json.loads(block.split("\n", 1)[0])
    assert shown == json.loads(json.dumps(rows))
    assert any("outline_position" in row for row in shown)


@pytest.mark.parametrize(
    ("anchors", "listable"),
    [
        (("exposure_flag", "outcome_flag"), ("exposure_flag", "outcome_flag")),
        ((" exposure_flag ", "exposure_flag", "", None), ("exposure_flag",)),
        (("not_a_variable", "outcome_flag"), ("outcome_flag",)),
        (("not_a_variable",), ()),
    ],
    ids=["both", "padded-repeated-empty", "one-allowed", "none-allowed"],
)
def test_the_anchor_rule_names_the_anchors_the_refusal_names(anchors, listable):
    variables = ["exposure_flag", "outcome_flag", "age_years", "sex_code"]
    assert listable_question_anchors(anchors, variables) == listable
    rule = question_anchor_rule_text(anchors, variables)
    payload = _outline_payload()
    for candidate in payload["design_selection"]["candidates"]:
        candidate["required_variables"] = ["age_years", "sex_code"]
    with pytest.raises(ProgressivePlanCompileError) as refused:
        validate_outline_design_selection(
            ProgressivePlanOutline.model_validate(payload),
            allowed_analysis_types=[payload["analysis_type"]],
            allowed_variables=variables,
            allowed_literature_citation_keys=[],
            question_anchors=[str(anchor or "") for anchor in anchors],
            required=True,
        )
    assert refused.value.reason_code == _ANCHOR_CODE
    assert refused.value.details["findings"] == [{"question_anchors": list(listable)}]
    if not listable:
        assert rule == ""
        return
    named = " or ".join(repr(anchor) for anchor in listable)
    assert f"list {named} among" in rule
    assert f"list {named} among" in str(refused.value)


def test_the_outline_prompt_states_the_anchor_rule_with_the_runs_anchors():
    context = _context()
    variables = select_progressive_variables(context)
    _ids, rows = outline_action_catalog(("association_study",))
    prompt = ProgressivePlannerAgent._user_prompt(
        context,
        analysis_types=("association_study",),
        variables=variables,
        action_rows=rows,
    )
    anchors = (context.primary_exposure, context.target_outcome)
    rule = question_anchor_rule_text(anchors, variables)
    assert rule
    block = prompt.split("Research question and sealed study anchors:\n", 1)[1]
    assert rule in block.split("\n\n", 1)[0]
