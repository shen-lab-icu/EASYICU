"""A design that binds no question anchor is refused with the anchors it can list.

The selected design must bind one of the run's question anchors (the primary
exposure, the target outcome).  The outline prompt shows the anchors but not
that the selected design must list one among its required_variables, so a
refusal that only says an anchor is missing leaves the Planner to write the
same selection again.  The refusal names every anchor among the run's allowed
variables; every other design refusal keeps the owner's own words.  Outlines
and contexts are synthetic.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.design_selection import (
    ResearchDesignSelectionError,
    validate_research_design_selection,
)
from easyicu.research_agent.planning.outline_design_selection import (
    validate_outline_design_selection,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _outline_payload,
)

_CODE = "progressive_design_selection_question_anchor_missing"
_VARIABLES = ["exposure_flag", "outcome_flag", "age_years", "sex_code"]


def _outline(*required_variables):
    payload = _outline_payload()
    if required_variables:
        for candidate in payload["design_selection"]["candidates"]:
            candidate["required_variables"] = list(required_variables)
    return ProgressivePlanOutline.model_validate(payload)


def _refused(outline, *, anchors, variables=_VARIABLES):
    with pytest.raises(ProgressivePlanCompileError) as refused:
        validate_outline_design_selection(
            outline,
            allowed_analysis_types=[outline.analysis_type],
            allowed_variables=variables,
            allowed_literature_citation_keys=[],
            question_anchors=anchors,
            required=True,
        )
    return refused.value


def _owner_message(outline, *, anchors, variables=_VARIABLES):
    with pytest.raises(ResearchDesignSelectionError) as refused:
        validate_research_design_selection(
            outline.design_selection,
            selected_analysis_type=outline.analysis_type,
            allowed_analysis_types=[outline.analysis_type],
            allowed_variables=variables,
            allowed_literature_citation_keys=[],
            question_anchors=anchors,
            required=True,
        )
    return str(refused.value)


def test_the_refusal_names_each_anchor_the_design_can_list():
    outline = _outline("age_years", "sex_code")
    anchors = ("exposure_flag", "outcome_flag")

    found = _refused(outline, anchors=anchors)

    assert found.reason_code == _CODE
    assert found.path == "design_selection.candidates.required_variables"
    assert found.details["message"] == (
        _owner_message(outline, anchors=anchors)
        + ": list 'exposure_flag' or 'outcome_flag' among the selected design's "
        "required_variables"
    )
    assert found.details["findings"] == [
        {"question_anchors": ["exposure_flag", "outcome_flag"]}
    ]


def test_an_empty_or_repeated_anchor_is_named_once():
    # The owner strips each anchor before it compares, so a padded one counts.
    found = _refused(
        _outline("age_years", "sex_code"),
        anchors=("exposure_flag", "", "exposure_flag", " outcome_flag "),
    )

    assert found.details["message"].endswith(
        ": list 'exposure_flag' or 'outcome_flag' among the selected design's "
        "required_variables"
    )
    assert found.details["findings"] == [
        {"question_anchors": ["exposure_flag", "outcome_flag"]}
    ]


def test_an_anchor_outside_the_allowed_variables_is_not_offered():
    variables = ["outcome_flag", "age_years", "sex_code"]
    outline = _outline("age_years", "sex_code")

    one = _refused(
        outline, anchors=("exposure_flag", "outcome_flag"), variables=variables
    )
    none = _refused(outline, anchors=("exposure_flag",), variables=variables)

    assert "'exposure_flag'" not in one.details["message"]
    assert one.details["message"].endswith(
        ": list 'outcome_flag' among the selected design's required_variables"
    )
    # No anchor can be listed: the owner's words stand, and the finding is empty.
    assert none.details["message"] == _owner_message(
        outline, anchors=("exposure_flag",), variables=variables
    )
    assert none.details["findings"] == [{"question_anchors": []}]


def test_any_other_refusal_keeps_the_owners_words():
    outline = _outline("exposure_flag", "unlisted_marker")
    anchors = ("exposure_flag", "outcome_flag")

    found = _refused(outline, anchors=anchors)

    assert found.reason_code == "progressive_design_selection_variable_unavailable"
    assert found.details["message"] == _owner_message(outline, anchors=anchors)
    assert "findings" not in found.details


def test_a_design_that_binds_an_anchor_is_accepted():
    validate_outline_design_selection(
        _outline(),
        allowed_analysis_types=["association_study"],
        allowed_variables=_VARIABLES,
        allowed_literature_citation_keys=[],
        question_anchors=("exposure_flag", "outcome_flag"),
        required=True,
    )


def test_the_planners_outline_check_names_the_anchor():
    with pytest.raises(ProgressivePlanCompileError) as refused:
        ProgressivePlannerAgent._validate_outline_authority(
            _outline("age_years", "sex_code"),
            analysis_types=["association_study"],
            variable_names=_VARIABLES,
            allowed_literature_citation_keys=[],
            primary_exposure="exposure_flag",
            target_outcome="outcome_flag",
        )

    assert refused.value.reason_code == _CODE
    assert "list 'exposure_flag' or 'outcome_flag'" in str(refused.value)
