"""The worked example in the Planner prompt must match the real schema.

The prompt teaches two typed specs by showing them. When a spec is
described in prose but never shown, a real Planner guesses its shape: on
2026-07-29 one guessed ``exposure_outcome_distribution_spec`` four
different ways in five paid attempts -- ``exposure_column`` for
``exposure``, and twice an ``{"column": ..., "levels": ...}`` object
where a bare column name belongs.

An example is only worth showing while it is still true, so this parses
the example out of the prompt the Planner actually receives, substitutes
its ``<placeholder>`` slots, and checks the resulting step against the
real schema *and* against the deterministic owner that must claim it.
Change a field name or a Literal without updating the example and this
fails.

The same applies to a closed vocabulary the prompt forbids inventing:
the robustness ``axis`` set is asserted to be published, read off the
contract that enforces it.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.core import _build_planner_user_prompt
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (  # noqa: E501
    exposure_outcome_distribution_executor_owns_step,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ExposureOutcomeDistributionSpec,
    ResearchContext,
    VariableRole,
)

# The example uses <angle-bracket> slots for values the Planner supplies.
# They are filled with the simplest values that satisfy the schema, so the
# test exercises the example's *structure* -- key names and nesting -- not
# the placeholder prose.
_PLACEHOLDER_FILL: dict[str, Any] = {
    "<one sentence>": "Describe the step.",
    "<variable names from context>": "exposure_flag",
    "<the single typed cohort input>": "analysis_cohort",
    "<declared grouping variable>": "exposure_flag",
    "<declared row variable>": "age",
    "<declared exposure column name>": "exposure_flag",
    "<declared outcome column name>": "outcome_flag",
    "<closed level 1>": 0,
    "<closed level 2>": 1,
    "<exactly one of outcome_levels>": 1,
}


def _fill(value: Any) -> Any:
    if isinstance(value, str):
        return _PLACEHOLDER_FILL.get(value, value)
    if isinstance(value, list):
        return [_fill(item) for item in value]
    if isinstance(value, dict):
        return {key: _fill(item) for key, item in value.items()}
    return value


def _minimal_context() -> ResearchContext:
    """Just enough context to render the prompt; the example is static."""

    return ResearchContext(
        research_question="Does the worked example still match the schema?",
        cohort=CohortDescriptor(
            cohort_name="example",
            database="miiv",
            n_patients=10,
            n_stays=10,
            id_columns=["stay_id"],
            outcome_columns=["outcome_flag"],
        ),
        variables=[
            ConceptDescriptor(
                name="exposure_flag",
                role=VariableRole.COMPOSITE_SCORE,
                source_concept="exposure_flag",
                dtype="int",
            ),
            ConceptDescriptor(
                name="outcome_flag",
                role=VariableRole.OUTCOME,
                source_concept="outcome_flag",
                dtype="int",
            ),
        ],
        target_outcome="outcome_flag",
        primary_exposure="exposure_flag",
    )


def _example_plan_payload() -> dict:
    """Pull the worked example out of the prompt the Planner receives."""

    prompt = _build_planner_user_prompt(_minimal_context())

    # The example is the first embedded object carrying a "steps" array.
    start = prompt.find('{\n  "research_question"')
    assert start != -1, "the planner prompt no longer contains a worked example"

    depth = 0
    for index in range(start, len(prompt)):
        char = prompt[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(prompt[start : index + 1])
    raise AssertionError("the worked example in the planner prompt is unbalanced")


def _distribution_example_step() -> dict:
    payload = _fill(_example_plan_payload())
    declaring = [
        step
        for step in payload["steps"]
        if "table:exposure_outcome_distribution" in step.get("expected_outputs", [])
    ]
    assert declaring, (
        "no example step declares table:exposure_outcome_distribution, so the "
        "Planner is told the spec is mandatory without ever being shown one"
    )
    assert len(declaring) == 1, "the example should show this product exactly once"
    return declaring[0]


def test_the_example_step_validates_as_a_real_analysis_step() -> None:
    """Whole-step validation, so cross-field rules are exercised too.

    The whole *plan* is deliberately not validated: the surrounding example
    is schematic, with single placeholders standing for lists, and it never
    validated literally. Asserting that would fail for reasons unrelated to
    the spec being taught here. The step is the unit that must be right.
    """

    step = _distribution_example_step()

    try:
        AnalysisStep.model_validate(step)
    except ValidationError as exc:  # pragma: no cover - failure path is the point
        pytest.fail(
            "the distribution example in the Planner prompt no longer matches "
            f"AnalysisStep; the Planner is being shown a shape the host "
            f"rejects:\n{exc}"
        )


def test_the_real_owner_claims_the_step_the_example_teaches() -> None:
    """The strongest form of the guard: the example produces an owned step.

    A shape that merely validates can still fall through to the Coder. What
    the example is for is a step the deterministic owner takes, so that is
    what is asserted.
    """

    step = AnalysisStep.model_validate(_distribution_example_step())

    assert exposure_outcome_distribution_executor_owns_step(step), (
        "the example validates but the deterministic owner refuses it, so a "
        "Planner copying it would still fall through to the Coder"
    )


def test_the_example_spec_validates_on_its_own() -> None:
    """Named separately so a spec-shape error is not one line of a step error."""

    spec = _distribution_example_step()["exposure_outcome_distribution_spec"]

    assert isinstance(spec, dict)
    parsed = ExposureOutcomeDistributionSpec.model_validate(spec)
    assert parsed.risk_difference_contrast is not None
    assert parsed.risk_difference_contrast.reference_exposure_level == 0
    assert parsed.risk_difference_contrast.comparison_exposure_level == 1
    # The Planner chooses contrast direction but never fabricates host grouping.
    assert parsed.dependence is None


def test_the_closed_robustness_axis_vocabulary_is_published_to_the_planner() -> None:
    """ "Never invent an unsupported axis" was there; the supported set was not.

    A real Planner guessed ``model`` on one run and ``functional_form`` on the
    next, each costing an attempt against a three-value closed set it was
    never shown. Read from the contract so the sentence cannot drift from what
    would reject it.
    """

    prompt = _build_planner_user_prompt(_minimal_context())
    allowed = typing.get_args(typing.get_type_hints(RobustnessSpec)["axis"])

    assert allowed, "RobustnessSpec.axis is no longer a closed Literal"
    for value in allowed:
        assert f"'{value}'" in prompt, (
            f"the Planner is forbidden from inventing an axis but is never "
            f"shown that {value!r} is allowed"
        )

    # The guesses must be steered somewhere, not merely refused.
    assert "separate analysis step" in prompt


@pytest.mark.parametrize(
    "wrong_shape",
    [
        pytest.param({"exposure_column": "exposure_flag"}, id="exposure_column"),
        pytest.param(
            {"exposure": {"column": "exposure_flag", "levels": [0, 1]}},
            id="nested_exposure_object",
        ),
    ],
)
def test_the_shapes_a_real_planner_guessed_are_still_refused(
    wrong_shape: dict,
) -> None:
    """Guard the example rather than widening the schema to accept the guesses."""

    base = {
        "exposure": "exposure_flag",
        "exposure_levels": [0, 1],
        "outcome": "outcome_flag",
        "outcome_levels": [0, 1],
        "outcome_positive_value": 1,
        "level_match_policy": "exact_typed",
        "denominator_policy": "all_declared_rows",
        "missing_outcome_policy": "structural_absence_is_non_event",
        "confidence_level": 0.95,
    }
    payload = {**base, **wrong_shape}
    if "exposure_column" in wrong_shape:
        payload.pop("exposure")

    with pytest.raises(ValidationError):
        ExposureOutcomeDistributionSpec.model_validate(payload)
