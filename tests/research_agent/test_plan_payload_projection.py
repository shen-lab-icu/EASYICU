"""The Planner payload projection must accept exactly what the schema declares.

``_normalise_plan_payload`` exists to drop keys a hosted model invents,
because the plan schemas are ``extra="forbid"`` and would otherwise reject
an entire plan over one hallucinated field. It must not also decide which
*declared* fields the Planner is allowed to fill -- but a hand-written
allowlist silently takes on that second job the moment a field is added to
a schema and not to the copy.

Two such drifts were live at once, and they fail differently:

* ``AnalysisStep.exposure_outcome_distribution_spec`` was required by the
  Planner validator and deleted by the projection, so the retry loop spent
  every attempt asking the model to send a field that was being thrown away
  before the check ran. Unwinnable by construction.
* ``TableOneSpec.standardized_difference_mode`` had no downstream validator
  demanding it, so it was deleted in silence and Table 1 was reported under
  whatever the default was, with no error anywhere.

The first test below is the durable one: it fails the next time any of
these five schemas grows a field, without naming any field itself.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from easyicu.research_agent.agents.core import (
    _declared_field_names,
    _normalise_plan_payload,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    PlannedModelRequirement,
    TableOneSpec,
    TableOneVariableSpec,
)


def test_plan_payload_owner_is_separate_from_agent_orchestration() -> None:
    """The compatibility export stays stable while the non-agent owner holds logic."""

    from easyicu.research_agent.agents import core, plan_payload

    assert core._normalise_plan_payload is plan_payload._normalise_plan_payload
    assert "def _normalise_plan_payload" not in inspect.getsource(core)


def _schema_fields(model: type) -> set:
    """Read the declared field names straight off the type.

    Deliberately NOT ``_declared_field_names``: the projection under test
    builds its accepted keys with that helper, so a test that also used it
    would compare the code against itself and pass no matter how wrong the
    helper became. Confirmed by mutation -- reintroducing the real drift
    left the earlier version of this test green.
    """

    fields = getattr(model, "model_fields", None)
    if fields is not None:
        return set(fields)
    return {field.name for field in dataclasses.fields(model)}


def _every_declared_field(model: type) -> dict:
    """A payload carrying every key ``model`` declares.

    The projection filters on key names and never inspects values, so a
    ``None`` placeholder is enough to prove a key survives.
    """

    return {name: None for name in _schema_fields(model)}


def _maximal_payload() -> dict:
    """One plan exercising all five projected levels at once."""

    table_one = _every_declared_field(TableOneSpec)
    table_one["variables"] = [_every_declared_field(TableOneVariableSpec)]

    step = _every_declared_field(AnalysisStep)
    step["step_id"] = "01_step"
    step["expected_outputs"] = []
    step["table_one_spec"] = table_one
    step["model_requirements"] = [_every_declared_field(PlannedModelRequirement)]
    step["input_consumption_contracts"] = [
        _every_declared_field(ArtifactConsumptionContract)
    ]

    robustness = _every_declared_field(RobustnessSpec)
    robustness["spec_id"] = "r1"

    plan = _every_declared_field(AnalysisPlan)
    plan["steps"] = [step]
    plan["robustness_specs"] = [robustness]
    return plan


def test_the_projection_accepts_every_field_the_schemas_declare() -> None:
    """Add a field to any projected schema and this fails until it is accepted."""

    normalized, dropped = _normalise_plan_payload(_maximal_payload())

    assert dropped == {bucket: [] for bucket in dropped}, (
        "the projection discarded a field its own schema declares: "
        f"{ {k: v for k, v in dropped.items() if v} }"
    )

    step = normalized["steps"][0]
    assert set(normalized) == _schema_fields(AnalysisPlan)
    assert set(step) == _schema_fields(AnalysisStep)
    assert set(step["table_one_spec"]) == _schema_fields(TableOneSpec)
    assert set(step["table_one_spec"]["variables"][0]) == _schema_fields(
        TableOneVariableSpec
    )
    assert set(step["model_requirements"][0]) == _schema_fields(PlannedModelRequirement)
    assert set(step["input_consumption_contracts"][0]) == _schema_fields(
        ArtifactConsumptionContract
    )
    assert set(normalized["robustness_specs"][0]) == _schema_fields(RobustnessSpec)


def test_a_key_no_schema_declares_is_still_dropped() -> None:
    """The projection keeps doing the job it exists for."""

    payload = _maximal_payload()
    payload["invented_top_level"] = "x"
    payload["steps"][0]["invented_step_key"] = "x"
    payload["steps"][0]["table_one_spec"]["invented_table_one_key"] = "x"

    _normalized, dropped = _normalise_plan_payload(payload)

    assert dropped["top_level"] == ["invented_top_level"]
    assert dropped["steps"] == ["01_step:invented_step_key"]
    assert dropped["table_one_spec"] == ["step[0]:invented_table_one_key"]


def test_the_distribution_spec_reaches_the_planner_validator() -> None:
    """The exact step shape that burned five paid Planner attempts."""

    payload = {
        "research_question": "q",
        "steps": [
            {
                "step_id": "03_prevalence_mortality_distribution",
                "expected_outputs": ["table:exposure_outcome_distribution"],
                "exposure_outcome_distribution_spec": {
                    "schema_version": 1,
                    "exposure": "exposure_flag",
                    "exposure_levels": [0, 1],
                    "outcome": "outcome_flag",
                    "outcome_levels": [0, 1],
                    "outcome_positive_value": 1,
                    "level_match_policy": "exact_typed",
                    "denominator_policy": "all_declared_rows",
                    "missing_outcome_policy": "structural_absence_is_non_event",
                    "confidence_level": 0.95,
                },
            }
        ],
    }

    normalized, dropped = _normalise_plan_payload(payload)

    spec = normalized["steps"][0]["exposure_outcome_distribution_spec"]
    assert spec["outcome_positive_value"] == 1
    assert spec["denominator_policy"] == "all_declared_rows"
    assert not dropped["steps"]


def test_the_table_one_standardized_difference_mode_is_not_deleted() -> None:
    """The drift that failed silently: no validator would have caught this."""

    payload = {
        "research_question": "q",
        "steps": [
            {
                "step_id": "02_table_one",
                "expected_outputs": ["table:table_one"],
                "table_one_spec": {
                    "group_by": "exposure_flag",
                    "standardized_difference_mode": "pooled",
                },
            }
        ],
    }

    normalized, dropped = _normalise_plan_payload(payload)

    table_one = normalized["steps"][0]["table_one_spec"]
    assert table_one["standardized_difference_mode"] == "pooled"
    assert not dropped["table_one_spec"]


def test_a_type_that_declares_no_fields_is_refused_rather_than_treated_as_empty() -> (
    None
):
    """An empty accepted-key set would drop every key the Planner sent."""

    class NotASchema:
        pass

    with pytest.raises(TypeError, match="cannot be read"):
        _declared_field_names(NotASchema)
