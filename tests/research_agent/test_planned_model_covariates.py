"""The adjustment set has to be declared, because it cannot be inferred.

``table:adjusted_association_estimates`` is the single most frequently declared
product no deterministic owner can emit -- 233 of 1812 recorded real steps, and
most of them declare it alone, so it is not a bundling problem.  It is also the
paper's primary result, written today by the LLM coder every time.

``PlannedModelRequirement`` already fixes outcome, outcome_type, method_family,
exposure_source, analysis_role and analysis_set.  The one thing missing is the
adjustment set.  On the real fresh19 step it *looks* derivable -- inputs minus
exposure minus outcome minus the registered ``_measured``/``_n`` companions
leaves exactly age, sex and charlson_max -- and
``test_the_adjustment_set_is_not_recoverable_from_inputs`` is why the host must
not do that arithmetic anyway.

``test_null_and_empty_are_different_declarations`` is the load-bearing one: a
host that read "not declared" as "no covariates" would fit an unadjusted model
and report it under the pre-specified adjusted estimand's name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.core import _declared_field_names
from easyicu.research_agent.schema import AnalysisStep, PlannedModelRequirement

_FIXTURE = Path(__file__).parent / "fixtures" / "real_plan_steps_fresh17_fresh19.json"
# The real fresh19 primary step, whose requirement carries no adjustment set.
_REAL_STEP_ID = "07_primary_adjusted_association"


def _requirement(**overrides) -> PlannedModelRequirement:
    payload = {
        "requirement_id": "primary_full_cohort_logistic",
        "outcome": "death",
        "outcome_type": "binary",
        "method_family": "logistic_regression",
        "exposure_source": "sep3_sofa2_max",
        "analysis_role": "primary",
        "analysis_set": "source_aware",
        "required_for_step_success": True,
    }
    payload.update(overrides)
    return PlannedModelRequirement.model_validate(payload)


def _real_step() -> dict:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    plan = next(e for e in document["plans"] if e["label"] == "fresh19")["plan"]
    return next(s for s in plan["steps"] if s["step_id"] == _REAL_STEP_ID)


def test_null_and_empty_are_different_declarations() -> None:
    """ "The planner did not say" is not "the planner said none"."""

    assert _requirement().covariates is None
    assert _requirement(covariates=[]).covariates == []


def test_a_declared_adjustment_set_is_kept_exactly_and_in_order() -> None:
    requirement = _requirement(covariates=["age", "sex", "charlson_max"])

    assert requirement.covariates == ["age", "sex", "charlson_max"]


def test_the_outcome_is_refused_as_a_covariate() -> None:
    """Conditioning on the outcome estimates a different quantity."""

    with pytest.raises(ValidationError, match="must not contain the outcome"):
        _requirement(covariates=["age", "death"])


def test_the_exposure_is_refused_as_a_covariate() -> None:
    """Adjusting for the exposure removes the declared association."""

    with pytest.raises(ValidationError, match="must not contain the exposure"):
        _requirement(covariates=["age", "sep3_sofa2_max"])


@pytest.mark.parametrize("bad", [["age", ""], ["age", "   "], ["age", "age"]])
def test_a_blank_or_repeated_covariate_is_refused(bad) -> None:
    with pytest.raises(ValidationError):
        _requirement(covariates=bad)


def test_the_real_recorded_step_still_validates_without_the_field() -> None:
    """Every plan already on disk predates this field and must still load.

    A required field here would have made the whole recorded corpus -- and any
    resumed run -- unreadable.
    """

    step = AnalysisStep.model_validate(_real_step())

    (requirement,) = step.model_requirements
    assert requirement.covariates is None
    assert requirement.exposure_source == "sep3_sofa2_max"
    assert requirement.outcome == "death"


def test_the_adjustment_set_is_not_recoverable_from_inputs() -> None:
    """The subtraction that must not be turned into host behaviour.

    On this real step the arithmetic happens to land on the right answer, which
    is exactly what makes it dangerous: ``inputs`` states what the step may
    read, not what the model conditions on.  A mediator, a post-exposure
    variable, or a column carried only for a table would be swept in silently,
    and the run would still report a pre-specified adjusted estimand.
    """

    step = AnalysisStep.model_validate(_real_step())
    (requirement,) = step.model_requirements
    companions = {
        value
        for value in step.inputs
        if value.endswith("_measured") or value.endswith("_n")
    }
    subtracted = [
        value
        for value in step.inputs
        if ":" not in value
        and value not in companions
        and value != requirement.exposure_source
        and value != requirement.outcome
    ]

    assert subtracted == ["age", "sex", "charlson_max"]
    # ...and the requirement still says nothing, which is the honest state.
    assert requirement.covariates is None


def test_the_planner_payload_projection_keeps_the_new_field() -> None:
    """A field the schema declares but the projection drops is invisible.

    ``_normalise_plan_payload`` exists to discard keys the model invented; it
    derives its allow-list from the schema, so this asserts that derivation
    rather than a second hand-maintained list.
    """

    assert "covariates" in _declared_field_names(PlannedModelRequirement)


def test_the_planner_is_told_the_field_exists() -> None:
    """A contract field no prompt mentions is a field no plan will ever carry."""

    from easyicu.research_agent.agents import core

    source = Path(core.__file__).read_text(encoding="utf-8")

    assert "`covariates`" in source
    assert "not reconstruct an adjustment set from the step inputs" in source
