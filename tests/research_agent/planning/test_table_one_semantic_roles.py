"""Identity coordinates are not clinical Table 1 rows or strata."""

import pytest

from easyicu.research_agent.authority.table_one_binding import bind_table_one_execution_spec
from easyicu.research_agent.planning.planner_output_contract import (
    validate_fresh_planner_typed_product_specs,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressivePlanOutline,
)
from easyicu.research_agent.planning.progressive_host_materialization import (
    host_materialize_progressive_step,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ConceptDescriptor,
    VariableRole,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _outline_payload,
)


def _with_coordinate(authority):
    context = _context()
    context.variables.append(
        ConceptDescriptor(
            name="record_key", dtype="int64", role=authority,
            observed_domain={"levels": [0, 1], "n_unique": 2, "is_binary": True},
        )
    )
    if authority == VariableRole.OTHER:
        context.cohort.id_columns.append("record_key")
    return context


def _step(*, coordinate_is_group=False):
    return AnalysisStep(
        step_id="baseline", intent="Describe the declared clinical variables.",
        method="table_one", expected_outputs=["table:table_one"],
        inputs=["record_key", "exposure_flag", "age_years"],
        table_one_spec={
            "group_by": "record_key" if coordinate_is_group else "exposure_flag",
            "group_levels": [0, 1],
            "variables": [{
                "name": "age_years" if coordinate_is_group else "record_key",
                "variable_kind": "continuous", "summary": "median_iqr",
                "test": "mann_whitney_or_kruskal",
            }],
        },
    )


@pytest.mark.parametrize("authority", [VariableRole.ID, VariableRole.INDEX, VariableRole.OTHER])
def test_host_table_one_excludes_role_and_cohort_declared_identifiers(authority):
    context = _with_coordinate(authority)
    payload = _outline_payload()
    payload["steps"][1]["variable_names"].append("record_key")
    outline = ProgressivePlanOutline.model_validate(payload)
    result = host_materialize_progressive_step(
        context=context, outline=outline, outline_step=outline.steps[1],
        foundation=ProgressiveFoundationMaterialization.model_validate(_foundation_payload()).foundation,
        available_product_refs=(("01_cohort", "artifact:analysis_cohort"),),
    )
    assert result is not None
    assert {row.name for row in result.step.table_one_variables} == {
        "age_years", "sex_code", "outcome_flag"
    }


@pytest.mark.parametrize("authority", [VariableRole.ID, VariableRole.INDEX, VariableRole.OTHER])
@pytest.mark.parametrize("coordinate_is_group", [False, True])
def test_fresh_plan_and_execution_binding_reject_identity_statistics(authority, coordinate_is_group):
    context = _with_coordinate(authority)
    step = _step(coordinate_is_group=coordinate_is_group)
    plan = AnalysisPlan(research_question=context.research_question, steps=[step])

    with pytest.raises(ValueError, match="table_one_identity_coordinate_ineligible"):
        validate_fresh_planner_typed_product_specs(plan, context=context)
    with pytest.raises(ValueError, match="table_one_identity_coordinate_ineligible"):
        bind_table_one_execution_spec(step, context)
    assert step._table_one_execution_binding is None


def test_column_spelling_does_not_invent_identity_authority():
    context = _with_coordinate(VariableRole.OTHER)
    context.cohort.id_columns.remove("record_key")
    step = _step()
    plan = AnalysisPlan(research_question=context.research_question, steps=[step])

    validate_fresh_planner_typed_product_specs(plan, context=context)
    assert bind_table_one_execution_spec(step, context) is not None
