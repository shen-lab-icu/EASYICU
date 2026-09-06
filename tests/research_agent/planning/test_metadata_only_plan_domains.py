"""Declared domains advertised to Planner must reach its compiler unchanged."""

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    _continuous_planning_variable_names,
)
from easyicu.research_agent.authority.declared_levels import closed_planning_levels_for
from easyicu.research_agent.planning.progressive_compiler import _compile_table_one
from easyicu.research_agent.planning.progressive_contract import ProgressiveSkeletonStep
from easyicu.research_agent.planning.progressive_host_materialization import _table_summary
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload


def _catalog_context():
    context = _context()
    sources = {"exposure_flag": "sex", "sex_code": "adm"}
    variables = [
        variable.model_copy(update={
            "observed_domain": None,
            "source_concept": sources.get(variable.name),
            "dtype": "float64",
        })
        for variable in context.variables
    ]
    return context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"n_stays": 0}),
        "variables": variables,
    })


def test_declared_catalog_category_is_not_advertised_as_continuous():
    context = _catalog_context()
    cards = ProgressivePlannerAgent._retrieved_data_cards(context, ("sex_code",))
    assert cards[0]["supports_closed_level_contrast"]
    assert "sex_code" not in _continuous_planning_variable_names(context)
    assert "age_years" in _continuous_planning_variable_names(context)


def test_host_table_summary_preserves_declared_categorical_semantics():
    variable = _catalog_context().variable("sex_code")
    assert _table_summary(variable) == "count_percent"
    assert variable.observed_domain is None


def test_table_one_compiles_the_same_unobserved_declared_domain_as_data_card():
    context = _catalog_context()
    step = ProgressiveSkeletonStep.model_validate(_payload()["steps"][1])
    result = _compile_table_one(
        context=context,
        variables={variable.name: variable for variable in context.variables},
        step=step, step_index=1,
    )
    assert result.group_levels == ["Female", "Male"]
    assert next(row for row in result.variables if row.name == "sex_code").levels == ["med", "surg", "other"]
    assert all(variable.observed_domain is None for variable in context.variables)


def test_observed_order_takes_precedence_over_dictionary_domain():
    variable = _catalog_context().variable("sex_code").model_copy(update={
        "observed_domain": {"levels": ["surg", "med"]},
    })
    assert closed_planning_levels_for(
        name=variable.name, variables={variable.name: variable},
    ) == ["surg", "med"]


def test_unknown_catalog_variable_has_no_invented_levels():
    variable = _catalog_context().variable("age_years")
    assert closed_planning_levels_for(
        name=variable.name, variables={variable.name: variable},
    ) == []
    assert closed_planning_levels_for(name="missing", variables={}) == []
