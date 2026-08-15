from __future__ import annotations

from easyicu.research_agent.agents.core import PlannerAgent
from easyicu.research_agent.research_context.prompt_scope import (
    scoped_planner_context,
    scoped_reporting_context,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


def _variable(
    name: str,
    *,
    role: VariableRole = VariableRole.OTHER,
    source: str | None = None,
    description: str = "",
) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        role=role,
        dtype="float64",
        source_concept=source,
        description=description,
        analysis_window="icu_admission[0,24]h" if source else None,
        observed_domain={"min": 0.0, "max": 4.0, "n_unique": 5},
    )


def _wide_context() -> ResearchContext:
    variables = [
        _variable("stay_id", role=VariableRole.ID),
        _variable("age", role=VariableRole.DEMOGRAPHIC),
        _variable("sex", role=VariableRole.DEMOGRAPHIC),
        _variable("death", role=VariableRole.OUTCOME),
        _variable(
            "renal_score_max",
            source="renal_score",
            description="Renal severity score",
        ),
        _variable("renal_score_n", role=VariableRole.META, source="renal_score"),
        _variable(
            "renal_score_measured",
            role=VariableRole.META,
            source="renal_score",
        ),
        _variable(
            "renal_component_max",
            source="renal_component",
            description="Renal component severity score",
        ),
        _variable(
            "renal_component_n", role=VariableRole.META, source="renal_component"
        ),
        _variable(
            "renal_component_measured",
            role=VariableRole.META,
            source="renal_component",
        ),
    ]
    variables.extend(
        _variable(
            f"unrelated_signal_{index}_mean",
            source=f"unrelated_signal_{index}",
            description="Unrelated physiologic signal",
        )
        for index in range(95)
    )
    return ResearchContext(
        research_question=(
            "Characterise the association between first-24h renal_score_max "
            "and death in adult ICU patients."
        ),
        cohort=CohortDescriptor(
            cohort_name="wide",
            database="miiv",
            n_patients=1000,
            n_stays=1000,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=variables,
        target_outcome="death",
        primary_exposure="renal_score_max",
    )


def test_wide_planner_context_is_scoped_without_mutating_authority() -> None:
    context = _wide_context()
    before = context.model_dump_json()

    scoped = scoped_planner_context(context)

    assert context.model_dump_json() == before
    assert len(context.variables) == 105
    assert {variable.name for variable in scoped.variables} >= {
        "stay_id",
        "age",
        "sex",
        "death",
        "renal_score_max",
        "renal_score_n",
        "renal_score_measured",
        "renal_component_max",
        "renal_component_n",
        "renal_component_measured",
    }
    assert "unrelated_signal_0_mean" not in {
        variable.name for variable in scoped.variables
    }


def test_typed_sensitivity_variable_and_binding_reach_the_planner() -> None:
    context = _wide_context().model_copy(
        update={
            "user_preferences": UserPreferences(
                sensitivity_specs=[
                    {
                        "spec_id": "nonlinear_signal_check",
                        "axis": "functional_form",
                        "strategy": "restricted_cubic_spline",
                        "execution_variables": ["unrelated_signal_0_mean"],
                    }
                ]
            )
        }
    )

    scoped = scoped_planner_context(context)
    prompt = PlannerAgent.request_messages(context)[1].content

    assert "unrelated_signal_0_mean" in {
        variable.name for variable in scoped.variables
    }
    assert "nonlinear_signal_check" in prompt
    assert "sensitivity_spec_ids" in prompt
    assert "restricted_cubic_spline_sensitivity" in prompt


def test_wide_planner_request_preserves_catalog_and_stays_bounded() -> None:
    context = _wide_context()

    messages = PlannerAgent.request_messages(context)
    prompt = messages[1].content
    metrics = PlannerAgent.request_metrics(context)

    assert metrics["total_bytes"] < metrics["limit_bytes"]
    assert "full_variable_count=105" in prompt
    assert "full_roster_sha256=" in prompt
    assert "unrelated_signal_0_mean | role=other | dtype=float64" in prompt
    detailed, catalog = prompt.split("PLANNER VARIABLE RESOURCE PROJECTION", maxsplit=1)
    assert "renal_score_max" in detailed
    assert "renal_score_n" in detailed
    assert "unrelated_signal_0_mean" not in detailed
    assert "unrelated_signal_0_mean" in catalog
    assert "You MAY select a catalog column" in catalog


def test_planner_catalog_deduplicates_shared_metadata_without_losing_columns() -> None:
    context = _wide_context()
    prompt = PlannerAgent.request_messages(context)[1].content
    catalog = prompt.split("PLANNER VARIABLE RESOURCE PROJECTION", maxsplit=1)[1]

    assert catalog.count("W1=icu_admission[0,24]h") == 1
    assert (
        "unrelated_signal_0_mean | role=other | dtype=float64 | "
        "source=unrelated_signal_0 | window_ref=W1"
    ) in catalog
    for index in range(95):
        assert catalog.count(f"unrelated_signal_{index}_mean |") == 1


def test_reporting_scope_keeps_study_coordinates_not_discovery_roster() -> None:
    context = _wide_context()
    before = context.model_dump_json()
    scoped = scoped_reporting_context(context)

    assert context.model_dump_json() == before
    assert {variable.name for variable in scoped.variables} >= {
        "stay_id",
        "age",
        "sex",
        "death",
        "renal_score_max",
        "renal_score_n",
        "renal_score_measured",
    }
    assert not any(
        variable.name.startswith("unrelated_signal_")
        for variable in scoped.variables
    )


def test_resource_scope_is_case_neutral() -> None:
    source = scoped_planner_context.__code__.co_consts
    rendered = " ".join(value for value in source if isinstance(value, str)).lower()

    assert "kdigo" not in rendered
    assert "lactate" not in rendered
    assert "vasopressor" not in rendered
