from __future__ import annotations

from easyicu.research_agent.agents.core import PlannerAgent
from easyicu.research_agent.research_context.prompt_scope import scoped_planner_context
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
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


def test_resource_scope_is_case_neutral() -> None:
    source = scoped_planner_context.__code__.co_consts
    rendered = " ".join(value for value in source if isinstance(value, str)).lower()

    assert "kdigo" not in rendered
    assert "lactate" not in rendered
    assert "vasopressor" not in rendered
