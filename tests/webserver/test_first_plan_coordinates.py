"""The first candidate plan reads the exposure the researcher actually named.

Every Web study used to need at least two planning rounds: the first candidate
was reviewed as ``scientific_capability_data_contract_unresolved`` because the
metadata-only planning coordinates had no exposure, and the typed exposure only
reached the StudyContext after that review.  The coordinate owner reads what
the researcher explicitly named; these tests fix two readings that were
missing, and that the reading stays a reading -- never a guess.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import research_launch_scientific, scientific_runtime_projection
from easyicu.webserver.study_intent import (
    deterministic_intent,
    explicit_exposure_aggregation,
)

_ADMISSION_QUESTION = (
    "在已准备好的 eICU Collaborative Research Database Demo v2.0.1 中，评估 ICU "
    "入院类型（内科、外科、其他）与住院死亡之间的关联。入院类型定义为 ICU 入院时"
    "的基线特征，不采用 landmark 设计。"
)
_KDIGO_QUESTION = (
    "在已准备好的 eICU Collaborative Research Database Demo v2.0.1 中，评估入 ICU "
    "24 小时内最高的 KDIGO AKI 分期（0/1/2/3）与住院死亡之间的关联；分期证据不足"
    "的病例单列为 unknown，不并入 0 级。"
)


@pytest.mark.parametrize(
    "question",
    [
        _ADMISSION_QUESTION,
        "Association of ICU admission type with in-hospital mortality",
    ],
)
def test_admission_type_is_read_as_the_named_exposure(question: str) -> None:
    slots = deterministic_intent(question)["slots"]
    assert slots["exposure"]["value"] == "adm"
    assert slots["exposure"]["provenance"] == "user_text"

    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=question, database="eicu_demo"
    )
    assert coordinates["primary_exposure"] == "adm"
    assert coordinates["target_outcome"] == "death"
    assert coordinates["execution_authorized"] is False


def test_admission_type_does_not_capture_readmission() -> None:
    slots = deterministic_intent("ICU readmission by sex")["slots"]
    assert (slots["exposure"] or {}).get("value") != "adm"
    assert (slots["outcome"] or {}).get("value") != "adm"


def _receipts(monkeypatch: pytest.MonkeyPatch, available: bool) -> list[str]:
    seen: list[str] = []

    def fake(path: object) -> bool:
        seen.append(str(path))
        return available

    monkeypatch.setattr(
        scientific_runtime_projection, "export_kdigo_strict_derivation_available", fake
    )
    return seen


def test_a_named_kdigo_stage_is_read_as_the_observability_preserving_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _receipts(monkeypatch, True)

    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=_KDIGO_QUESTION, database="eicu_demo", export_path="/bound/export"
    )

    assert coordinates["primary_exposure"] == "aki_stage_strict"
    # The strict stage is already the window reading; "最高" in the question
    # names an aggregation of the source concept, not of the derived stage.
    assert coordinates["primary_exposure_aggregation"] is None
    assert seen == ["/bound/export"]


def test_without_receipts_the_named_stage_is_left_for_the_runtime_to_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _receipts(monkeypatch, False)
    unbound = research_launch_scientific._metadata_only_planning_coordinates(
        question=_KDIGO_QUESTION, database="eicu_demo", export_path="/july/export"
    )
    assert unbound["primary_exposure"] == "aki_stage"
    # The source-concept reading is exactly what it was before.
    named = explicit_exposure_aggregation(_KDIGO_QUESTION, concept_id="aki_stage")
    assert unbound["primary_exposure_aggregation"] == (
        named.aggregation if named is not None else None
    )

    # Without a bound source the reading is never substituted either.
    seen = _receipts(monkeypatch, True)
    no_source = research_launch_scientific._metadata_only_planning_coordinates(
        question=_KDIGO_QUESTION, database="eicu_demo"
    )
    assert no_source["primary_exposure"] == "aki_stage"
    assert seen == []


def test_a_binary_aki_exposure_is_not_turned_into_a_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swapping a 0-3 stage for a yes/no AKI phenotype would change the question."""

    _receipts(monkeypatch, True)
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question="急性肾损伤与住院死亡的关联", database="eicu_demo", export_path="/x"
    )
    assert coordinates["primary_exposure"] in {None, "aki"}


_LACTATE_QUESTION = "研究成人 ICU 患者前24小时最高乳酸与院内死亡"
_OUTPUTS = {"lact_v2": "lact", "lact_v3": "lact", "death_v2": "death"}


def _bound_source(monkeypatch: pytest.MonkeyPatch, carried: tuple[str, ...]) -> None:
    from types import SimpleNamespace

    from easyicu import concept_output_sources
    from easyicu.research_agent.acquisition import catalog as acquisition_catalog

    monkeypatch.setattr(concept_output_sources, "COMPOSITE_CONCEPT_OUTPUT_SOURCES", _OUTPUTS)
    monkeypatch.setattr(
        acquisition_catalog,
        "build_available_catalog",
        lambda path: SimpleNamespace(
            concepts=[SimpleNamespace(concept_id=concept) for concept in carried]
        ),
    )


@pytest.mark.parametrize(
    ("carried", "exposure", "outcome"),
    [
        # The source publishes both named concepts only as versioned outputs.
        (("lact_v2", "death_v2"), "lact_v2", "death_v2"),
        # A source that carries the concept itself keeps that column.
        (("lact", "lact_v2", "death"), "lact", "death"),
        # Two outputs of one concept are ambiguous; the name stays as named.
        (("lact_v2", "lact_v3", "death"), "lact", "death"),
    ],
)
def test_named_coordinates_read_the_column_the_bound_source_publishes(
    monkeypatch: pytest.MonkeyPatch,
    carried: tuple[str, ...],
    exposure: str,
    outcome: str,
) -> None:
    """The planning schema must carry the column a coordinate names.

    Package materialization follows owner-declared composite outputs. A
    coordinate that keeps the public concept name leaves the requested
    exposure out of the zero-row planning schema, and every plan then fails
    the primary-result gate that requires that exact column.
    """

    _bound_source(monkeypatch, carried)

    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=_LACTATE_QUESTION, database="miiv", export_path="/bound/export"
    )

    assert coordinates["primary_exposure"] == exposure
    assert coordinates["target_outcome"] == outcome
    assert coordinates["endpoint"].name == outcome
    # The named operation still applies to the named concept's column.
    assert coordinates["primary_exposure_aggregation"] == "max"


def test_without_a_bound_source_the_coordinates_stay_as_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bound_source(monkeypatch, ("lact_v2", "death_v2"))

    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=_LACTATE_QUESTION, database="miiv"
    )

    assert (coordinates["primary_exposure"], coordinates["target_outcome"]) == (
        "lact",
        "death",
    )


def _association_plan(exposure_column: str):
    from easyicu.research_agent.contracts.model_terms import ModelTermSpec
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        PlannedModelRequirement,
    )

    requirement = PlannedModelRequirement(
        requirement_id="m1",
        analysis_role="primary",
        analysis_set="complete_case",
        required_for_step_success=True,
        exposure_source=exposure_column,
        outcome="death",
        outcome_type="binary",
        method_family="statsmodels_logit_mle",
        covariates=["age"],
        model_terms=[
            ModelTermSpec(
                name=exposure_column, role="exposure", coding="continuous",
                transform="identity",
            ),
            ModelTermSpec(
                name="age", role="covariate", coding="continuous", transform="identity"
            ),
        ],
    )
    step = AnalysisStep(
        step_id="06_primary",
        intent="Fit the primary adjusted association model.",
        method="adjusted_association_models",
        inputs=["table:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        planned_analysis_role="primary",
        model_requirements=[requirement],
    )
    return AnalysisPlan(
        research_question=_ASSOCIATION_QUESTION,
        analysis_type="association_study",
        steps=[step],
    )


_ASSOCIATION_QUESTION = "我想研究 ICU 患者的乳酸水平和院内死亡有没有关系。"


def test_planning_execution_and_the_primary_gate_read_one_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that publishes a concept under another column name.

    The planning coordinate, the package materializer's source concept and the
    column the primary-result gate requires are all that output column; a plan
    that keeps the public concept name is rejected.
    """

    from easyicu.research_agent.planning.primary_result_contract import (
        validate_required_primary_result,
    )
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
    )

    _bound_source(monkeypatch, ("lact_v2", "death", "age"))

    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=_ASSOCIATION_QUESTION, database="miiv", export_path="/bound/export"
    )
    column = coordinates["primary_exposure"]
    assert (column, coordinates["target_outcome"]) == ("lact_v2", "death")

    # Execution resolves both the planning column and the public name to it.
    by_id = {"lact_v2": object(), "death": object(), "age": object()}
    for name in (column, "lact"):
        assert (
            research_launch_scientific._source_concept_for_operational_column(
                name, by_id=by_id
            )
            == column
        )

    context = ResearchContext(
        research_question=_ASSOCIATION_QUESTION,
        cohort=CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=8, n_stays=8
        ),
        primary_exposure=column,
        target_outcome=coordinates["target_outcome"],
        variables=[
            ConceptDescriptor(name=column, description="lactate", dtype="float64"),
            ConceptDescriptor(name="death", description="death", dtype="int64"),
            ConceptDescriptor(name="age", description="age", dtype="float64"),
        ],
    )
    validate_required_primary_result(plan=_association_plan(column), context=context)
    with pytest.raises(ValueError, match="exact ResearchContext operational exposure"):
        validate_required_primary_result(plan=_association_plan("lact"), context=context)
