"""Synthetic owner definitions, never patient data or hosted question text."""

from types import SimpleNamespace

import pytest

from easyicu.concept.schema import ConceptDefinition
from easyicu.research_agent.acquisition import catalog
from easyicu.webserver import research_launch_scientific as launch
from easyicu.webserver.pi_copilot import plan_decisions


@pytest.fixture
def owners(monkeypatch):
    from easyicu.concept import loader
    from easyicu.concept import catalog as concept_catalog
    from easyicu.webserver import study_intent

    definitions = {
        "recovery": ConceptDefinition.from_name_and_payload("recovery", {
            "target": "id_tbl", "unit": "hours", "description": "Whole episode recovery duration",
        }),
        "analyte": ConceptDefinition.from_name_and_payload("analyte", {
            "unit": "mmol/L", "description": "Measured analyte concentration",
        }),
        "ward_group": ConceptDefinition.from_name_and_payload("ward_group", {
            "class": "fct_cncpt", "target": "id_tbl", "description": "Admission ward category",
        }),
        "undefined": ConceptDefinition.from_name_and_payload("undefined", {}),
    }
    monkeypatch.setattr(loader, "load_dictionary", lambda: definitions)
    monkeypatch.setitem(concept_catalog.CONCEPT_GROUPS_INTERNAL, "outcome", ["event", "recovery"])
    menu = catalog.AvailableCatalog(source="synthetic", concepts=[
        catalog.CatalogConcept(concept_id=name, description=definition.description or "")
        for name, definition in definitions.items()
    ] + [catalog.CatalogConcept(concept_id="event", column_role="event_status")])
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda db: menu)
    monkeypatch.setattr(study_intent, "deterministic_intent", lambda question: {"slots": {
        "analysis_family": {"value": "description", "provenance": "keyword"},
        "exposure": {"value": "ward_group", "provenance": "keyword", "evidence": "ward_group"},
    }})
    monkeypatch.setattr(study_intent, "named_study_concepts", lambda q: [(('ward_group',), 'ward_group')])
    monkeypatch.setattr(study_intent, "catalog_named_concepts", lambda q: ["ward_group"])
    monkeypatch.setattr(study_intent, "explicit_exposure_aggregation", lambda q, concept_id:
                        SimpleNamespace(aggregation="max") if concept_id == "analyte" else None)
    return definitions


def test_question_named_measurement_becomes_the_grouped_continuous_endpoint(owners, monkeypatch):
    from easyicu.webserver import study_intent
    monkeypatch.setattr(study_intent, "named_study_concepts", lambda q:
                        [(('ward_group',), 'ward_group'), (('analyte',), 'analyte')])
    monkeypatch.setattr(study_intent, "catalog_named_concepts", lambda q: ["ward_group", "analyte"])
    result = launch._metadata_only_planning_coordinates(
        question="Describe maximum analyte in the first 24 hours by ward_group.", database="synthetic", stated_exposure="ward_group",
    )
    assert result["target_outcome"] == "analyte"
    assert result["endpoint"].kind == "continuous"


def test_multiple_named_quantities_do_not_choose_a_primary_outcome(owners, monkeypatch):
    from easyicu.webserver import study_intent
    monkeypatch.setattr(study_intent, "catalog_named_concepts", lambda q: ["ward_group", "analyte", "recovery"])
    result = launch._metadata_only_planning_coordinates(
        question="Describe analyte and recovery by ward_group.", database="synthetic", stated_exposure="ward_group",
    )
    assert result["target_outcome"] is None
    assert result["endpoint"] is None


def test_one_categorical_measurement_is_not_a_grouped_outcome_question(owners):
    result = launch._metadata_only_planning_coordinates(
        question="Describe the synthetic ward_group distribution.", database="synthetic",
    )
    assert result["primary_exposure"] is None
    assert result["target_outcome"] is None
    assert any(row["role"] == "measurement" for row in result["coordinate_readings"])


def test_unselected_measurement_summary_does_not_acquire_an_endpoint(owners, monkeypatch):
    from easyicu.webserver import study_intent
    monkeypatch.setattr(study_intent, "explicit_exposure_aggregation", lambda *args, **kwargs: None)
    with pytest.raises(launch.ResearchPipelineRunError) as error:
        launch._metadata_only_planning_coordinates(
            question="Describe analyte.", database="synthetic", configured_outcome="analyte",
        )
    assert error.value.code == "windowed_summary_statistic_unresolved"


@pytest.mark.parametrize("role", ["count", "measurement_status", "event_time"])
def test_measurement_metadata_cannot_stand_in_for_a_continuous_value(owners, role):
    assert catalog.catalog_endpoint(catalog.CatalogConcept(concept_id="analyte", column_role=role),
                                    column="measured_column", aggregation="max") is None


def test_an_unknown_requested_concept_does_not_offer_events(owners):
    study = {"question": "Describe an undefined synthetic quantity.", "data_source": {"database": "synthetic"},
             "execution_concepts": {"outcome": "unregistered_quantity"}}
    assert plan_decisions.endpoint_decision_options(study) == []
    assert plan_decisions.plan_decision_context({}, "OUTCOME_DEFINITION_UNRESOLVED", study)["endpoint_definition_unavailable"]


@pytest.mark.parametrize("outcome,aggregation", [("recovery", None), ("analyte", "max")])
def test_defined_continuous_outcome_and_stay_group_reach_planning(owners, outcome, aggregation):
    coordinates = launch._metadata_only_planning_coordinates(
        question="Describe the synthetic measurement in the first 24 hours by ward_group.", database="synthetic",
        configured_outcome=outcome, stated_exposure="ward_group",
    )
    assert coordinates["endpoint"] is not None
    assert coordinates["endpoint"].kind == "continuous"
    assert coordinates["endpoint"].levels is None
    assert coordinates["endpoint"].absence_semantics == "absent_row_is_unmeasured"
    assert coordinates["target_outcome_aggregation"] == aggregation
    assert coordinates["primary_exposure"] == "ward_group"
    assert any(row["role"] == "exposure" and row["concept"] == "ward_group"
               for row in coordinates["coordinate_readings"])
    assert coordinates["execution_authorized"] is False


def test_undefined_outcome_stops_without_unrelated_endpoint_choices(owners):
    study = {"question": "Describe undefined by ward_group.", "data_source": {"database": "synthetic"},
             "execution_concepts": {"outcome": "undefined"}}
    coordinates = launch._metadata_only_planning_coordinates(
        question=study["question"], database="synthetic", configured_outcome="undefined",
    )
    assert coordinates["endpoint"] is None
    context = plan_decisions.plan_decision_context({}, "OUTCOME_DEFINITION_UNRESOLVED", study)
    assert not context.get("endpoint_options")
    assert context["endpoint_definition_unavailable"] is True


def test_defined_requested_outcome_is_the_endpoint_choice(owners):
    study = {"question": "Describe recovery by ward_group.", "data_source": {"database": "synthetic"},
             "execution_concepts": {"outcome": "recovery"}}
    assert [row["concept"] for row in plan_decisions.endpoint_decision_options(study)] == ["recovery"]


def test_binary_descriptive_coordinates_keep_event_semantics(owners):
    result = launch._metadata_only_planning_coordinates(
        question="Describe event rates by ward_group.", database="synthetic",
        configured_outcome="event", stated_exposure="ward_group",
    )
    assert result["endpoint"].kind == "binary"
    assert result["endpoint"].levels == [0, 1]
    assert result["primary_exposure"] == "ward_group"


@pytest.mark.parametrize("change", [
    {"description": None}, {"units": None}, {"class_name": "fct_cncpt"}, {"levels": [1, 2]},
])
def test_numeric_storage_is_not_an_endpoint_definition(owners, change):
    for field, value in change.items():
        setattr(owners["recovery"], field, value)
    result = launch._metadata_only_planning_coordinates(
        question="Describe recovery.", database="synthetic", configured_outcome="recovery",
    )
    assert result["endpoint"] is None


@pytest.mark.parametrize("outcome,aggregation,column", [
    ("recovery", None, "recovery"), ("analyte", "max", "analyte_max"),
])
def test_materialization_preserves_the_selected_continuous_endpoint(owners, monkeypatch, tmp_path, outcome, aggregation, column):
    import json
    import pandas as pd
    from easyicu.research_agent.acquisition import foundation
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    menu = catalog.build_database_capability_catalog("synthetic")
    monkeypatch.setattr(foundation, "build_available_catalog", lambda path: menu)
    frame = pd.DataFrame({"stay_id": [1, 2, 3, 4], column: [1., 2., 3., 4.]})
    parquet = tmp_path / "synthetic.parquet"
    frame.to_parquet(parquet)
    provenance = tmp_path / "synthetic.json"
    provenance.write_text(json.dumps({"columns": list(frame.columns)}))
    monkeypatch.setattr(materializer, "materialize_to_parquet", lambda **kwargs:
                        {"parquet": parquet, "provenance": provenance})
    result = foundation.acquire_universe_for_question(
        export_dir=tmp_path, output_dir=tmp_path, question="Describe a synthetic duration by ward.",
        llm=ScriptedMockLLMClient([json.dumps({"selected_concepts": [outcome]})]), target_outcome=outcome,
        target_outcome_aggregation=aggregation, outcome_concepts=[], require_outcome=False,
        static_concepts=[outcome] if aggregation is None else [],
        required_feature_concepts=[outcome] if aggregation else [], emit_trajectory=False,
    )
    assert result.endpoint.kind == "continuous"
    assert result.endpoint.name == column
    assert result.analysis_columns[outcome] == column


def test_web_acquisition_receives_the_selected_outcome_operation():
    import ast
    from pathlib import Path
    from easyicu.webserver import agent_pipeline_runs

    tree = ast.parse(Path(agent_pipeline_runs.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "acquire_universe_for_question"
             and any(kw.arg == "target_outcome" for kw in node.keywords)]
    assert calls
    for call in calls:
        operation = next(kw.value for kw in call.keywords if kw.arg == "target_outcome_aggregation")
        assert ast.unparse(operation) == "metadata_planning_coordinates.get('target_outcome_aggregation')"
