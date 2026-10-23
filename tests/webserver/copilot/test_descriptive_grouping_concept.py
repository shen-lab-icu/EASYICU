"""A looked-up stay grouping makes the named measurement a typed outcome.

All questions and metadata are synthetic; no source rows or provider calls.
"""

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.acquisition import catalog
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver import research_launch_scientific as owner
from easyicu.webserver import study_contexts

from .stated_variable_fixtures import _update


GROUP = "arrival_band"
WORDS = "收治带"
QUESTION = "请描述 eICU 合成队列中不同收治带患者入 ICU 首日最高血钾的分布。"


@pytest.fixture
def grouping_catalog(monkeypatch):
    from easyicu.concept import loader
    from easyicu.research_agent.research_context import typed

    definitions = dict(loader.load_dictionary())
    definitions[GROUP] = SimpleNamespace(class_name="fct_cncpt", target="id_tbl")
    monkeypatch.setattr(loader, "load_dictionary", lambda: definitions)
    original = catalog.build_database_capability_catalog
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda database: catalog.AvailableCatalog(
        source=database, concepts=[*original(database).concepts, catalog.CatalogConcept(concept_id=GROUP)],
    ))
    monkeypatch.setitem(typed._CONCEPT_FACTOR_CACHE, GROUP, True)


def _coordinates(*, question=QUESTION, **kwargs):
    return owner._metadata_only_planning_coordinates(
        question=question, database="eicu", stated_exposure=WORDS, **kwargs,
    )


def _context(coordinates, *, question=QUESTION, measurement="k_max"):
    endpoint = coordinates["endpoint"]
    built = build_research_context(
        research_question=question, cohort=pd.DataFrame({
            "stay_id": pd.Series(dtype="int64"), GROUP: pd.Series(dtype="float64"),
            measurement: pd.Series(dtype="float64"),
        }), cohort_name="synthetic", database="eicu", id_columns=["stay_id"],
        primary_exposure=coordinates["primary_exposure"], target_outcome=endpoint.name if endpoint else None,
        endpoint=endpoint,
    )
    context = getattr(built, "context", built)
    return context.model_copy(update={
        "cohort": context.cohort.model_copy(update={
            "provenance": {"evidence_stage": "metadata_only_planning", "patient_rows_read": False},
        }),
        "variables": [item.model_copy(update={"source_concept": GROUP}) if item.name == GROUP else item
                      for item in context.variables],
    })


def _compile(context, measurement="k_max"):
    skeleton = ProgressivePlanSkeleton.model_validate({
        "analysis_type": "descriptive_epidemiology", "rationale": "Describe a synthetic grouped measurement.",
        "cohort": {"name": "primary", "selection_mode": "all_input_rows"},
        "display_labels": [{"key": GROUP, "value": "Arrival band"}, {"key": measurement, "value": "Measurement"}],
        "steps": [
            {"step_id": "cohort", "module_id": "cohort_definition", "planned_analysis_role": "auxiliary",
             "objective": "Account for the stay denominator."},
            {"step_id": "overall", "module_id": "custom_analysis", "planned_analysis_role": "primary",
             "objective": "Describe the overall measurement distribution.",
             "scientific_action_id": "descriptive.descriptive_summary", "custom_method": "empirical_summary",
             "depends_on": ["cohort"], "raw_inputs": [measurement],
             "outputs": [{"product_id": "table:cohort_summary", "semantic_role": "custom"},
                         {"product_id": "table:distribution_summary", "semantic_role": "custom"}]},
            {"step_id": "groups", "module_id": "continuous_outcome_summary", "planned_analysis_role": "secondary",
             "objective": "Describe the maximum within every arrival band.", "depends_on": ["cohort"],
             "primary_exposure": GROUP, "raw_inputs": [GROUP, measurement]},
        ],
    })
    return compile_progressive_plan(skeleton=skeleton, context=context)[0]


@pytest.mark.parametrize("concept,words", [("k", "血钾"), ("hgb", "血红蛋白")])
def test_a_windowed_measurement_compiles_as_a_typed_grouped_outcome(grouping_catalog, concept, words):
    question = QUESTION.replace("血钾", words)
    column = f"{concept}_max"
    coordinates = _coordinates(question=question, stated_exposure_concept=GROUP)
    context = _context(coordinates, question=question, measurement=column)
    plan = _compile(context, column)
    assert plan.steps[-1].inputs == ["artifact:analysis_cohort", GROUP, column]
    assert context.variable(column).role.value == "outcome"
    assert coordinates["primary_exposure"] == GROUP
    assert coordinates["target_outcome"] == concept
    assert coordinates["target_outcome_aggregation"] == "max"
    assert context.endpoint.kind == "continuous" and context.endpoint.name == column
    outcome = next(row for row in coordinates["coordinate_readings"] if row["role"] == "outcome")
    assert outcome["evidence"] in question and words in outcome["evidence"]
    assert coordinates["execution_authorized"] is False


def test_words_only_preserve_the_original_compile_refusal(grouping_catalog):
    coordinates = _coordinates()
    assert coordinates["primary_exposure"] is coordinates["target_outcome"] is coordinates["endpoint"] is None
    context = _context(coordinates)
    assert context.variable("k_max").role.value == "lab"
    with pytest.raises(ValueError, match="progressive_continuous_outcome_invalid.*raw_inputs"):
        _compile(context)


def test_a_whole_stay_quantity_compiles_with_the_same_grouping_reading(grouping_catalog):
    question = "请描述 eICU 合成队列中不同收治带患者的 ICU 住院时长分布。"
    coordinates = _coordinates(question=question, stated_exposure_concept=GROUP)
    assert coordinates["target_outcome"] == "los_icu"
    assert coordinates["endpoint"].kind == "continuous"
    context = _context(coordinates, question=question, measurement="los_icu")
    assert _compile(context, "los_icu").steps[-1].inputs == ["artifact:analysis_cohort", GROUP, "los_icu"]


def test_a_binary_measure_retains_its_owner_endpoint(grouping_catalog):
    coordinates = _coordinates(question="请描述 eICU 合成队列中不同收治带患者的院内死亡比例。",
                               stated_exposure_concept=GROUP)
    assert coordinates["primary_exposure"] == GROUP
    assert coordinates["target_outcome"] == "death"
    assert coordinates["endpoint"].kind == "binary" and coordinates["endpoint"].levels == [0, 1]


@pytest.mark.parametrize("kind,target", [("num_cncpt", "id_tbl"), ("fct_cncpt", "ts_tbl")])
def test_only_an_owner_defined_stay_grouping_can_replace_the_keyword_slot(grouping_catalog, monkeypatch, kind, target):
    from easyicu.concept.loader import load_dictionary

    monkeypatch.setitem(load_dictionary(), GROUP, SimpleNamespace(class_name=kind, target=target))
    assert _coordinates(stated_exposure_concept=GROUP) == _coordinates()


def test_a_measurements_checked_identifier_does_not_make_it_a_grouping(grouping_catalog):
    coordinates = owner._metadata_only_planning_coordinates(
        question=QUESTION, database="eicu", stated_exposure="血钾", stated_exposure_concept="k",
    )
    assert coordinates["primary_exposure"] is None
    assert coordinates["endpoint"] is None


def test_an_event_status_is_also_an_owner_defined_grouping(grouping_catalog, monkeypatch):
    original = catalog.build_database_capability_catalog
    def events(database):
        result = original(database)
        next(item for item in result.concepts if item.concept_id == GROUP).column_role = "event_status"
        return result
    monkeypatch.setattr(catalog, "build_database_capability_catalog", events)
    from easyicu.concept.loader import load_dictionary
    monkeypatch.setitem(load_dictionary(), GROUP, SimpleNamespace(class_name="lgl_cncpt", target="ts_tbl"))
    assert _coordinates(stated_exposure_concept=GROUP)["primary_exposure"] == GROUP


@pytest.mark.parametrize("question", [
    "在 eICU 合成队列中，收治带与入 ICU 首日最高血钾是否相关？",
    "在 eICU 合成队列中，用收治带和入 ICU 首日最高血钾预测院内死亡。",
])
def test_other_families_keep_their_coordinate_and_conflict_readings(grouping_catalog, question):
    assert _coordinates(question=question, stated_exposure_concept=GROUP) == _coordinates(question=question)


@pytest.mark.parametrize("question,concept", [
    (QUESTION.replace("eICU ", ""), GROUP), (QUESTION, "not_in_database"),
    (QUESTION.replace(WORDS, "未命名组"), GROUP),
])
def test_an_unchecked_concept_never_changes_the_planning_projection(grouping_catalog, question, concept):
    assert _coordinates(question=question, stated_exposure_concept=concept) == _coordinates(question=question)


def test_a_bound_source_checks_a_group_even_without_a_named_database(grouping_catalog, monkeypatch):
    monkeypatch.setattr(catalog, "build_available_catalog", lambda path: catalog.AvailableCatalog(
        source="synthetic", concepts=[catalog.CatalogConcept(concept_id=GROUP), catalog.CatalogConcept(concept_id="k")],
    ))
    coordinates = _coordinates(question=QUESTION.replace("eICU ", ""), export_path="/synthetic/source",
                               stated_exposure_concept=GROUP)
    assert coordinates["primary_exposure"] == GROUP


def _save(monkeypatch, *, concept=GROUP, question=QUESTION, source=None):
    current = {"id": "synthetic-group-study", "revision": 1, "question": question, "primary_exposure": WORDS}
    if source is not None:
        current["data_source"] = source
    return _update(monkeypatch, current=current, question=question, field="primary_exposure_concept",
                   statements=concept, concepts=(GROUP, "k"))


@pytest.mark.parametrize("bound", [False, True])
def test_the_save_checks_and_projects_the_optional_concept(grouping_catalog, monkeypatch, bound):
    source = {"path": "/synthetic/source", "database": "eicu"} if bound else None
    result, writes = _save(monkeypatch, source=source)
    assert result["status"] == "ok"
    assert writes[0]["primary_exposure_concept"] == GROUP
    assert result["details"]["study"]["primary_exposure_concept"] == GROUP


@pytest.mark.parametrize("question,concept", [
    (QUESTION.replace("eICU ", ""), GROUP), (QUESTION, "not_in_database"),
    (QUESTION.replace(WORDS, "未命名组"), GROUP),
])
def test_an_unchecked_save_preserves_the_question_and_refuses_the_identifier(grouping_catalog, monkeypatch, question, concept):
    result, writes = _save(monkeypatch, question=question, concept=concept)
    assert result["status"] == "ok" and writes[0]["question"] == question
    assert "primary_exposure_concept" not in writes[0]
    assert result["details"]["primary_exposure_concept_unread"]


def test_the_store_protects_and_invalidates_the_checked_reading(grouping_catalog, tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot.projections import project_study_setup_receipt

    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    raw = {"id": "stored-group-study", "question": QUESTION, "primary_exposure": WORDS}
    original = study_contexts.upsert_context(raw)
    with pytest.raises(study_contexts.StudyContextError, match="server_owned"):
        study_contexts.upsert_context({**raw, "primary_exposure_concept": GROUP})
    saved = study_contexts.upsert_context({**raw, "primary_exposure_concept": GROUP},
                                         _server_question_wording_write=True)
    assert study_contexts.get_context(saved["id"])["primary_exposure_concept"] == GROUP
    assert project_study_setup_receipt(saved).configuration["primary_exposure_concept"] == GROUP
    binding = study_contexts.build_agent_context_binding(
        {**saved, "data_source": {"path": str(tmp_path)}}, export_path=str(tmp_path),
    )
    assert binding["informational"]["primary_exposure_concept"] == GROUP
    assert study_contexts.scientific_configuration_sha256(saved) != study_contexts.scientific_configuration_sha256(original)
    changed = study_contexts.upsert_context({"id": saved["id"], "primary_exposure": "血钾"})
    assert changed["primary_exposure_concept"] is None
    restored = study_contexts.restore_turn_configuration_snapshot(
        saved["id"], {"schema_version": "easyicu.pi-turn-study-snapshot/1", "study_context_id": saved["id"],
                      "source_revision": saved["revision"], "configuration": {
                          "question": QUESTION, "primary_exposure": WORDS, "primary_exposure_concept": GROUP,
                      }}, expected_revision=changed["revision"],
    )
    assert restored["primary_exposure_concept"] == GROUP
    changed_question = study_contexts.upsert_context({"id": saved["id"], "question": "只描述合成测量。"})
    assert changed_question["primary_exposure_concept"] is None


def test_the_question_words_and_concept_can_be_saved_together(grouping_catalog, monkeypatch, tmp_path):
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, ToolExecutionContext
    from easyicu.webserver.pi_copilot.study_context_update import update_study_context

    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    current = study_contexts.upsert_context({"id": "group-study", "question": QUESTION})
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="group-session", binding=AuthorityBinding(
            study_context_id=current["id"], study_revision=current["revision"],
        )), user_message=QUESTION, allowed_actions={"configure"},
    )
    result = update_study_context(context, {
        "question": QUESTION, "primary_exposure": WORDS, "primary_exposure_concept": GROUP,
    }, load_context=lambda _: current, project_workflow=lambda *args, **kwargs: {})
    assert result["status"] == "ok"
    saved = study_contexts.get_context(current["id"])
    assert saved["primary_exposure"] == WORDS and saved["primary_exposure_concept"] == GROUP


def test_a_changed_source_rechecks_the_saved_concept(grouping_catalog, monkeypatch):
    monkeypatch.setattr(catalog, "build_available_catalog", lambda path: catalog.AvailableCatalog(
        source="synthetic", concepts=[catalog.CatalogConcept(concept_id="k")],
    ))
    with pytest.raises(ValueError, match="bound source"):
        owner.checked_exposure_concept(WORDS, GROUP, question=QUESTION,
                                       source={"path": "/synthetic/changed"})
    assert _coordinates(export_path="/synthetic/changed", stated_exposure_concept=GROUP) == _coordinates(
        export_path="/synthetic/changed",
    )


def test_a_looked_up_identifier_cannot_contradict_known_words(grouping_catalog):
    with pytest.raises(ValueError, match="dictionary reads"):
        owner.checked_exposure_concept("血钾", GROUP, question=QUESTION, source={}, database="eicu")


@pytest.mark.parametrize("question", [
    "请描述 eICU 合成队列中收治带的分布。",
    "请描述 eICU 合成队列中不同收治带患者血钾的分布。",
    "请描述 eICU 合成队列中不同收治带患者入 ICU 首日最高血钾和首日最高血钠的分布。",
])
def test_a_grouping_does_not_invent_a_measure_or_its_summary(grouping_catalog, question):
    coordinates = _coordinates(question=question, stated_exposure_concept=GROUP)
    assert coordinates["endpoint"] is None


@pytest.mark.parametrize("configured", [False, True])
def test_the_launch_carries_the_measurement_through_the_zero_row_schema(grouping_catalog, monkeypatch, tmp_path, configured):
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.webserver import agent_pipeline_runs, research_pipeline_run_preparation as preparation

    menu = catalog.build_database_capability_catalog("eicu")
    monkeypatch.setattr(catalog, "build_available_catalog", lambda path: menu)
    study = {"id": "group-study", "question": QUESTION, "primary_exposure": WORDS,
             "primary_exposure_concept": GROUP, "data_source": {"database": "eicu"}}
    if configured:
        study["execution_concepts"] = {"outcome": "k"}
    original = json.dumps(study, sort_keys=True)
    request = preparation.ResearchPipelineLaunchRequest(
        export_path=str(tmp_path), study_context=study, project_root=str(tmp_path), provider={},
        provider_environment={}, credential_source="synthetic", literature_search_authorized=False,
        plan_revision_source_run_id="", execution_resume_source_run_id="", development_resume_source_job_id="",
        budget_mode="planner_canary", runner_image=None,
    )
    prepared = preparation._prepare_scientific_launch(request)
    assert prepared.primary_exposure == GROUP and prepared.target == "k"
    assert "k_max" in prepared.metadata_operationalized_columns
    assert prepared.cohort_window == (0.0, 24.0)
    endpoint = prepared.metadata_planning_coordinates["endpoint"]
    acquisition = agent_pipeline_runs._metadata_only_planning_acquisition(
        database="eicu", export_path=tmp_path, question=QUESTION, output_dir=tmp_path / "catalog",
        target_outcome=prepared.target, endpoint=endpoint, required_concepts=(GROUP, "k"),
        operationalized_columns=prepared.metadata_operationalized_columns,
        llm=ScriptedMockLLMClient([json.dumps({"selected_concepts": [GROUP, "k"]})]),
    )
    # Exercise the runner's actual binding expression without starting a run.
    tree = ast.parse(Path(agent_pipeline_runs.__file__).read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "metadata_only_planning and execution_resume_inputs is None"
                 and any(isinstance(child, ast.Assign) and ast.unparse(child.targets[0]) == "pipeline_target"
                         for child in node.body))
    expression = next(child.value for child in block.body if isinstance(child, ast.Assign)
                      and ast.unparse(child.targets[0]) == "pipeline_target")
    target = eval(compile(ast.Expression(expression), "<runner-target>", "eval"), {  # noqa: S307 -- trusted owner code
        **vars(agent_pipeline_runs), "execution_concepts": study.get("execution_concepts", {}),
        "metadata_planning_coordinates": prepared.metadata_planning_coordinates,
    })
    different = eval(compile(ast.Expression(expression), "<runner-target>", "eval"), {  # noqa: S307 -- trusted owner code
        **vars(agent_pipeline_runs), "execution_concepts": {"outcome": "different_outcome"},
        "metadata_planning_coordinates": prepared.metadata_planning_coordinates,
    })
    assert different == "different_outcome"
    context = build_research_context(
        research_question=QUESTION, cohort=acquisition.universe_path, cohort_name="synthetic", database="eicu",
        target_outcome=target, primary_exposure=prepared.primary_exposure, endpoint=acquisition.endpoint,
    )
    context = context.model_copy(update={"variables": [
        item.model_copy(update={"source_concept": GROUP}) if item.name == GROUP else item
        for item in context.variables
    ]})
    assert context.cohort.n_stays == 0
    assert context.variable("k_max").role.value == "outcome"
    assert _compile(context).steps[-1].inputs == ["artifact:analysis_cohort", GROUP, "k_max"]
    assert json.dumps(study, sort_keys=True) == original
    # The same request without the checked reading still cannot bind the grouping.
    without = preparation._prepare_scientific_launch(replace(request, study_context={
        key: value for key, value in study.items() if key != "primary_exposure_concept"
    }))
    assert without.primary_exposure is None
    if not configured:
        assert without.metadata_planning_coordinates["endpoint"] is None


def test_endpoint_choices_use_the_checked_grouping_and_its_bound_source(grouping_catalog, monkeypatch):
    from easyicu.webserver.pi_copilot.plan_decisions import endpoint_decision_options

    monkeypatch.setattr(catalog, "build_available_catalog", lambda path: catalog.build_database_capability_catalog("eicu"))
    study = {"question": QUESTION.replace("eICU ", ""), "primary_exposure": WORDS,
             "primary_exposure_concept": GROUP, "data_source": {"database": "eicu", "path": "/synthetic/source"}}
    assert [row["concept"] for row in endpoint_decision_options(study)] == ["k"]


@pytest.mark.parametrize("invalid", [None, "changed_summary", "missing_source", "missing_requested_outcome"])
def test_a_reviewed_aggregate_binds_its_source_without_relaxing_coverage(grouping_catalog, monkeypatch, tmp_path, invalid):
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview
    from easyicu.webserver import agent_pipeline_runs, agent_runs

    study = {"id": "group-study", "question": QUESTION, "primary_exposure": WORDS,
             "primary_exposure_concept": GROUP, "data_source": {"database": "eicu"}}
    project = tmp_path / "candidate"
    inner = project / "pipeline" / "group-candidate"
    inner.mkdir(parents=True)
    capsule = json.dumps({"scientific_identity": {
        "question": QUESTION, "database": "eicu", "primary_exposure": GROUP,
        "target_outcome": "k_min" if invalid == "changed_summary" else "k_max", "user_preferences": {},
    }}).encode()
    (inner / "run_input_capsule.json").write_bytes(capsule)
    (inner / "human_review_checkpoint.json").write_text(json.dumps({
        "state": "pending", "run_input_capsule_sha256": hashlib.sha256(capsule).hexdigest(),
    }))
    inputs = project / "pipeline_input"
    inputs.mkdir()
    selected = [GROUP] if invalid == "missing_source" else [GROUP, "k"]
    pd.DataFrame(columns=[*selected, "k_max", "k_min"]).to_parquet(inputs / "planner_catalog.parquet")
    (inputs / "planner_catalog_receipt.json").write_text(json.dumps({
        "selected_concepts": selected, "operationalized_columns": ["k_max", "k_min"],
    }))
    requested = ["k_max", "death"] if invalid == "missing_requested_outcome" else ["k_max"]
    review = PlanScientificReview(
        status="analysis_only", approval_allowed=True, top_journal_candidate=False, score=90,
        dimension_scores={"study_design": 90}, findings=[], facts={"requested_outcomes": requested},
        context_sha256="a" * 64, plan_sha256="b" * 64, literature_sha256="c" * 64,
        figure_strategy_sha256="d" * 64, generated_at="2026-10-23T00:00:00Z",
    )
    plan = _compile(_context(_coordinates(stated_exposure_concept=GROUP)))
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **kwargs: {"runs": [{
        "run_id": "group-candidate", "project_dir": str(project),
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
    }]})
    monkeypatch.setattr(agent_runs, "read_run_review", lambda path: {"ok": True, "artifact_payloads": {
        "scientific_plan_review.json": review.model_dump(mode="json"), "agent_plan.json": plan.model_dump(mode="json"),
    }})

    def load():
        return agent_pipeline_runs._load_candidate_plan_materialization_authority(
            study=study, project_root=str(tmp_path), source_run_id="group-candidate", database="eicu", covariates=(),
        )

    if invalid:
        with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError, match="does not match"):
            load()
    else:
        authority = load()
        assert authority.primary_exposure == GROUP
        assert authority.target_outcome == "k" and authority.outcome_concepts == ("k",)


def test_the_failed_compile_explains_the_unbound_coordinates_in_chinese(grouping_catalog):
    from easyicu.webserver import agent_pipeline_runs
    from tests.support.node import run_node

    with pytest.raises(ValueError) as caught:
        _compile(_context(_coordinates()))
    assert "categorical grouping and a numeric measure" in agent_pipeline_runs._progressive_compile_failure_message(caught.value)
    typed = agent_pipeline_runs._safe_pipeline_typed_failure(caught.value)
    node = shutil.which("node")
    assert node is not None
    script = """
let errorText;
global.window = {EU_LANG:'zh', EU_HTML:{esc: String},
                 EasyICU:{guidedPi:{declare: (name, api) => {errorText = api;}}}};
require(process.argv[1]);
const owner = errorText.create({tr: (en, zh) => zh, staticPreview: () => false});
process.stdout.write(owner.runFailureText('research_pipeline_progressive_compile_failed', {code: process.argv[2]}));
"""
    path = Path(agent_pipeline_runs.__file__).parent / "static/js/screens-guided-pi-error-text.js"
    text = run_node(node, script, str(path), typed["reason_code"], check=True).stdout
    assert all(words in text for words in ("分组", "核验", "数值指标", "汇总哪个指标", "没有运行分析"))
    assert "progressive_continuous_outcome_invalid" not in text


def test_the_conversation_looks_up_and_saves_the_exposure_reading():
    main = (Path(owner.__file__).parent / "pi_copilot/node_app/src/main.mjs").read_text()
    rule = next(line for line in main.splitlines() if '"Question wording rule:' in line)
    assert "look up the exposure or grouping with easyicu_list_source_concepts" in rule
    assert "named database or bound source" in rule
    assert "save its exact identifier in primary_exposure_concept" in rule
