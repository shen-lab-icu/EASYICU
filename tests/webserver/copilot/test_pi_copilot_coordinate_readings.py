"""A keyword-selected coordinate is visible before the plan is approved."""
from __future__ import annotations

import pytest

from easyicu.webserver import research_launch_scientific


@pytest.mark.parametrize("question", [
    "Association of first-day maximum lactate with in-hospital mortality",
    "评估入 ICU 首日最高血乳酸与住院死亡之间的关联",
])
def test_candidate_coordinates_keep_their_keyword_source(question):
    result = research_launch_scientific._metadata_only_planning_coordinates(
        question=question, database="miiv",
    )
    assert {row["role"] for row in result["coordinate_readings"]} == {"outcome", "exposure"}
    assert all(row["basis"] == "keyword" and row["evidence"] for row in result["coordinate_readings"])
    assert result["execution_authorized"] is False


def test_configured_outcome_is_not_relabelled_as_keyword_authority():
    result = research_launch_scientific._metadata_only_planning_coordinates(
        question="Association of lactate with in-hospital mortality",
        database="miiv", configured_outcome="death",
    )
    row = next(row for row in result["coordinate_readings"] if row["role"] == "outcome")
    assert row["basis"] == "configured" and row["evidence"] is None


def test_prediction_does_not_invent_an_exposure_reading():
    result = research_launch_scientific._metadata_only_planning_coordinates(
        question="Predict mortality using lactate and compare with APACHE IVa",
        database="eicu",
    )
    assert not any(row["role"] == "exposure" for row in result["coordinate_readings"])


@pytest.mark.parametrize("database,concept,question", [
    ("eicu", "age", "描述一个合成 ICU 队列中年龄的分布。"),
    ("miiv", "lact", "描述一个合成 MIMIC-IV 队列中首日血乳酸的分布。"),
    ("eicu", "age", "Describe the age distribution in a synthetic ICU cohort."),
    ("miiv", "lact", "Describe first-day blood lactate in a synthetic MIMIC-IV cohort."),
    ("miiv", "lact", "What proportion of synthetic ICU stays have high blood lactate?"),
])
def test_described_measurement_is_a_keyword_hint_without_an_exposure(database, concept, question):
    result = research_launch_scientific._metadata_only_planning_coordinates(
        question=question, database=database,
    )
    assert result["primary_exposure"] is None
    assert result["target_outcome"] is None
    assert result["primary_exposure_aggregation"] is None
    assert not any(row["role"] == "exposure" for row in result["coordinate_readings"])
    hint = next(row for row in result["coordinate_readings"] if row["role"] == "measurement")
    assert hint["concept"] == concept and hint["basis"] == "keyword"
    assert hint["evidence"].casefold() in question.casefold()
    assert any(concept in row["concepts"] and row["evidence"]
               for row in result["question_named_concepts"])
    assert result["source"] == "question_keywords_plus_database_capability"
    assert result["execution_authorized"] is False


@pytest.mark.parametrize("question", [
    "评估合成 ICU 队列中年龄与住院死亡的关系。",
    "Study the association between age and hospital mortality in a synthetic ICU cohort.",
    "Describe the cohort and estimate the association between age and hospital mortality.",
])
def test_association_keeps_its_keyword_exposure(question):
    result = research_launch_scientific._metadata_only_planning_coordinates(
        question=question, database="eicu",
    )
    assert result["primary_exposure"] == "age"
    assert any(row["role"] == "exposure" and row["basis"] == "keyword"
               for row in result["coordinate_readings"])


def test_keyword_receipt_quotes_the_original_case_and_spacing():
    question = "Is LACTATE associated with HOSPITAL   DEATH?"
    rows = research_launch_scientific._metadata_only_planning_coordinates(
        question=question, database="miiv",
    )["coordinate_readings"]
    assert {row["role"]: row["evidence"] for row in rows} == {
        "outcome": "DEATH", "exposure": "LACTATE",
    }
    assert all(row["evidence"] in question for row in rows)


def _proposed():
    return [{"role": "exposure", "concept": "lact", "column": "lact",
             "category": "blood gas", "basis": "keyword", "evidence": "lactate"}]


def test_measurement_hint_is_recorded_without_becoming_a_coordinate(tmp_path):
    import json
    from easyicu.webserver import planning_coordinate_readings as owner

    hint = {**_proposed()[0], "role": "measurement"}
    rows = owner.context_readings([hint], columns={}, configured={})
    assert rows == []
    owner.write_planning_coordinate_readings(tmp_path, rows, hints=[hint])
    payload = json.loads((tmp_path / owner.PLANNING_COORDINATE_READINGS_FILENAME).read_text())
    assert payload["hints"] == [hint]
    assert payload["readings"] == []
    assert owner.load_planning_coordinate_readings(tmp_path, context_columns={}) == []


def test_launch_receipt_wiring_preserves_the_measurement_hint(tmp_path):
    """Exercise the launch's receipt call without starting a pipeline or reading data."""
    import ast
    import json
    from pathlib import Path
    from easyicu.webserver import agent_pipeline_runs, planning_coordinate_readings

    tree = ast.parse(Path(agent_pipeline_runs.__file__).read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_planning_coordinate_readings"
    ]
    assert len(calls) == 1
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question="Describe the age distribution of a synthetic ICU cohort.", database="eicu",
    )
    namespace = {
        "planning_coordinate_readings": planning_coordinate_readings,
        "wrapper_dir": tmp_path, "metadata_planning_coordinates": coordinates,
        "pipeline_target": None, "resolved_primary_exposure": None,
        "execution_concepts": {}, "configured_primary_exposure": None,
        "_target_outcome": lambda study: None, "study": {},
    }
    # Execute only the repository-owned call with the synthetic namespace above.
    eval(compile(ast.Expression(calls[0]), "<launch-receipt>", "eval"), namespace)  # noqa: S307
    payload = json.loads((tmp_path / "planning_coordinate_readings.json").read_text())
    assert payload["readings"] == []
    assert payload["hints"] == [{
        "role": "measurement", "concept": "age", "column": "age",
        "basis": "keyword", "evidence": "age", "category": "demographics",
    }]


def test_readings_bind_only_the_actual_context_coordinate(tmp_path):
    from easyicu.webserver import planning_coordinate_readings as owner

    rows = owner.context_readings(_proposed(), columns={"exposure": "lact_max"}, configured={})
    assert rows[0]["column"] == "lact_max" and rows[0]["basis"] == "keyword"
    assert owner.context_readings(_proposed(), columns={"exposure": "lact_clearance"}, configured={}) == []
    owner.write_planning_coordinate_readings(tmp_path, rows)
    assert owner.load_planning_coordinate_readings(tmp_path, context_columns={"exposure": "lact_max"}) == rows
    assert owner.load_planning_coordinate_readings(tmp_path, context_columns={"exposure": "lact_min"}) == []
    assert owner.load_planning_coordinate_readings(tmp_path, context_columns=None) is None


def test_configured_exposure_has_no_keyword_receipt(tmp_path):
    from easyicu.webserver import planning_coordinate_readings as owner

    rows = owner.context_readings(_proposed(), columns={"exposure": "sofa"}, configured={"exposure": True})
    assert rows[0]["basis"] == "configured" and rows[0]["evidence"] is None
    owner.write_planning_coordinate_readings(tmp_path, rows)
    assert owner.load_planning_coordinate_readings(tmp_path, context_columns={"exposure": "sofa"}) == []


@pytest.mark.parametrize("value", ["{invalid", "[]", '{"schema_version":"other"}',
                                    '{"schema_version":"easyicu.planning_coordinate_readings/1","readings":"bad"}'])
def test_unverifiable_receipt_is_not_treated_as_a_known_empty_reading(tmp_path, value):
    from easyicu.webserver import planning_coordinate_readings as owner

    (tmp_path / owner.PLANNING_COORDINATE_READINGS_FILENAME).write_text(value)
    rows = owner.load_planning_coordinate_readings(tmp_path, context_columns={})
    assert rows is None
    assert owner.project_coordinate_readings(rows)["status"] == "unavailable"


def test_symlink_receipt_and_duplicate_roles_are_unavailable(tmp_path):
    from easyicu.webserver import planning_coordinate_readings as owner

    file = tmp_path / owner.PLANNING_COORDINATE_READINGS_FILENAME
    other = tmp_path / "other.json"
    other.write_text("{}")
    file.symlink_to(other)
    assert owner.load_planning_coordinate_readings(tmp_path, context_columns={}) is None
    file.unlink()
    owner.write_planning_coordinate_readings(tmp_path, [_proposed()[0], _proposed()[0]])
    assert owner.load_planning_coordinate_readings(tmp_path, context_columns={}) is None


def test_missing_legacy_record_adds_no_confirmation_claim(tmp_path):
    from easyicu.webserver import planning_coordinate_readings as owner

    assert owner.load_planning_coordinate_readings(tmp_path, context_columns=None) == []
    assert owner.project_coordinate_readings([]) is None


def test_actual_research_context_columns_are_used(tmp_path):
    from easyicu.webserver import planning_coordinate_readings as owner
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    context = ResearchContext(
        research_question="Association of lactate with mortality",
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=0),
        variables=[], target_outcome="death", primary_exposure="lact_max",
    )
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    assert owner.planned_context_columns(tmp_path) == {"outcome": "death", "exposure": "lact_max"}
    (tmp_path / "research_context.json").write_text("{}")
    assert owner.planned_context_columns(tmp_path) is None


def test_catalog_names_and_bounded_receipt_are_projected():
    from easyicu.webserver import planning_coordinate_readings as owner

    projected = owner.project_coordinate_readings(_proposed())
    assert projected["status"] == "shown"
    assert projected["items"][0]["label_en"] == "Lactate"
    assert projected["items"][0]["label_zh"] == "乳酸"
    assert owner.project_coordinate_readings([{**_proposed()[0], "column": "/private/source"}])["status"] == "unavailable"
    assert owner.project_coordinate_readings([{**_proposed()[0], "basis": []}])["status"] == "unavailable"
    assert owner.project_coordinate_readings([{**_proposed()[0], "evidence": None}])["status"] == "unavailable"


def test_pending_review_and_workflow_carry_the_receipt_without_changing_approval(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver import agent_pipeline_runs, planning_coordinate_readings as owner, study_contexts
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
    from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    wrapper, run_dir = tmp_path / "wrapper", tmp_path / "run"
    run_dir.mkdir()
    context = ResearchContext(
        research_question="Association of lactate with mortality",
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=0),
        variables=[], target_outcome="death", primary_exposure="lact_max",
    )
    (run_dir / "research_context.json").write_text(context.model_dump_json())
    rows = owner.context_readings(_proposed(), columns={"exposure": "lact_max"}, configured={})
    owner.write_planning_coordinate_readings(wrapper, rows)
    study = complete_study()
    registry = agent_pipeline_runs.PendingReviewRegistry()
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    pending = SimpleNamespace(run_id="reading-run", run_dir=str(run_dir), requests=(),
                              resume_scope="planning", resumable_here=True)
    registry.register(agent_pipeline_runs._PendingRun(
        pipeline=SimpleNamespace(), pending=pending, wrapper_dir=wrapper, study=study,
        provider={}, acquisition=SimpleNamespace(), created_at=1.0,
    ))
    review = agent_pipeline_runs.pending_review("reading-run")
    assert review["planning_coordinate_readings"] == rows
    digest = study_contexts.scientific_configuration_sha256(study)
    review.update(run_id="reading-run", scientific_configuration_sha256=digest,
                  scientific_plan_review={"status": "ready_for_approval", "findings": [], "approval_allowed": True})
    run = {"run_id": "reading-run", "engine": "easyicu.research_agent.pipeline", "run_type": "full",
           "run_status": "human_review_pending", "gate_status": "blocked",
           "scientific_configuration_sha256": digest,
           "pending_review_reason_codes": ["operator_plan_approval_required"],
           "artifact_names": ["agent_plan.json", "scientific_plan_review.json"]}
    inputs = dict(study=study, active_export_present=True, active_job=None, latest_run=run)
    before = build_research_workflow_snapshot(**inputs, plan_review_authority={**review, "planning_coordinate_readings": []})
    after = build_research_workflow_snapshot(**inputs, plan_review_authority=review)
    assert after.plan_review_summary["coordinate_readings"]["items"][0]["label_zh"] == "乳酸"
    assert before.next_action_code == after.next_action_code
    assert before.model_dump(exclude={"plan_review_summary"}) == after.model_dump(exclude={"plan_review_summary"})


def test_each_database_catalog_category_has_a_reader_name():
    import json
    import shutil
    from pathlib import Path
    from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
    from tests.support.node import run_node

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    categories = sorted({item.category for database in ["miiv", "eicu", "hirid", "aumc", "sic"]
                         for item in build_database_capability_catalog(database).concepts if item.category})
    script = """
global.window = {EU_LANG:'zh'};
require(process.argv[1]);
const keys = JSON.parse(process.argv[2]);
const zh = keys.map(key => window.AGENT_READER_VOCAB.category(key));
window.EU_LANG = 'en';
const en = keys.map(key => window.AGENT_READER_VOCAB.category(key));
process.stdout.write(JSON.stringify({zh,en,unknown:window.AGENT_READER_VOCAB.category('__proto__')}));
"""
    static = Path(__file__).resolve().parents[3] / "src/easyicu/webserver/static/js/screens-agent-reader-vocab.js"
    result = run_node(node, script, str(static), json.dumps(categories), check=True)
    rows = json.loads(result.stdout)
    assert all(rows["zh"]) and all(rows["en"])
    assert rows["unknown"] == ""
