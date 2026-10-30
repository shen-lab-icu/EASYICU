"""Preparation discovery carries verified, not-yet-applied population concepts."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.scientific_plan_review_gate import persist_or_validate_scientific_plan_review
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.webserver import agent_pipeline_runs, raw_preparation_candidate, sources
from tests.research_agent.planning.progressive_planner_fixtures import _payload
from tests.support.source_substitutions import _review
from tests.webserver.test_sic_raw_preparation_grant import _case


def _pending_with_unapplied_creatinine(tmp_path, result):
    question = "纳入基线肌酐不超过4 mg/dL的成年患者，缺失者排除，报告院内死亡率。"
    context = result.context.model_copy(update={
        "research_question": question,
        "variables": [v for v in result.context.variables if v.name != "creat" and v.source_concept != "creat"],
    })
    # A planner's first-24-hour proposal is not proof of admission baseline.
    # This test discovers the missing source concept while leaving approval
    # blocked; the clinical definition must still be independently reviewed.
    spec = PopulationSpec.model_validate({"criteria": [{
        "id": "c1", "kind": "measurement", "role": "include", "source": "question",
        "quote": "基线肌酐不超过4 mg/dL", "concept": "creat", "summary": "first",
        "op": "<=", "value": 4, "unit": "mg/dL",
        "window": {"start_hours": 0, "end_hours": 24}, "missing_policy": "exclude",
    }]})
    compiled = compile_population(spec, context)
    assert compiled.criteria[0].disposition != "applied_by_plan"
    plan = result.plan.model_copy(update={"research_question": question, "cohort": compiled.cohort_definition()})
    evidence = EvidenceStore(tmp_path)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence,
                                               current_review=_review(context, plan))
    skeleton_data = _payload()
    skeleton_data["cohort"]["population_spec"] = spec.model_dump(mode="json")
    skeleton = ProgressivePlanSkeleton.model_validate(skeleton_data).model_dump(mode="json")
    skeleton_sha = canonical_sha256(skeleton)
    plan_sha = canonical_sha256(plan.model_dump(mode="json"))
    files = {"scientific_plan_review": evidence.get("scientific_plan_review").sha256}

    def write(key, name, payload):
        content = json.dumps(payload, ensure_ascii=False, indent=2).encode()
        (tmp_path / name).write_bytes(content)
        files[key] = hashlib.sha256(content).hexdigest()

    write("progressive_plan_skeleton", "progressive_plan_skeleton.json", skeleton)
    write("research_context", "research_context.json", context.model_dump(mode="json"))
    write("progressive_plan_compile_receipt", "progressive_plan_compile_receipt.json", {
        "schema_version": "easyicu.progressive_plan_compile_receipt/1",
        "owner": "easyicu.planning.progressive_compiler_v1", "compiler_version": "1",
        "skeleton_sha256": skeleton_sha, "analysis_plan_sha256": "1" * 64, "compiled_steps": [],
    })
    body = {"schema_version": "easyicu.progressive_planning_authority/2", "planner_strategy": "progressive_v2",
        "outline_sha256": "1" * 64, "outline_structured_output_authority_sha256": None,
        "foundation_sha256": "1" * 64, "foundation_structured_output_authority_sha256": None,
        "ordered_steps": [{"step_id": "step1", "materialization_sha256": "1" * 64,
                           "structured_output_authority_sha256": None}],
        "compiled_skeleton_sha256": skeleton_sha, "compiled_analysis_plan_sha256": "1" * 64,
        "normalized_plan_sha256": plan_sha, "normalized_plan_authority_sha256": "1" * 64,
        "outline_artifact_sha256": "1" * 64, "foundation_artifact_sha256": "1" * 64,
        "materialization_ledger_sha256": "1" * 64,
        "skeleton_artifact_sha256": files["progressive_plan_skeleton"],
        "compile_receipt_artifact_sha256": files["progressive_plan_compile_receipt"],
        "planner_prompt_metrics_sha256": "1" * 64, "analysis_plan_artifact_sha256": "1" * 64,
        "normalized_plan_artifact_sha256": "1" * 64, "strict_transport_bound": False}
    body["authority_sha256"] = canonical_sha256(body)
    write("progressive_planning_authority", "progressive_planning_authority.json", body)
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256=files)
    pending = SimpleNamespace(run_id="run", run_dir=str(tmp_path), requests=[
        SimpleNamespace(payload={"plan_review_authority": authority.model_dump(mode="json")})])
    return pending, plan


def test_unapplied_chinese_threshold_reaches_existing_metadata_discovery(tmp_path, monkeypatch):
    source_fixture = tmp_path / "source-fixture"
    source_fixture.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result, source_root, _, _ = _case(source_fixture, monkeypatch, approved=False)
    pending, plan = _pending_with_unapplied_creatinine(run_dir, result)
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (pending, {"id": "study"}))
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [{"path": str(source_root)}]})
    observed = []
    capture = raw_preparation_candidate.capture_data_provenance

    def observe(*args, **kwargs):
        observed.append(tuple(kwargs["concepts"]))
        return capture(*args, **kwargs)

    monkeypatch.setattr(raw_preparation_candidate, "capture_data_provenance", observe)
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    assert "creat" in observed[-1]
    assert not any(item.concept_id == "creat" for item in (*plan.cohort.inclusion, *plan.cohort.exclusion))
    assert outcome["status"] == "blocked"
    assert "dependency_cohort_inputs_unknown" in outcome["blocker_codes"]
    assert not (run_dir / "raw_preparation_checkpoint.json").exists()
    assert not (run_dir / "approved_executable_plan.json").exists()

    # A changed skeleton no longer supplies the review-bound criterion. The
    # unrelated input roster remains blocked; no changed request is approved.
    path = run_dir / "progressive_plan_skeleton.json"
    original_skeleton = path.read_bytes()
    path.write_bytes(original_skeleton + b" ")
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    assert "creat" not in observed[-1]
    assert outcome["status"] == "blocked"
    assert "dependency_cohort_inputs_unknown" in outcome["blocker_codes"]
    assert not (run_dir / "raw_preparation_checkpoint.json").exists()

    path.write_bytes(original_skeleton)
    context_path = run_dir / "research_context.json"
    context_path.write_bytes(context_path.read_bytes() + b" ")
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    assert "creat" not in observed[-1]
    assert outcome["status"] == "blocked"
    assert "dependency_cohort_inputs_unknown" in outcome["blocker_codes"]
    assert not (run_dir / "raw_preparation_checkpoint.json").exists()

    # Even a recomputed self-hash for a different plan cannot borrow the
    # signed scientific review for the original plan.
    context_path.write_bytes(context_path.read_bytes()[:-1])
    forged = json.loads(json.dumps(pending.requests[0].payload))
    plan_packet = forged["plan_review_authority"]
    plan_packet["plan_payload"]["research_question"] = "不同的研究问题"
    plan_packet["plan_sha256"] = canonical_sha256(plan_packet["plan_payload"])
    pending.requests = [SimpleNamespace(payload=forged)]
    with pytest.raises(ValueError, match="dependency_scientific_review_mismatch"):
        raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    assert not (run_dir / "raw_preparation_checkpoint.json").exists()


@pytest.mark.parametrize("change", ["plan", "shape"])
def test_population_projection_must_bind_the_paused_plan(change):
    from easyicu.webserver.raw_preparation_dependencies import reviewed_population_provenance_concepts

    projection = {"schema_version": "easyicu.reviewed_population_spec_projection/1",
                  "plan_sha256": "a" * 64, "criteria": []}
    if change == "plan":
        projection["plan_sha256"] = "b" * 64
    else:
        projection["criteria"] = {"untrusted": True}
    with pytest.raises(ValueError, match="population_projection_mismatch"):
        reviewed_population_provenance_concepts(projection=projection, plan_sha256="a" * 64)


def test_first_observed_still_requires_a_separate_source_bound_adapter():
    from easyicu.webserver.raw_preparation_dependencies import reviewed_population_provenance_concepts

    projection = {"schema_version": "easyicu.reviewed_population_spec_projection/1",
                  "plan_sha256": "a" * 64, "criteria": [{"criterion": {
                      "id": "c1", "kind": "first_observed_icu_single_hospital_adult",
                      "source": "question", "role": "include", "quote": "可确定的首次入住",
                  }}]}
    assert reviewed_population_provenance_concepts(projection=projection, plan_sha256="a" * 64) == ()
