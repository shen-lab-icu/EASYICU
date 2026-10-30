"""A review displays only the exact typed population bound by its authority."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.cohort_contract import cohort_concept_id_scope, sealed_cohort_concept_ids
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.schema import AnalysisPlan, ConceptDescriptor, VariableRole
from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec
from tests.research_agent.planning.progressive_planner_fixtures import _payload
from tests.research_agent.planning.test_a_population_spec_compiles_to_the_cohort_it_states import _ctx


@pytest.fixture
def review(tmp_path):
    context = _ctx(ConceptDescriptor(name="creat_first", role=VariableRole.LAB,
        dtype="float64", source_concept="creat", unit="mg/dL",
        analysis_window="icu_admission[0,24]h"))
    spec = PopulationSpec.model_validate({"criteria": [{
        "id": "c1", "kind": "measurement", "role": "include", "source": "question",
        "quote": "first creatinine up to 4 mg/dL in the first 24 h", "concept": "creat",
        "summary": "first", "op": "<=", "value": 4, "unit": "mg/dL",
        "window": {"start_hours": 0, "end_hours": 24}, "missing_policy": "exclude",
    }]})
    compiled = compile_population(spec, context)
    with cohort_concept_id_scope(sealed_cohort_concept_ids(context)):
        plan = AnalysisPlan(research_question="synthetic", steps=[], cohort=compiled.cohort_definition())
    sk = _payload(); sk["cohort"]["population_spec"] = spec.model_dump(mode="json")
    skeleton = ProgressivePlanSkeleton.model_validate(sk).model_dump(mode="json")
    skeleton_hash = canonical_sha256(skeleton)
    plan_sha = canonical_sha256(plan.model_dump(mode="json"))
    files = {}

    def write(key, name, value):
        content = json.dumps(value, ensure_ascii=False, indent=2).encode()
        (tmp_path / name).write_bytes(content)
        files[key] = hashlib.sha256(content).hexdigest()

    write("progressive_plan_skeleton", "progressive_plan_skeleton.json", skeleton)
    write("research_context", "research_context.json", context.model_dump(mode="json"))
    write("progressive_plan_compile_receipt", "progressive_plan_compile_receipt.json", {
        "schema_version": "easyicu.progressive_plan_compile_receipt/1",
        "owner": "easyicu.planning.progressive_compiler_v1", "compiler_version": "1",
        "skeleton_sha256": skeleton_hash, "analysis_plan_sha256": "1" * 64, "compiled_steps": [],
    })
    body = {"schema_version": "easyicu.progressive_planning_authority/2", "planner_strategy": "progressive_v2",
        "outline_sha256": "1"*64, "outline_structured_output_authority_sha256": None,
        "foundation_sha256": "1"*64, "foundation_structured_output_authority_sha256": None,
        "ordered_steps": [{"step_id": "step1", "materialization_sha256": "1"*64,
                           "structured_output_authority_sha256": None}],
        "compiled_skeleton_sha256": skeleton_hash, "compiled_analysis_plan_sha256": "1"*64,
        "normalized_plan_sha256": plan_sha, "normalized_plan_authority_sha256": "1"*64,
        "outline_artifact_sha256": "1"*64, "foundation_artifact_sha256": "1"*64,
        "materialization_ledger_sha256": "1"*64,
        "skeleton_artifact_sha256": files["progressive_plan_skeleton"],
        "compile_receipt_artifact_sha256": files["progressive_plan_compile_receipt"],
        "planner_prompt_metrics_sha256": "1"*64, "analysis_plan_artifact_sha256": "1"*64,
        "normalized_plan_artifact_sha256": "1"*64, "strict_transport_bound": False}
    body["authority_sha256"] = canonical_sha256(body)
    write("progressive_planning_authority", "progressive_planning_authority.json", body)
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256=files)
    pending = SimpleNamespace(requests=[SimpleNamespace(payload={"plan_review_authority": authority.model_dump(mode="json")})])
    return tmp_path, pending


def test_review_displays_exact_compiled_spec_from_the_bound_source(review):
    root, pending = review
    result = reviewed_population_spec(run_dir=root, pending=pending)
    assert result is not None
    row = result["criteria"][0]
    assert row["criterion"]["quote"] == "first creatinine up to 4 mg/dL in the first 24 h"
    assert row["criterion"]["unit"] == "mg/dL"
    assert row["criterion"]["window"] == {"start_hours": 0.0, "end_hours": 24.0}
    assert row["criterion"]["missing_policy"] == "exclude"
    assert row["disposition"] == "applied_by_plan"


@pytest.mark.parametrize("filename", ["progressive_plan_skeleton.json", "research_context.json",
                                      "progressive_planning_authority.json", "progressive_plan_compile_receipt.json"])
def test_review_refuses_modified_or_missing_bound_artifacts(review, filename):
    root, pending = review
    path = root / filename
    path.write_bytes(path.read_bytes() + b" ")
    assert reviewed_population_spec(run_dir=root, pending=pending) is None


def test_review_refuses_another_plan_and_missing_authority(review):
    root, pending = review
    raw = pending.requests[0].payload["plan_review_authority"]
    altered = dict(raw["plan_payload"]); altered["research_question"] = "A different question"
    raw["plan_payload"] = altered
    assert reviewed_population_spec(run_dir=root, pending=pending) is None
    assert reviewed_population_spec(run_dir=root, pending=SimpleNamespace(requests=[])) is None


def test_multiple_requests_can_include_an_unrelated_request_without_plan_authority(review):
    root, pending = review
    pending.requests.append(SimpleNamespace(payload={"kind": "another_review"}))
    assert reviewed_population_spec(run_dir=root, pending=pending) is not None


def test_multiple_different_review_authorities_fail_closed(review):
    root, pending = review
    raw = pending.requests[0].payload["plan_review_authority"]
    different = dict(raw)
    different["evidence_sha256"] = dict(raw["evidence_sha256"], other="f" * 64)
    pending.requests.append(SimpleNamespace(payload={"plan_review_authority": different}))
    assert reviewed_population_spec(run_dir=root, pending=pending) is None
