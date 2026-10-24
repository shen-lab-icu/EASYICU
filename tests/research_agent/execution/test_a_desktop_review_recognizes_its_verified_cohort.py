"""A script-free host cohort must not be mistaken for model-generated code."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    write_locked_cohort_definition,
)
from easyicu.research_agent.execution.cohort_adoption import (
    adopt_existing_host_cohort_materialization,
)
from easyicu.research_agent.execution.owner_declaration import (
    HostPlanStepNeedsSandbox,
    require_host_rendered_plan,
)
from easyicu.research_agent.schema import AnalysisStep, CohortDefinitionSpec
from tests.support.host_cohort import (
    _sealable_plan,
    typed_run as typed_run,
)


def _plan():
    plan = _sealable_plan()
    step = plan.steps[0]
    step.planned_analysis_role = "auxiliary"
    step.cohort_definition_spec = CohortDefinitionSpec(
        identity_column="stay_id",
        eligibility_criteria=[
            {"criterion_id": "age_bound", "description": "Age at least 55"}
        ],
    )
    return plan


def _lock(run_dir, plan):
    write_locked_cohort_definition(
        run_dir=run_dir, plan=plan, evidence=EvidenceStore(run_dir),
        prompt_pack_version="test", llm_signature="test",
    )


def test_review_and_execute_recognize_the_same_verified_products(typed_run):
    run_dir, _ = typed_run
    plan = _plan()
    _lock(run_dir, plan)
    before = {
        str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_dir.rglob("*") if path.is_file()
    }

    require_host_rendered_plan(plan=plan, run_dir=run_dir)

    after = {
        str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_dir.rglob("*") if path.is_file()
    }
    assert before == after, "Pre-review ownership must not write an approval or receipt"
    records, preexecuted, findings = [], set(), []
    adopt_existing_host_cohort_materialization(
        plan=plan, run_dir=run_dir, cohort_path=run_dir / "cohort_analysis.parquet",
        evidence=EvidenceStore(run_dir), prompt_pack_version="test",
        llm_signature="test", gate_stamp={}, per_step_records=records,
        preexecuted_step_ids=preexecuted, findings=findings,
    )
    assert preexecuted == {plan.steps[0].step_id}
    assert records[0]["status"] == "ok"
    assert records[0]["step_summary"]["materialized_cohort_authority_ref"]
    assert not findings


@pytest.mark.parametrize("proof", ["no_run", "no_files", "bytes", "flow", "definition", "ledger", "lock"])
def test_missing_or_changed_materialization_cannot_grant_host_execution(typed_run, tmp_path, proof):
    run_dir, _ = typed_run
    plan = _plan()
    _lock(run_dir, plan)
    if proof == "no_run":
        run_dir = None
    elif proof == "no_files":
        run_dir = tmp_path / "empty"
    elif proof == "bytes":
        with (run_dir / "cohort_analysis.parquet").open("ab") as stream:
            stream.write(b"changed")
    elif proof == "flow":
        (run_dir / "cohort_analysis_flow.csv").unlink()
    elif proof == "definition":
        plan.cohort = CohortDefinition(name="different", inclusion=())
    elif proof == "ledger":
        ledger_path = run_dir / "cohort_analysis_provenance.json"
        ledger = json.loads(ledger_path.read_text())
        ledger["n_analysis_cohort"] += 1
        ledger_path.write_text(json.dumps(ledger))
    else:
        (run_dir / "cohort_locked.json").unlink()
    with pytest.raises(HostPlanStepNeedsSandbox):
        require_host_rendered_plan(plan=plan, run_dir=run_dir)


@pytest.mark.parametrize("declaration", ["duplicate", "untyped", "different_method", "primary"])
def test_product_names_do_not_replace_a_unique_typed_cohort_declaration(typed_run, declaration):
    run_dir, _ = typed_run
    plan = _plan()
    _lock(run_dir, plan)
    if declaration == "duplicate":
        plan.steps.append(plan.steps[0].model_copy(update={"step_id": "second_cohort"}))
    elif declaration == "untyped":
        plan.steps[0].cohort_definition_spec = None
    elif declaration == "different_method":
        plan.steps[0].method = "custom algorithm"
    else:
        plan.steps[0].planned_analysis_role = "primary"
    with pytest.raises(HostPlanStepNeedsSandbox):
        require_host_rendered_plan(plan=plan, run_dir=run_dir)


def test_a_verified_cohort_cannot_authorize_an_unowned_analysis(typed_run):
    run_dir, _ = typed_run
    plan = _plan()
    _lock(run_dir, plan)
    plan.steps.append(AnalysisStep(
        step_id="custom", intent="Unowned calculation", method="custom algorithm",
        expected_outputs=["table:custom"],
    ))
    with pytest.raises(HostPlanStepNeedsSandbox) as error:
        require_host_rendered_plan(plan=plan, run_dir=run_dir)
    assert error.value.easyicu_safe_diagnostic["step_id"] == "custom"


@pytest.mark.parametrize("resumed", [False, True])
def test_pipeline_supplies_the_actual_run_directory_before_review(monkeypatch, tmp_path, resumed):
    from easyicu.research_agent import pipeline

    plan = _plan()
    owner = SimpleNamespace()
    class Checked(Exception):
        pass

    def checked(actual_owner, actual_plan, *, run_dir):
        assert actual_owner is owner
        assert actual_plan is plan
        assert run_dir == tmp_path
        raise Checked

    monkeypatch.setattr(pipeline, "require_host_plan_before_review", checked)
    with pytest.raises(Checked):
        pipeline._pipeline_run___human_review_invoker(
            SimpleNamespace(
                plan=plan,
                plan_path=(
                    tmp_path / "evidence/analysis_plan__analysis_plan.json"
                    if resumed else tmp_path / "analysis_plan.json"
                ),
                evidence=SimpleNamespace(root=tmp_path) if resumed else None,
            ),
            reviewed_plan=[], self=owner,
        )


def _with_audit_figures():
    plan = _plan()
    plan.steps.extend([
        AnalysisStep(
            step_id="audit", intent="Audit missingness and measurement process",
            method="missing_data", planned_analysis_role="auxiliary",
            inputs=["sex", "death", "artifact:analysis_cohort"],
            expected_outputs=["table:measurement_missingness", "table:measurement_process_audit"],
            measurement_audit_spec={"products": [
                {"product_id": "measurement_missingness", "audit": "measurement_missingness"},
                {"product_id": "measurement_process_audit", "audit": "measurement_process"},
            ]},
        ),
        AnalysisStep(
            step_id="flow_figure", intent="Draw the exact host cohort flow",
            method="visualization", planned_analysis_role="auxiliary",
            inputs=["table:cohort_flow"], expected_outputs=["figure:flow"],
            input_consumption_contracts=[{"input_key": "table:cohort_flow", "mode": "all_rows"}],
        ),
        AnalysisStep(
            step_id="audit_figure", intent="Draw the exact missingness and process tables",
            method="visualization", planned_analysis_role="auxiliary",
            inputs=["table:measurement_missingness", "table:measurement_process_audit"],
            expected_outputs=["figure:data_quality"],
            input_consumption_contracts=[
                {"input_key": "table:measurement_missingness", "mode": "all_rows"},
                {"input_key": "table:measurement_process_audit", "mode": "all_rows"},
            ],
        ),
    ])
    return plan


def test_pre_review_capability_does_not_grant_runtime_figure_ownership(typed_run):
    from easyicu.research_agent.execution.runners.selection import resolve_standard_executor

    run_dir, _ = typed_run
    plan = _with_audit_figures()
    _lock(run_dir, plan)
    require_host_rendered_plan(plan=plan, run_dir=run_dir)
    # No invented table bindings: actual execution still needs the table
    # columns, consumption contracts and verified provenance after analysis.
    for figure in plan.steps[-2:]:
        assert resolve_standard_executor(figure, plan=plan).claimed_by is None


@pytest.mark.parametrize("change", ["duplicate_parent", "unowned_parent", "no_consumption", "wrong_input"])
def test_pending_figure_needs_its_unique_declared_host_producers(typed_run, change):
    run_dir, _ = typed_run
    plan = _with_audit_figures()
    _lock(run_dir, plan)
    if change == "duplicate_parent":
        plan.steps.insert(2, plan.steps[1].model_copy(update={"step_id": "audit_again"}))
    elif change == "unowned_parent":
        plan.steps[1].method = "custom algorithm"
        plan.steps[1].measurement_audit_spec = None
    elif change == "no_consumption":
        plan.steps[-1].input_consumption_contracts = []
    else:
        plan.steps[-1].inputs = ["table:unproduced"]
    with pytest.raises(HostPlanStepNeedsSandbox):
        require_host_rendered_plan(plan=plan, run_dir=run_dir)


def test_ambiguous_cohort_products_are_refused_before_loading_a_grant(tmp_path, monkeypatch):
    from easyicu.research_agent.cohort import schema
    from easyicu.research_agent.execution.owner_declaration import _verified_host_cohort_step

    plan = _plan()
    plan.steps.append(plan.steps[0].model_copy(update={"step_id": "competing_cohort"}))
    consulted = []

    def load(**kwargs):
        consulted.append(kwargs)
        return None

    monkeypatch.setattr(schema, "load_materialized_analysis_cohort_result", load)
    assert _verified_host_cohort_step(plan, tmp_path) is None
    assert consulted == []
