"""An approved prediction cohort can use its materialized measurement status."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.analysis_cohort import ExecutionCohortSchemaError
from easyicu.research_agent.authority.execution_input import ExecutionInputAuthorityState
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision
from easyicu.research_agent.planning.cohort_contract import (
    cohort_concept_id_scope,
    concept_id_exists,
)
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from easyicu.research_agent.research_context.typed import parse_research_context_json
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver.agent_pipeline_runs import _write_review_resume_failure_diagnostic
from easyicu.webserver.execution_failure import track_failure_location
from tests.support.durable_human_review import _force_approvable_plan_review, _pipeline
from tests.support.pipeline_contracts import disable_article_contract, stable_plan_rules
from tests.support.typed_export import typed_export

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_COMPANION = "lact_measured"
_COHORT_STEP_ID = "screen_lab_availability"
_UNRECORDED = "unrecorded_fixture_measured"
_QUESTION = (
    "Among ICU stays with a documented lactate result, how well does the result "
    "predict death before hospital discharge?"
)


def _pending_prediction_plan(tmp_path, monkeypatch):
    disable_article_contract(monkeypatch)
    _force_approvable_plan_review(monkeypatch)
    export = typed_export(
        tmp_path / "export",
        labs=pd.DataFrame(
            {
                "stay_id": range(1, 61),
                "charttime": [1.0] * 60,
                "age": [45, 60, 75] * 20,
                "lact": [1.0, 2.0, 3.0, 4.0, 5.0, None] * 10,
                "mech_vent": [False] * 60,
            }
        ),
        outcomes=pd.DataFrame({"stay_id": range(1, 61), "death": [False, True] * 30}),
    )
    cohort = materialize_to_parquet(
        tmp_path / "materialized",
        data_path=export,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )["parquet"]
    plan = json.dumps(
        {
            "research_question": _QUESTION,
            "analysis_type": "prediction_model",
            "cohort": {
                "name": "measured_stays",
                "inclusion": [
                    {
                        "concept_id": _COMPANION,
                        "time_window": {
                            "anchor": "icu_admission",
                            "start_offset_hours": 0,
                            "end_offset_hours": 24,
                        },
                        "aggregation": "first",
                        "op": "==",
                        "value": 1,
                    }
                ],
            },
            "steps": [
                {
                    "step_id": _COHORT_STEP_ID,
                    "planned_analysis_role": "auxiliary",
                    "intent": "Account for the bound analysis cohort.",
                    "inputs": ["lact_max"],
                    "expected_outputs": [
                        "artifact:analysis_cohort",
                        "table:cohort_flow",
                    ],
                    "cohort_definition_spec": {"identity_column": "stay_id"},
                },
                {
                    "step_id": "primary_model",
                    "planned_analysis_role": "primary",
                    "intent": "Fit a mortality prediction model.",
                    "inputs": ["lact_max", "death", "artifact:analysis_cohort"],
                    "expected_outputs": [
                        "table:prediction_scores",
                        "table:model_performance",
                    ],
                    "method": "logistic prediction model",
                    "scientific_action_id": "prediction.discrimination_calibration",
                },
            ],
        }
    )
    workdir = tmp_path / "runs"

    def pipeline():
        llm = PatternScriptedMockLLMClient(
            stable_plan_rules(plan), contextual_default=True
        )
        return _pipeline(
            workdir,
            llm=llm,
            enable_literature=False,
            enable_replanning=False,
        )

    first = pipeline()
    pending = first.run(
        question=_QUESTION,
        cohort=cohort,
        target_outcome="death",
        database="miiv",
        stop_after_analysis=True,
        stop_after_step_id=_COHORT_STEP_ID,
    )
    return first, pending, pipeline


def _decisions(pending):
    return [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="test reviewer",
            decided_at="2026-11-17T00:00:00Z",
        )
        for request in pending.requests
    ]


@pytest.mark.parametrize("restore", [False, True], ids=["live", "durable"])
def test_approved_prediction_resumes_with_recorded_measurement_companion(
    tmp_path,
    monkeypatch,
    restore,
):
    first, pending, pipeline = _pending_prediction_plan(tmp_path, monkeypatch)
    run_dir = Path(pending.run_dir)
    checkpoint_path = run_dir / "human_review_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    reviewed_plan = checkpoint.plan_handoff["plan"]
    assert reviewed_plan["analysis_type"] == "prediction_model"
    assert reviewed_plan["cohort"]["inclusion"][0]["concept_id"] == _COMPANION
    assert (
        reviewed_plan["steps"][0]["cohort_definition_spec"]["identity_column"]
        == "stay_id"
    )
    assert _COMPANION in checkpoint.plan_handoff["cohort_concept_ids"]
    assert "Initial cohort audit passed." in (run_dir / "audit_log.jsonl").read_text(
        encoding="utf-8"
    )
    assert len(pd.read_parquet(run_dir / "cohort_analysis.parquet")) == 50
    assert not concept_id_exists(_COMPANION)

    result = (pipeline() if restore else first).resume_human_review(
        _decisions(pending),
        run_id=pending.run_id,
    )

    assert result.run_id == pending.run_id
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    step_statuses = {
        record["step_id"]: record["status"] for record in manifest["per_step_records"]
    }
    assert step_statuses[_COHORT_STEP_ID] == "ok"
    assert load_checkpoint(checkpoint_path, require_pending=False).state == "completed"
    assert not concept_id_exists(_COMPANION)


def test_approved_cohort_binding_refuses_an_unrecorded_concept(tmp_path, monkeypatch):
    from easyicu.research_agent.authority import execution_input

    first, pending, _ = _pending_prediction_plan(tmp_path, monkeypatch)
    run_dir = Path(pending.run_dir)
    checkpoint_path = run_dir / "human_review_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    assert _UNRECORDED not in checkpoint.plan_handoff["cohort_concept_ids"]
    assert not concept_id_exists(_UNRECORDED)
    real_bind = execution_input.bind_execution_cohort_authority

    def bind_damaged_handoff(*, plan, **kwargs):
        # Simulate an invalid handoff at the replay boundary. The upstream
        # approval digest and cohort lock normally refuse this change first.
        return real_bind(plan=_plan_with_unrecorded_concept(plan), **kwargs)

    monkeypatch.setattr(
        execution_input, "bind_execution_cohort_authority", bind_damaged_handoff
    )
    with pytest.raises(
        ExecutionCohortSchemaError, match=f"unknown concept_id: {_UNRECORDED}"
    ) as caught:
        with track_failure_location(runner_kind="docker") as progress_callback:
            first.resume_human_review(
                _decisions(pending),
                run_id=pending.run_id,
                progress_callback=progress_callback,
            )

    assert load_checkpoint(checkpoint_path, require_pending=False).state == "failed"
    assert not first.has_resumable_human_review
    assert not concept_id_exists(_UNRECORDED)
    diagnostic_path = _write_review_resume_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=caught.value,
        code="research_pipeline_approved_run_failed",
        review_resumable=False,
    )
    assert diagnostic_path is not None
    receipt = json.loads((tmp_path / diagnostic_path).read_text(encoding="utf-8"))
    assert receipt["typed_failure"] == {
        "owner": "easyicu.execution.cohort_authority_v1",
        "reason_code": "execution_cohort_schema_invalid",
        "phase": "execution_input_binding",
        "step_id": _COHORT_STEP_ID,
        "runner_kind": "not_started",
        "exception_type": "ExecutionCohortSchemaError",
    }
    assert _UNRECORDED not in json.dumps(receipt)


def _plan_with_unrecorded_concept(plan):
    cohort = plan.cohort.to_dict()
    cohort["inclusion"][0]["concept_id"] = _UNRECORDED
    return plan.model_copy(update={"cohort": cohort})


def _reviewed_binding_inputs(pending):
    run_dir = Path(pending.run_dir)
    handoff = load_checkpoint(run_dir / "human_review_checkpoint.json").plan_handoff
    concept_ids = tuple(handoff["cohort_concept_ids"])
    with cohort_concept_id_scope(concept_ids):
        plan = AnalysisPlan.model_validate(handoff["plan"])
    return {
        "universe_path": run_dir / "cohort.parquet",
        "analysis_path": run_dir / "cohort_analysis.parquet",
        "trajectory_binding": None,
        "run_dir": run_dir,
        "legacy_trajectory_verifier": None,
        "plan": plan,
        "context": parse_research_context_json((run_dir / "research_context.json").read_bytes()),
        "cohort_concept_ids": concept_ids,
    }


def test_cohort_rebind_receipt_reports_the_active_step(tmp_path, monkeypatch):
    _, pending, _ = _pending_prediction_plan(tmp_path, monkeypatch)
    inputs = _reviewed_binding_inputs(pending)
    state = ExecutionInputAuthorityState.bind(**inputs)

    with pytest.raises(
        ExecutionCohortSchemaError, match=f"unknown concept_id: {_UNRECORDED}"
    ) as caught:
        with track_failure_location(runner_kind="docker") as progress:
            progress({"stage": "step", "step_id": _COHORT_STEP_ID, "status": "complete"})
            progress({"stage": "step", "step_id": "primary_model", "status": "running"})
            state.rebind_cohort(
                plan=_plan_with_unrecorded_concept(inputs["plan"]),
                context=inputs["context"],
            )

    assert state.selected_path == inputs["analysis_path"]
    wrapper = RuntimeError("internal rebind failure")
    wrapper.__cause__ = caught.value
    diagnostic_path = _write_review_resume_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=wrapper,
        code="research_pipeline_approved_run_failed",
        review_resumable=False,
    )
    assert diagnostic_path is not None
    receipt = json.loads((tmp_path / diagnostic_path).read_text(encoding="utf-8"))
    assert receipt["typed_failure"] == {
        "owner": "easyicu.execution.cohort_authority_v1",
        "reason_code": "execution_cohort_schema_invalid",
        "phase": "execute",
        "step_id": "primary_model",
        "runner_kind": "docker",
        "exception_type": "ExecutionCohortSchemaError",
    }
    assert _UNRECORDED not in json.dumps(receipt)


def test_cohort_rebind_retains_a_single_pass_concept_roster(tmp_path, monkeypatch):
    _, pending, _ = _pending_prediction_plan(tmp_path, monkeypatch)
    inputs = _reviewed_binding_inputs(pending)
    concept_ids = inputs["cohort_concept_ids"]
    assert _COMPANION in concept_ids

    state = ExecutionInputAuthorityState.bind(
        **{**inputs, "cohort_concept_ids": iter(concept_ids)}
    )
    assert state.selected_path == inputs["analysis_path"]
    state.rebind_cohort(plan=inputs["plan"], context=inputs["context"])
    assert state.selected_path == inputs["analysis_path"]
    assert state.cohort_concept_ids == concept_ids
    assert not concept_id_exists(_COMPANION)
