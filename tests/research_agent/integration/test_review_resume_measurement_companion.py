"""An approved prediction cohort can use its materialized measurement status."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision
from easyicu.research_agent.planning.cohort_contract import concept_id_exists
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from tests.support.durable_human_review import _force_approvable_plan_review, _pipeline
from tests.support.pipeline_contracts import disable_article_contract, stable_plan_rules
from tests.support.typed_export import typed_export

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_COMPANION = "lact_measured"
_QUESTION = (
    "Can early lactate predict hospital mortality among stays with a measurement?"
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
                    "step_id": "cohort_definition",
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
        stop_after_step_id="cohort_definition",
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
    assert step_statuses["cohort_definition"] == "ok"
    assert load_checkpoint(checkpoint_path, require_pending=False).state == "completed"
    assert not concept_id_exists(_COMPANION)


def test_approved_cohort_binding_refuses_an_unrecorded_concept(tmp_path, monkeypatch):
    from easyicu.research_agent.authority import execution_input
    from easyicu.research_agent.authority.analysis_cohort import (
        ExecutionCohortSchemaError,
    )
    from easyicu.webserver.agent_pipeline_runs import (
        _write_review_resume_failure_diagnostic,
    )
    from easyicu.webserver.execution_failure import track_failure_location

    first, pending, _ = _pending_prediction_plan(tmp_path, monkeypatch)
    run_dir = Path(pending.run_dir)
    checkpoint_path = run_dir / "human_review_checkpoint.json"
    unknown = "unrecorded_fixture_measured"
    checkpoint = load_checkpoint(checkpoint_path)
    assert unknown not in checkpoint.plan_handoff["cohort_concept_ids"]
    assert not concept_id_exists(unknown)
    real_bind = execution_input.bind_execution_cohort_authority

    def bind_damaged_handoff(*, plan, **kwargs):
        # Simulate an invalid handoff at the replay boundary. The upstream
        # approval digest and cohort lock normally refuse this change first.
        cohort = plan.cohort.to_dict()
        cohort["inclusion"][0]["concept_id"] = unknown
        return real_bind(plan=plan.model_copy(update={"cohort": cohort}), **kwargs)

    monkeypatch.setattr(
        execution_input, "bind_execution_cohort_authority", bind_damaged_handoff
    )
    with pytest.raises(
        ExecutionCohortSchemaError, match=f"unknown concept_id: {unknown}"
    ) as caught:
        with track_failure_location(runner_kind="docker") as progress_callback:
            first.resume_human_review(
                _decisions(pending),
                run_id=pending.run_id,
                progress_callback=progress_callback,
            )

    assert load_checkpoint(checkpoint_path, require_pending=False).state == "failed"
    assert not first.has_resumable_human_review
    assert not concept_id_exists(unknown)
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
        "step_id": "cohort_definition",
        "runner_kind": "not_started",
        "exception_type": "ExecutionCohortSchemaError",
    }
    assert unknown not in json.dumps(receipt)
