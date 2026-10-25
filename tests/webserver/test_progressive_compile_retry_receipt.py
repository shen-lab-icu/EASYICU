"""Compile rejections reach the failure receipt without model or compiler prose."""

import json

import pytest

from easyicu.research_agent.agents.progressive_attempt import (
    ProgressivePlannerAttemptState, bind_progressive_planner_failure_facts,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.progressive_artifacts import (
    ProgressiveCompileReplayAttempt, ProgressiveCompilerFinding,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError, ProgressiveStepMaterialization
from easyicu.webserver import agent_pipeline_runs
from tests.research_agent.planning.progressive_planner_fixtures import _materialization_payloads


@pytest.mark.parametrize("wrapped", [False, True])
def test_compile_history_survives_wrapping_without_rejected_values(tmp_path, wrapped):
    error = ProgressivePlanCompileError("progressive_analysis_module_unavailable", "private synthetic compiler text")
    materialization = ProgressiveStepMaterialization.model_validate(_materialization_payloads()[1])
    state = ProgressivePlannerAttemptState(compile_failure_attempts=[
        ProgressiveCompileReplayAttempt(
            revision=revision, step_schema_authority_sha256=None, materialization=materialization,
            materialization_sha256=canonical_sha256(materialization.model_dump(mode="json")),
            compiler_finding=ProgressiveCompilerFinding.from_details(error.details),
        ) for revision in range(3)
    ])
    bind_progressive_planner_failure_facts(error, state.freeze())
    # Exception attributes are mutable; the web boundary must re-project them.
    error.easyicu_compile_retry_metadata[0]["error_message"] = "private synthetic model output"
    outer = RuntimeError("private synthetic wrapper") if wrapped else error
    if wrapped:
        outer.__cause__ = error
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={}, exc=outer,
        code="research_pipeline_progressive_compile_failed", execution_retry_id=None,
    )
    receipt = json.loads((tmp_path / "diagnostics/research_pipeline_failure.json").read_text())
    history = receipt["structured_retry_history"]
    assert [row["attempt"] for row in history] == [1, 2, 3]
    assert {row["reason_code"] for row in history} == {"progressive_analysis_module_unavailable"}
    assert "private synthetic" not in json.dumps(receipt)
    assert "raw_inputs" not in json.dumps(history)


def test_parse_history_remains_visible_when_no_compile_revisions_exist(tmp_path):
    error = RuntimeError("private parse failure")
    error.easyicu_structured_attempt_metadata = [{"attempt": 1, "error_class": "ValueError", "raw_head": "private output"}]
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={}, exc=error,
        code="research_pipeline_planner_contract_failed", execution_retry_id=None,
    )
    receipt = json.loads((tmp_path / "diagnostics/research_pipeline_failure.json").read_text())
    assert len(receipt["structured_retry_history"]) == 1
    assert receipt["structured_retry_history"] == receipt["structured_attempts"]
    assert "private" not in json.dumps(receipt)


def test_python_stop_sentence_names_the_module_failure():
    sentence = agent_pipeline_runs._progressive_compile_failure_message(
        ProgressivePlanCompileError("progressive_analysis_module_unavailable", "private rejected output"),
    )
    assert sentence == (
        "The candidate plan used an analysis step that EasyICU cannot run for this kind of study, "
        "so planning stopped and no analysis was run. Generate the plan again."
    )
