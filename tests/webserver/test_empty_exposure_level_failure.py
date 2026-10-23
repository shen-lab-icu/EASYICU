"""The host carries empty-level failures into the conversation and retry policy."""

from dataclasses import replace
import json
import shutil

import pytest

from easyicu.research_agent.execution import retry_basis
from easyicu.research_agent.execution.failure_classification import (
    DECLARED_EXPOSURE_LEVEL_EMPTY,
    classify_runtime_failure,
    runtime_failure_codes,
)
from easyicu.research_agent.reporting.completion import step_completion_projection
from easyicu.research_agent.orchestration.workflow import HumanReviewPending, HumanReviewRequest
from easyicu.webserver.agent_pipeline_runs import _failed_closed_detail, _mark_nonconvergent_plan_revision
from easyicu.webserver.execution_retry import assess_failed_step
from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import STATIC


@pytest.mark.parametrize("diagnostic", [
    "TableOneContractError: A Planner-declared Table 1 group is empty",
    "ValueError: Planner-declared Table 1 groups are absent from exposure: [opaque]",
    "RuntimeError: risk_difference_contrast requires at least one analysed row in "
    "both the declared reference and comparison levels",
])
def test_executor_empty_levels_remain_typed_through_the_run_and_retry(diagnostic, tmp_path, monkeypatch):
    decision = classify_runtime_failure(
        run_log=diagnostic, timed_out=False, step_id="summary", returncode=1,
        deterministic_executor_used=True,
    )
    assert decision is not None
    assert decision.finding.detail["reason_code"] == DECLARED_EXPOSURE_LEVEL_EMPTY
    assert decision.step_updates["runtime_repair_route"] == "fail_closed"
    assert decision.step_updates["llm_repair_used"] is False
    record = {"step_id": "summary", **decision.step_updates}
    completion = step_completion_projection(
        required_step_ids=["summary"], record_by_step={"summary": record},
        status_by_step={"summary": record["status"]}, step_ok=lambda _: False,
    )
    assert completion["failed_steps"][0]["reason_code"] == DECLARED_EXPOSURE_LEVEL_EMPTY
    assert _failed_closed_detail({"execution_complete": False, **completion}) == {
        "reason_code": DECLARED_EXPOSURE_LEVEL_EMPTY,
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"per_step_records": [record]}))
    basis = retry_basis.load_failed_step_retry_basis(tmp_path, "summary")
    assert basis.failure_class == "typed_stop"
    assert basis.stop_reason_code == DECLARED_EXPOSURE_LEVEL_EMPTY
    assert basis.stop_repeats_on_unchanged_retry
    # Supply a synthetic unchanged identity, never inspect a Docker backend.
    identity = retry_basis.AttemptIdentity("a" * 64, "b" * 64, "c" * 64, None)
    monkeypatch.setattr("easyicu.webserver.execution_retry.current_attempt_identity", lambda **_: identity)
    assessment = assess_failed_step(replace(basis, identity=identity))
    assert assessment.state == "futile"
    assert assessment.reason_code == "execution_retry_repeats_typed_stop"


def test_unknown_reasons_and_successful_steps_do_not_gain_an_empty_level_stop():
    assert runtime_failure_codes({"runtime_failure_reason": DECLARED_EXPOSURE_LEVEL_EMPTY}) == {}
    assert runtime_failure_codes({
        "status": "ok", "runtime_failure_class": "plan_data_contract",
        "runtime_failure_reason": DECLARED_EXPOSURE_LEVEL_EMPTY,
    }) == {}
    assert classify_runtime_failure(
        run_log="RuntimeError: unrelated arithmetic failure", timed_out=False,
        step_id="summary", returncode=1, deterministic_executor_used=True,
    ) is None


def test_empty_level_copy_is_one_chinese_sentence_that_requests_a_new_plan():
    node = shutil.which("node")
    assert node
    script = r"""
      let owner;
      global.window = { EU_LANG: 'zh', EU_HTML: {esc: value => String(value)}, EasyICU: { guidedPi: {
        declare: (_name, api) => { owner = api; }, optional: () => null,
      } } };
      require(process.argv[1]);
      const copy = owner.create({tr: (_en, zh) => zh, staticPreview: () => false});
      process.stdout.write(copy.runFailureText('research_agent_pipeline_failed_closed', {
        code: 'declared_exposure_level_empty',
      }));
    """
    text = run_node(node, script, str(STATIC / "js/screens-guided-pi-error-text.js"), check=True).stdout
    assert text == "计划中的一个分组没有可分析的住院，因此运行已停止，请重新生成计划，检查入组条件并保留所有分组。"


def test_exhausted_population_repairs_stop_without_approval():
    code = "POPULATION_EMPTIES_EXPOSURE_LEVEL"
    pending = HumanReviewPending(
        run_id="synthetic-revision", thread_id="synthetic-thread", run_dir="synthetic",
        requests=(HumanReviewRequest.create(
            kind="scientific_stop", summary="Population and groups conflict.",
            authority_sha256="a" * 64,
            payload={"reason": "plan_scientific_changes_required"},
        ),),
    )
    stopped = _mark_nonconvergent_plan_revision(
        pending, source_codes=[code], current_review={
            "approval_allowed": False,
            "facts": {"remediation_buckets": {"agent_plan_revision": [code]}},
        },
    )
    payload = stopped.requests[0].payload
    assert payload["reason"] == "agent_plan_revision_nonconvergent"
    assert payload["approval_allowed"] is False
    assert payload["current_agent_plan_revision_codes"] == [code]


def test_an_emptied_declared_group_is_fixed_by_a_fresh_plan_not_a_retry():
    from easyicu.webserver import stop_next_actions as owner

    assert owner.STOP_POLICIES[DECLARED_EXPOSURE_LEVEL_EMPTY] == "edit_plan"
    for reason in (DECLARED_EXPOSURE_LEVEL_EMPTY, "research_agent_pipeline_failed_closed"):
        actions = owner.next_actions_for_stop(
            reason, {"reason_code": DECLARED_EXPOSURE_LEVEL_EMPTY}
        )
        assert [action.kind for action in actions] == ["edit_plan"]
