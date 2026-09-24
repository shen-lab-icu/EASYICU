"""A failed approved execution offers a retry only when it could change."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.execution_retry import ExecutionRetryAssessment
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
)
from tests.webserver.copilot.research_workflow_fixtures import complete_study

STATIC = Path(__file__).parents[3] / "src" / "easyicu" / "webserver" / "static"
NODE_APP = STATIC.parent / "pi_copilot" / "node_app" / "src"
FUTILE = {
    "state": "futile",
    "reason_code": "execution_retry_repeats_failure",
    "failed_step_id": "figure",
    "repair_attempts": 2,
    "repair_limit": 2,
    "changed_components": [],
    "image_checked": True,
}


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def _failed_execution(study: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "run-execution-failed",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "gate_reason": "research_agent_pipeline_failed_closed",
        "run_status": "blocked",
        "scientific_configuration_sha256": (
            study_context_owner.scientific_configuration_sha256(dict(study))
        ),
        "artifact_names": [
            "agent_plan.json",
            "evidence_ledger.json",
            "source_run_manifest.json",
        ],
    }


def _snapshot(
    execution_retry: Optional[Mapping[str, Any]], latest_run: Optional[dict] = None
):
    study = complete_study()
    return build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run=latest_run or _failed_execution(study),
        execution_retry=execution_retry,
    )


def test_a_futile_assessment_replaces_the_retry_with_its_explanation() -> None:
    snapshot = _snapshot(FUTILE)

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.next_action_code == "failed_pipeline_execution_retry_futile"
    assert by_id["plan"].reason_code == "failed_pipeline_execution_retry_futile"
    assert dict(snapshot.execution_retry or {}) == FUTILE


@pytest.mark.parametrize("state", ["available", "unknown"])
def test_only_a_proven_repeat_withdraws_the_retry(state: str) -> None:
    snapshot = _snapshot(dict(FUTILE, state=state))

    assert snapshot.next_action_code == "failed_pipeline_execution_retry_available"


def test_an_assessment_never_reaches_a_run_that_is_not_a_failed_execution() -> None:
    study = complete_study()
    planning_failure = dict(
        _failed_execution(study), gate_reason="research_pipeline_plan_contract_exhausted"
    )

    snapshot = _snapshot(FUTILE, latest_run=planning_failure)

    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"
    assert snapshot.execution_retry is None


def _project(monkeypatch: pytest.MonkeyPatch, row: dict, answer: Any) -> Any:
    from easyicu.webserver.pi_copilot import workflow as owner

    monkeypatch.setattr(owner.sources, "load_registry", lambda: {})
    monkeypatch.setattr(owner, "registered_export_matches_study", lambda *_args: True)
    monkeypatch.setattr(owner, "list_bound_run_history", lambda **_kwargs: [row])
    monkeypatch.setattr(owner.agent_pipeline_runs, "pending_review", lambda _run_id: None)
    monkeypatch.setattr(owner.agent_runs, "read_run_review", lambda _project_dir: {})
    monkeypatch.setattr(
        owner.agent_pipeline_runs, "execution_retry_assessment", answer
    )
    study = complete_study()
    return owner.build_project_workflow_projection(
        study_context_id=study["id"], study_override=study,
    ).workflow


def test_the_project_projection_asks_the_resume_owner_for_the_failed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[dict[str, Any]] = []

    def answer(**kwargs: Any) -> ExecutionRetryAssessment:
        asked.append(kwargs)
        return ExecutionRetryAssessment(
            "futile", "execution_retry_repeats_failure", failed_step_id="figure",
            repair_attempts=2, repair_limit=2, image_checked=True,
        )

    workflow = _project(monkeypatch, _failed_execution(complete_study()), answer)

    assert workflow.next_action_code == "failed_pipeline_execution_retry_futile"
    assert [call["source_run_id"] for call in asked] == ["run-execution-failed"]
    assert "max_age_seconds" not in asked[0]


def test_no_other_state_pays_for_the_assessment(monkeypatch: pytest.MonkeyPatch) -> None:
    study = complete_study()
    planning_failure = dict(
        _failed_execution(study), gate_reason="research_pipeline_plan_contract_exhausted"
    )

    def answer(**_kwargs: Any) -> ExecutionRetryAssessment:
        pytest.fail("only a failed approved execution is assessed")

    workflow = _project(monkeypatch, planning_failure, answer)

    assert workflow.next_action_code == "failed_pipeline_requires_fresh_plan"


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    return node


def test_the_futile_card_explains_the_spent_budget_and_offers_only_a_fresh_plan() -> None:
    script = f"""
      global.window = {{}};
      eval({_read("js/screens-guided-pi-confirmation.js")!r});
      const confirmation = window.EU_GUIDED_PI_CONFIRMATION.create({{
        tr: (en, zh) => zh || en, esc: value => String(value), iconHtml: () => '',
        resourceButton: () => '', sessionIsStale: () => false, busy: () => false,
        workflow: () => ({{
          next_action_code: 'failed_pipeline_execution_retry_futile',
          execution_retry: {json.dumps(FUTILE)},
        }}),
        session: () => ({{ archived_child_jobs: [] }}),
      }});
      const spec = confirmation.workflowConfirmation();
      process.stdout.write(JSON.stringify({{
        spec, html: confirmation.workflowConfirmationHtml(),
      }}));
    """
    rendered = json.loads(
        subprocess.run([_node(), "--eval", script], check=True, capture_output=True, text=True).stdout
    )

    spec = rendered["spec"]
    assert spec["title"] == "从失败步骤重试只会重复同样的失败"
    assert "代码修复 2/2 次" in spec["note"]
    assert spec["approve"] == "重新生成研究计划"
    assert spec["hideEdit"] is True
    assert "data-gpi-confirm-edit" not in rendered["html"]
    assert "从失败步骤重试分析" not in rendered["html"]


def test_approving_the_futile_card_starts_a_fresh_plan_bound_to_the_failed_run() -> None:
    script = f"""
      global.window = {{}};
      eval({_read("js/screens-guided-pi-plan-actions.js")!r});
      let submitted = null;
      const actions = window.EU_GUIDED_PI_PLAN_ACTIONS.create({{
        tr: (en, zh) => en, projectId: () => 'project',
        session: () => ({{session_id: 'session', binding: {{run_id: 'failed-approved', study_context_id: 'study'}},
          research_provider: {{provider: 'openai'}}}}),
        workflow: () => ({{next_action_code: 'failed_pipeline_execution_retry_futile'}}),
        busy: () => false, sessionIsStale: () => false,
        nextActions: {{governedPlanGrants: () => ['provider_run']}},
        api: () => ({{
          loadStudyContext: async () => ({{context: {{question: 'question', data_source: {{path: '/prepared'}}}}}}),
          startAgentRun: async value => {{submitted = value; return {{job_id: 'new-plan'}};}},
        }}),
        appendMessage: () => {{}}, setBusy: () => {{}}, render: () => {{}},
        setError: () => {{}}, errorText: error => String(error),
        recordHostAction: async () => {{}}, watchChildJob: () => {{}},
      }});
      actions.confirmWorkflow({{code: 'failed_pipeline_execution_retry_futile'}})
        .then(() => console.log(JSON.stringify(submitted)));
    """
    submitted = json.loads(
        subprocess.run([_node(), "--eval", script], check=True, capture_output=True, text=True).stdout
    )

    assert submitted["plan_revision_source_run_id"] == "failed-approved"
    assert submitted["planner_start_mode"] == "auto"
    assert "execution_resume_source_run_id" not in submitted


def test_an_explicit_replan_request_stays_governed_in_the_futile_state() -> None:
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value || '') }} }};
      eval({_read("js/screens-guided-pi-next-actions.js")!r});
      console.log(JSON.stringify(window.EU_GUIDED_PI_NEXT_ACTIONS.governedPlanGrants(
        '重新生成研究计划', 'failed_pipeline_execution_retry_futile')));
    """
    grants = json.loads(
        subprocess.run([_node(), "--eval", script], check=True, capture_output=True, text=True).stdout
    )

    assert grants == ["provider_run"]


def test_the_copilot_prompt_treats_the_futile_state_as_plan_lifecycle() -> None:
    source = (NODE_APP / "main.mjs").read_text(encoding="utf-8")
    lifecycle = source.split("const PLAN_LIFECYCLE_WORKFLOW_CODES = new Set([", 1)[1]
    lifecycle = lifecycle.split("]);", 1)[0]

    assert '"failed_pipeline_execution_retry_available"' in lifecycle
    assert '"failed_pipeline_execution_retry_futile"' in lifecycle
