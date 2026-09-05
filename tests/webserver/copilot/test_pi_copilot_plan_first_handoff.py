from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


STATIC = Path(__file__).parents[3] / "src" / "easyicu" / "webserver" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


MODULES_SOURCE = _read("js/screens-guided-pi-modules.js")


def test_pre_data_planner_is_presented_as_a_candidate_with_the_next_steps() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")

    owner = _read("js/screens-guided-pi-confirmation.js")
    script = f"""
      global.window = {{}};
      eval({MODULES_SOURCE!r});
      eval({owner!r});
      const host = {{
        tr: (en, zh) => zh || en,
        esc: value => String(value),
        iconHtml: () => '',
        resourceButton: () => '',
        sessionIsStale: () => false,
        workflow: () => ({{ next_action_code: 'provider_ready_to_generate_plan' }}),
        session: () => ({{ archived_child_jobs: [] }}),
        busy: () => false,
      }};
      const confirmation = window.EasyICU.guidedPi.require('confirmation').create(host);
      const spec = confirmation.workflowConfirmation();
      process.stdout.write(JSON.stringify({{
        title: spec.title,
        message: spec.message,
        note: spec.note,
        steps: spec.flowSteps,
        flowTitle: spec.flowTitle,
        flowHint: spec.flowHint,
        flowCurrent: spec.flowCurrent,
        html: confirmation.workflowConfirmationHtml(),
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    rendered = json.loads(completed.stdout)

    assert rendered["title"] == "现在生成候选研究计划吗？"
    assert rendered["message"] == "开始生成候选研究计划。"
    assert "计划会先决定需要哪些数据" in rendered["note"]
    assert rendered["steps"] == [
        "生成候选计划",
        "按计划准备或复用数据",
        "审阅数据准备情况",
        "审核可执行计划并开始分析",
    ]
    assert rendered["flowTitle"] == "接下来会发生什么"
    assert rendered["flowHint"] == "流程预览 · 现在无需操作"
    assert rendered["flowCurrent"] == "当前：等待生成候选计划"
    assert 'class="gpi-confirmation-flow-overview"' in rendered["html"]
    assert 'class="gpi-confirmation-flow"' in rendered["html"]
    assert 'class="is-current"' in rendered["html"]


def test_plan_first_copy_stays_in_the_guided_copilot_owner() -> None:
    confirmation = _read("js/screens-guided-pi-confirmation.js")
    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    shell = _read("js/screens-guided-pi.js")
    regeneration = _read("js/screens-guided-pi-regeneration.js")
    child_job = _read("js/screens-guided-pi-childjob.js")
    css = _read("css/guided-pi.css")

    assert "计划与分析前数据检查已准备好" in confirmation
    assert "生成候选研究计划" in plan_actions
    assert "生成候选研究计划" not in shell
    # Legacy persisted action text remains replayable in the dedicated branch
    # classifier rather than inflating the main Copilot screen owner.
    assert "生成正式研究计划" in regeneration
    assert "生成正式研究计划" not in shell
    assert "正在生成研究计划" in child_job
    assert "正在生成正式研究计划" not in child_job
    assert ".gpi-confirmation-flow" in css
    assert ".gpi-confirmation-resources.is-expanded" in css
    assert ".gpi-confirmation-more" in css
    assert ".gpi-plan-conversation" in css
    assert ".gpi-plan-conversation-more" in css
    assert "按计划准备或复用数据" in confirmation
    for non_owner in ("css/agent.css", "css/agent-plan.css", "css/guided.css"):
        assert ".gpi-confirmation-flow" not in _read(non_owner)
        assert ".gpi-confirmation-more" not in _read(non_owner)
        assert ".gpi-plan-conversation" not in _read(non_owner)
        assert ".gpi-plan-conversation-more" not in _read(non_owner)


def test_executable_plan_review_is_expanded_and_has_two_primary_choices() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")

    owner = _read("js/screens-guided-pi-confirmation.js")
    script = f"""
      global.window = {{}};
      eval({MODULES_SOURCE!r});
      eval({owner!r});
      const host = {{
        tr: (en, zh) => zh || en,
        esc: value => String(value),
        iconHtml: () => '',
        resourceButton: resource => `<button class="gpi-resource-link">${{resource.label}}</button>`,
        sessionIsStale: () => false,
        workflow: () => ({{
          next_action_code: 'operator_plan_approval_required',
          plan_review_summary: {{ run_id: 'run-test' }},
        }}),
        session: () => ({{
          archived_child_jobs: [],
          binding: {{ run_id: 'run-test' }},
          data_source_authorization: {{
            extraction_scope: 'study_required',
            source: {{ database: 'MIMIC-IV' }},
          }},
        }}),
        busy: () => false,
      }};
      const confirmation = window.EasyICU.guidedPi.require('confirmation').create(host);
      process.stdout.write(confirmation.workflowConfirmationHtml());
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    rendered = completed.stdout

    assert "计划与分析前数据检查已准备好" in rendered
    assert "快速审阅" in rendered
    assert "研究计划" in rendered
    assert "数据准备检查" in rendered
    assert "文献依据" in rendered
    assert "最终分析队列、数据预处理和模型仍待执行" in rendered
    assert 'class="gpi-confirmation-resources is-expanded"' in rendered
    assert '<details class="gpi-confirmation-resources">' not in rendered
    assert "修改计划" in rendered
    assert "批准并开始分析" in rendered
    assert "其他操作" in rendered
    assert rendered.count('class="btn ') == 2


def test_executable_plan_review_keeps_plan_details_in_one_disclosure() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")

    owner = _read("js/screens-guided-pi-confirmation.js")
    script = f"""
      global.window = {{}};
      eval({MODULES_SOURCE!r});
      eval({owner!r});
      const workflow = {{
        next_action_code: 'operator_plan_approval_required',
        plan_conversation_preview: {{
          step_count: 9,
          table_count: 4,
          figure_count: 3,
          items: [
            {{ key: 'population_and_unit', text: '纳入符合条件的 ICU 住院。' }},
            {{ key: 'exposure_and_timing', text: '使用预设时间窗内的乳酸。' }},
            {{ key: 'outcome_and_followup', text: '结局为院内死亡。' }},
            {{ key: 'adjustment_and_model', text: '运行预设校正模型。' }},
            {{ key: 'missing_data', text: '报告缺失并按计划处理。' }},
            {{ key: 'sensitivity_and_feasibility', text: '分析前检查覆盖率并运行敏感性分析。' }},
          ],
        }},
      }};
      const host = {{
        tr: (en, zh) => zh || en,
        esc: value => String(value),
        iconHtml: () => '',
        resourceButton: resource => `<button>${{resource.label}}</button>`,
        sessionIsStale: () => false,
        workflow: () => workflow,
        session: () => ({{
          archived_child_jobs: [], binding: {{ run_id: 'run-test' }},
          data_source_authorization: {{ extraction_scope: 'study_required', source: {{ database: 'MIMIC-IV' }} }},
        }}),
        busy: () => false,
      }};
      const confirmation = window.EasyICU.guidedPi.require('confirmation').create(host);
      process.stdout.write(confirmation.workflowConfirmationHtml());
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    rendered = completed.stdout

    assert "我已经根据你的研究问题生成了一份候选计划" in rendered
    assert "目前还没有开始分析" in rendered
    assert "研究人群与分析单位" in rendered
    assert "暴露定义与时间窗" in rendered
    assert "查看 6 项候选计划摘要" in rendered
    assert '<details class="gpi-plan-conversation-summary">' in rendered
    assert "缺失数据处理" in rendered
    assert "9 个步骤 · 4 张表 · 3 张图" in rendered
    assert rendered.index('class="gpi-plan-conversation"') < rendered.index(
        'class="gpi-confirmation'
    )


def test_scientific_plan_review_separates_summary_from_complete_evidence() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")

    owner = _read("js/screens-guided-pi-confirmation.js")
    script = f"""
      global.window = {{}};
      eval({MODULES_SOURCE!r});
      eval({owner!r});
      const workflow = {{
        next_action_code: 'plan_scientific_changes_required',
        plan_review_summary: {{ run_id: 'run-test', authorization_questions: [], remediation_buckets: {{ agent_plan_revision: [], external_evidence: [], independent_review: [] }} }},
        plan_conversation_preview: {{ items: [{{ key: 'population_and_unit', text: '纳入符合条件的 ICU 住院。' }}] }},
      }};
      const host = {{
        tr: (en, zh) => zh || en,
        esc: value => String(value),
        iconHtml: () => '',
        resourceButton: resource => `<button>${{resource.label}}</button>`,
        sessionIsStale: () => false,
        workflow: () => workflow,
        session: () => ({{ archived_child_jobs: [], binding: {{ run_id: 'run-test' }} }}),
        busy: () => false,
      }};
      const confirmation = window.EasyICU.guidedPi.require('confirmation').create(host);
      process.stdout.write(confirmation.workflowConfirmationHtml());
    """
    rendered = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    ).stdout

    assert "我已经根据你的研究问题生成了一份候选计划" in rendered
    assert '<details class="gpi-plan-conversation-summary">' in rendered
    assert "查看 1 项候选计划摘要" in rendered
    assert '<details class="gpi-confirmation-resources">' in rendered
    assert 'class="gpi-confirmation-resources is-expanded"' not in rendered
    assert "打开完整计划" in rendered


@pytest.mark.parametrize("archived_authorization_question", [False, True])
def test_repeated_stay_runtime_gap_stays_blocked_without_a_method_question(
    archived_authorization_question,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")

    owner = _read("js/screens-guided-pi-confirmation.js")
    questions = (
        [{"code": "REPEATED_STAY_IDENTITY_UNAVAILABLE"}]
        if archived_authorization_question
        else []
    )
    script = f"""
      global.window = {{}};
      eval({MODULES_SOURCE!r});
      eval({owner!r});
      const workflow = {{
        next_action_code: 'plan_scientific_changes_required',
        plan_review_summary: {{
          run_id: 'run-test',
          authorization_questions: {json.dumps(questions)},
          remediation_buckets: {{
            agent_plan_revision: [],
            runtime_capability: ['REPEATED_STAY_IDENTITY_UNAVAILABLE'],
            external_evidence: [],
            independent_review: [],
          }},
        }},
        plan_conversation_preview: {{ items: [{{ key: 'population_and_unit', text: '纳入 ICU 住院。' }}] }},
      }};
      const host = {{
        tr: (en, zh) => zh || en,
        esc: value => String(value).replaceAll('&', '&amp;').replaceAll('"', '&quot;'),
        iconHtml: () => '',
        resourceButton: resource => `<button>${{resource.label}}</button>`,
        sessionIsStale: () => false,
        workflow: () => workflow,
        session: () => ({{ archived_child_jobs: [], binding: {{ run_id: 'run-test' }} }}),
        busy: () => false,
      }};
      const confirmation = window.EasyICU.guidedPi.require('confirmation').create(host);
      process.stdout.write(JSON.stringify({{
        spec: confirmation.workflowConfirmation(),
        html: confirmation.workflowConfirmationHtml(),
      }}));
    """
    result = json.loads(
        subprocess.run(
            [node, "--eval", script], check=True, capture_output=True, text=True
        ).stdout
    )
    rendered = result["html"]

    assert result["spec"]["nonApprovable"] is True
    assert "暂不能执行" in rendered
    assert "1 项运行时合同仍被阻断" in rendered
    assert "不需要再次回答科学设定问题" in rendered
    assert "数据合同或独立证据缺口会保持阻断" in rendered
    assert "打开完整计划" in rendered
    assert "查看审阅详情" in rendered
    assert "请选择重复 ICU 入住的研究目标" not in rendered
    assert 'data-gpi-plan-decision-option=' not in rendered
    assert 'data-gpi-confirm-action' not in rendered
    assert 'data-gpi-confirm-edit' not in rendered
    assert 'class="gpi-plan-conversation"' in rendered
    assert '<details class="gpi-plan-conversation-summary">' in rendered
