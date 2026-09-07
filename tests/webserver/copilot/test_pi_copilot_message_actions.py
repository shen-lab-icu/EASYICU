"""Conversation actions, governed plan retries, and in-place Pi revisions."""

from __future__ import annotations
import json
import shutil
import subprocess
import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    STATIC as STATIC,
    _ESCAPE_OWNER as _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)


def test_copilot_next_step_owner_projects_clickable_choices_and_safe_fallback() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const choices = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '已确认 MIMIC-IV v3.1。\\n**下一步：**\\n请选择研究单位。\\n' +
        '- 所有符合条件的 ICU stays\\n- 每位患者首次 ICU stay'
      );
      const fallback = window.EU_GUIDED_PI_NEXT_ACTIONS.project('成人是否定义为 **年龄 ≥18 岁？**');
      const generic = window.EU_GUIDED_PI_NEXT_ACTIONS.project('研究配置已经保存。');
      const inline = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '定义已保存。\\n**下一步：请选择主要结局：**\\n- ICU 内死亡\\n- 住院死亡'
      );
      const markdownHeading = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        'Demo 仅为样本。\\n### 下一步：\\n请选择一个选项：\\n' +
        '- 选择 MIMIC-IV Demo\\n- 选择 eICU Demo\\n- 继续无数据规划'
      );
      const localSource = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '数据源待确认。\\n**下一步：**\\n' +
        '- 使用推荐的 MIMIC-IV v3.1 数据导出\\n' +
        '- 选择并注册本机上的其他 MIMIC-IV v3.1 数据目录'
      );
      const databases = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '请选择数据库。\\n**下一步：**\\n' +
        '- 使用 MIMIC-IV v3.1\\n- 使用 eICU v2.0\\n' +
        '- 使用 AmsterdamUMCdb\\n- 使用 HiRID v1.1.1\\n' +
        '- 使用 MIMIC-III v1.4\\n- 使用 SICdb v1.0.6'
      );
      const compactDatabases = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '请选择一个确切的数据库版本。\\n**下一步：**\\n' +
        '-选择 MIMIC-IV v3.1\\n-选择 eICU v2.0\\n' +
        '-选择 AmsterdamUMCdb\\n-选择 HiRID v1.1.1\\n' +
        '-选择 MIMIC-III v1.4\\n-选择 SICdb v1.0.6'
      );
      const genericSix = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '请选择一项。\\n**下一步：**\\n' +
        '- 选项一\\n- 选项二\\n- 选项三\\n- 选项四\\n- 选项五\\n- 选项六'
      );
      const researchEntry = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '我先确认一下你的目标\\n\\n' +
        '你是想先发掘或评估这个方向，还是已经决定研究它并准备往下做？\\n' +
        '- 先发掘或评估可能方向\\n- 按当前问题进入研究方案\\n' +
        '- 我还不确定先帮我判断'
      );
      console.log(JSON.stringify({{choices, fallback, generic, inline, markdownHeading, databases, compactDatabases, genericSix, researchEntry}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(choices, {{language: 'zh'}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(
        {{body: '请选择研究单位。', prompt: '', choices: ['首次 ICU stay']}},
        {{language: 'zh'}}
      ));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(markdownHeading, {{language: 'zh'}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(localSource, {{language: 'zh'}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(databases, {{language: 'zh'}}));
      console.log('COMPACT=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(compactDatabases, {{language: 'zh'}}));
      console.log('ENTRY=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(researchEntry, {{language: 'zh'}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render({{
        body: '待确认的研究设定',
        prompt: '请选择一项：',
        choices: [
          '我确认采用成人 ICU 人群、ICU stay 分析单位、Sepsis-3 主要暴露、ICU 住院期间死亡结局和入科后 24 小时外层特征窗，并授权 EasyICU 准备数据。',
          '我想修改或增加一项关键研究要求。',
        ],
      }}, {{language: 'zh'}}));
      console.log('RESOLVED=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(localSource, {{
        language: 'zh',
        dataSourceAuthorization: {{
          status: 'confirmed',
          source: {{label: 'MIMIC-IV', reference_release: '3.1'}},
        }},
      }}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(generic, {{language: 'zh'}}));
      console.log('SUPPRESSED=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(
        generic,
        {{language: 'zh', suppressFallback: true}}
      ));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert '"body":"已确认 MIMIC-IV v3.1。"' in completed.stdout
    assert '"choices":["所有符合条件的 ICU stays","每位患者首次 ICU stay"]' in completed.stdout
    assert '"prompt":"请选择主要结局："' in completed.stdout
    assert '"prompt":"请选择主要结局：**"' not in completed.stdout
    assert '"explicit":false' in completed.stdout
    assert '"asking":false' in completed.stdout
    assert 'data-gpi-next-choice="所有符合条件的 ICU stays"' in completed.stdout
    assert '"choices":["选择 MIMIC-IV Demo","选择 eICU Demo","继续无数据规划"]' in completed.stdout
    assert 'data-gpi-next-choice="确认并授权本轮准备并注册 MIMIC-IV Demo。"' in completed.stdout
    assert "下载并准备 MIMIC-IV Demo" in completed.stdout
    assert 'data-gpi-next-choice="继续无数据规划"' in completed.stdout
    assert 'data-gpi-next-local-database="miiv"' in completed.stdout
    assert '"choices":["使用 MIMIC-IV v3.1","使用 eICU v2.0","使用 AmsterdamUMCdb","使用 HiRID v1.1.1","使用 MIMIC-III v1.4","使用 SICdb v1.0.6"]' in completed.stdout
    assert 'data-gpi-next-choice="使用 MIMIC-III v1.4"' in completed.stdout
    assert 'data-gpi-next-choice="使用 SICdb v1.0.6"' in completed.stdout
    assert '"compactDatabases":{"body":"请选择一个确切的数据库版本。","prompt":"","choices":["选择 MIMIC-IV v3.1","选择 eICU v2.0","选择 AmsterdamUMCdb","选择 HiRID v1.1.1","选择 MIMIC-III v1.4","选择 SICdb v1.0.6"]' in completed.stdout
    assert 'COMPACT=<section class="gpi-next-step"' in completed.stdout
    assert 'data-gpi-next-choice="选择 MIMIC-IV v3.1"' in completed.stdout
    assert '"researchEntry":{"body":"我先确认一下你的目标' in completed.stdout
    assert '"prompt":"请选择一种开始方式。"' in completed.stdout
    assert 'ENTRY=<section class="gpi-next-step"' in completed.stdout
    assert 'data-gpi-next-choice="按当前问题进入研究方案"' in completed.stdout
    assert '"choices":["选项一","选项二","选项三","选项四"]' in completed.stdout
    assert '"选项五"' not in completed.stdout
    assert 'data-gpi-next-resolved-source' in completed.stdout
    assert 'data-gpi-data-source-continue' in completed.stdout
    assert '已接入 MIMIC-IV v3.1' in completed.stdout
    assert '尚未开始数据提取或分析' in completed.stdout
    assert '查看数据准备确认' in completed.stdout
    assert '不会在这里生成研究计划' in completed.stdout
    assert '>确认并准备数据<' in completed.stdout
    assert 'data-gpi-next-choice="我确认采用成人 ICU 人群、ICU stay 分析单位、Sepsis-3 主要暴露、ICU 住院期间死亡结局和入科后 24 小时外层特征窗，并授权 EasyICU 准备数据。"' in completed.stdout
    assert "EasyICU 会直接执行对应操作或继续对话" in completed.stdout
    assert "其他，我自己输入" in completed.stdout
    assert 'class="gpi-next-custom"' in completed.stdout
    assert 'data-gpi-next-custom-form' in completed.stdout
    assert 'data-gpi-next-custom-input' in completed.stdout
    assert 'placeholder="输入其他选择或补充说明"' in completed.stdout
    assert '<button type="submit"' in completed.stdout
    assert "继续对话" in completed.stdout
    assert "SUPPRESSED=" in completed.stdout
    assert "SUPPRESSED=<" not in completed.stdout
    assert "<script>" not in completed.stdout


def test_planner_confirmation_owns_the_only_next_actions() -> None:
    screen = _read("js/screens-guided-pi.js")
    owner = _read("js/screens-guided-pi-next-actions.js")
    assert "provider_ready_to_generate_plan" in screen
    assert "workflowActionCode:" in screen
    assert "suppressFallback:" in screen
    assert "options.suppressFallback && !step.explicit" in owner
    assert "data-gpi-premature-plan-guard" in owner


def test_model_plan_choice_is_replaced_until_workflow_owner_is_ready() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const projected = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '研究问题已明确。\\n**下一步：**\\n' +
        '- 生成正式研究计划，由 EasyICU 说明依据\\n' +
        '- 先补充研究要求'
      );
      const authorization = {{status: 'confirmed', source: {{label: 'MIMIC-IV'}}}};
      console.log('BLOCKED=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(projected, {{
        language: 'zh', dataSourceAuthorization: authorization,
        workflowActionCode: 'study_setup_incomplete',
      }}));
      console.log('READY=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(projected, {{
        language: 'zh', dataSourceAuthorization: authorization,
        workflowActionCode: 'provider_ready_to_generate_plan',
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    blocked, ready = completed.stdout.split("READY=", 1)
    assert "data-gpi-premature-plan-guard" in blocked
    assert "查看数据准备确认" in blocked
    assert "生成正式研究计划" not in blocked
    assert "data-gpi-premature-plan-guard" not in ready
    assert "生成正式研究计划" in ready


def test_model_plan_choice_can_only_receive_provider_grant_from_plan_workflow() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const grants = window.EU_GUIDED_PI_NEXT_ACTIONS.governedPlanGrants;
      console.log(JSON.stringify(grants(
        '授权基于当前 E2 StudyContext 生成新的正式研究计划，并在分析前暂停审核',
        'plan_configuration_superseded'
      )));
      console.log(JSON.stringify(grants(
        '按科学审阅要求生成新正式研究计划并在分析前暂停审核',
        'plan_scientific_changes_required'
      )));
      console.log(JSON.stringify(grants(
        '在 EasyICU 主机确认一次性 provider_run 授权已注入当前回合后，重新发送修订请求',
        'plan_scientific_changes_required'
      )));
      console.log(JSON.stringify(grants(
        '暂不生成修订版正式研究计划',
        'plan_scientific_changes_required'
      )));
      console.log(JSON.stringify(grants(
        '每位患者仅保留第一次 ICU 入院，请按这一规则修订并重新提交候选研究计划；分析保持暂停。',
        'plan_scientific_changes_required'
      )));
      console.log(JSON.stringify(grants(
        '保留全部 ICU 入院，并在模型中处理同一患者的重复记录；请按这一规则修订并重新提交候选研究计划，分析保持暂停。',
        'plan_scientific_changes_required'
      )));
      console.log(JSON.stringify(grants(
        '授权基于当前 E2 StudyContext 生成新的正式研究计划，并在分析前暂停审核',
        'study_setup_incomplete'
      )));
      console.log(JSON.stringify(grants(
        '在主机授权卡中启用一次性 provider_run grant',
        'plan_configuration_superseded'
      )));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert completed.stdout.splitlines() == [
        '["provider_run"]',
        '["provider_run"]',
        '["configure","provider_run"]',
        "[]",
        '["configure","provider_run"]',
        '["configure","provider_run"]',
        "[]",
        "[]",
    ]


def test_scientific_plan_revision_requests_a_fresh_governed_plan() -> None:
    guided = _read("js/screens-guided-pi.js")
    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    tool_owner = (STATIC.parent / "pi_copilot" / "tools.py").read_text(
        encoding="utf-8"
    )

    assert "'plan_scientific_changes_required'," in plan_actions
    assert "confirm_fresh_plan_generation" in plan_actions
    assert "plan_revision_source_run_id: revisionSourceRunId" in plan_actions
    assert "literature_search_authorized: true" in plan_actions
    assert "revisingScientificPlan" in plan_actions
    assert "continueSystemOwnedPlanProgression" in plan_actions
    assert "continueUserRequestedSystemProgression" in plan_actions
    assert "const startedTransitions = new Set()" in plan_actions
    assert "function transitionKey(reasonCode)" in plan_actions
    assert "continueSystemOwnedPlanProgression" in guided
    assert "await PLAN_ACTIONS.continueUserRequestedSystemProgression(text)" in guided
    assert "plan_revision_source_run_id" not in guided
    assert 'planner_start_mode=strategy' in tool_owner
    assert 'fresh_run_required = bool(same_study_plan and not current_review_is_resumable)' in tool_owner


def test_demo_next_step_is_one_click_and_supports_an_existing_local_copy() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const projected = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '官方 Demo 尚未准备。\\n**下一步：**\\n' +
        '- 授权下载并准备官方 MIMIC-IV Demo\\n' +
        '- 暂不下载，继续完善研究设计'
      );
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(projected, {{ language: 'zh' }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert 'data-gpi-next-grants="extract"' in completed.stdout
    assert "确认并授权本轮准备并注册" in completed.stdout
    assert "使用已经下载好的 MIMIC-IV Demo" in completed.stdout
    assert 'data-gpi-next-local-database="miiv"' in completed.stdout
    assert "暂不下载，继续完善研究设计" in completed.stdout


def test_recommended_prepared_export_choice_receives_configuration_grant() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const projected = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '请选择数据来源。\\n**下一步：**\\n' +
        '- 使用推荐的 EasyICU 本地导出（MIMIC-IV v3.1，94,458 个 ICU stay，19 个模块）\\n' +
        '- 选择并注册其他本地 MIMIC-IV v3.1 数据目录'
      );
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(projected, {{ language: 'zh' }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert 'data-gpi-next-grants="configure"' in completed.stdout
    assert 'data-gpi-next-local-database="miiv"' in completed.stdout


def test_preview_plan_confirmation_routes_to_data_preparation_not_same_plan_runner() -> None:
    shell = _read("js/screens-guided-pi.js")
    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    confirmation = _read("js/screens-guided-pi-confirmation.js")

    branch = plan_actions.split(
        "if (confirmation.code === 'plan_execution_upgrade_required')", 1
    )[1].split("if (confirmation.code === 'failed_pipeline_execution_retry_available'", 1)[0]
    assert "startFormalPlanGeneration(confirmation.code)" in branch
    assert "host.sendText(confirmation.message, confirmation.grants)" not in branch
    assert "candidate_plan_only: true" not in shell
    assert "const projectedAllowlist = new Set(['extract', 'configure']);" in plan_actions
    assert "['configure', 'extract', 'provider_run', 'literature']" in confirmation
    assert "候选研究计划已生成，请确认数据准备" in confirmation
    assert "并不代表数据包已经准备好" in confirmation
    assert "确认方案并准备数据" in confirmation
    assert "计划已从候选变量中选出明确的调整集" in confirmation
    assert "确认候选计划会先将其建议的基线变量保存到新的 StudyContext 修订版" in confirmation
    assert "任务已经启动，但本次对话回执未能保存" not in shell
    assert "EasyICU could not persist the host-action receipt." in shell


def test_copilot_public_projection_hides_internal_pi_codes_and_owner_paths() -> None:
    owner = _read("js/screens-guided-pi.js")
    activity = _read("js/screens-guided-pi-activity.js")

    assert "function publicAssistantText" in owner
    assert "pi_action_authorization_required" in owner
    assert "EasyICU 内部状态" in owner
    assert "本轮一次性数据准备授权" in owner
    assert "官方 Demo 准备流程" in owner
    assert "const meta = stepDuration(step);" in activity
    assert "step.code, step.owner" not in activity
    assert "easyicu_prepare_demo_source: tr('Download and prepare official demo data'" in activity


def test_copilot_message_owner_wires_only_latest_next_step_to_send_or_focus() -> None:
    owner = _read("js/screens-guided-pi.js")
    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    events = _read("js/screens-guided-pi-events.js")
    data_binding_owner = _read("js/screens-guided-pi-data-binding.js")
    css = _read("css/guided-pi.css")

    assert "MODULES.require('nextActions')" in owner
    assert "row.complete !== false" in owner
    assert "row === latestAssistant && !interactionLocked && !stale" in owner
    assert "sendText(message, governedNextChoiceGrants(nextChoice, message))" in events
    assert "function governedNextChoiceGrants(element, message)" in plan_actions
    assert "nextActions.governedPlanGrants(message, workflowCode())" in plan_actions
    assert "event.target.closest('[data-gpi-next-focus]')" in events
    assert "event.target.closest('[data-gpi-next-custom-form]')" in events
    # The free-text box goes through the same governed grant decision as a
    # choice button. It used to send `[]`, which stripped the turn's grants and
    # made "其他，我自己输入" fail authorization where the button beside it
    # succeeded.
    assert "sendText(message, governedNextChoiceGrants(null, message))" in events
    assert "sendText(message, [])" not in events
    assert "dataSourceAuthorization: DATA_CONSENT && DATA_CONSENT.authorization(state.session)" in owner
    assert "event.target.closest('[data-gpi-data-source-continue]')" in events
    assert "continueAfterDataSourceConfirmation()" in events
    continuation = owner.split("async function continueAfterDataSourceConfirmation()", 1)[1].split(
        "async function sendMessage()", 1
    )[0]
    assert "await sendText(" in continuation
    assert "'advance_after_data_source_confirmation'" in continuation
    assert "false," in continuation
    assert "regenerateMessage(" not in continuation
    assert owner.index("showProjectContinuationCards ? dataConsentHtml") > owner.index(
        "data-gpi-log"
    )
    assert "if (await continueAfterDataSourceConfirmation()) return true;" in data_binding_owner
    next_actions = _read("js/screens-guided-pi-next-actions.js")
    assert "查看数据准备确认" in next_actions
    assert "只会先汇总数据准备所需的最少信息" in next_actions
    assert "不会在这里生成研究计划" in next_actions
    assert "不必等待全量数据提取" in next_actions
    assert "继续到下一个关键决策" not in next_actions
    assert ".gpi-next-step" in css
    assert ".gpi-next-actions" in css
    assert ".gpi-next-custom{flex:1 0 100%" in css
    assert ".gpi-next-custom-row input{min-width:0;flex:1" in css
    for non_owner in ("css/guided.css", "css/guided-projects.css", "css/guided-startup.css"):
        assert ".gpi-next-custom" not in _read(non_owner)


def test_successful_local_source_action_opens_native_workspace_immediately() -> None:
    owner = _read("js/screens-guided-pi.js")
    events = _read("js/screens-guided-pi-events.js")

    assert "event.code || '') === 'easyicu_local_source_workspace_ready'" in owner
    assert "resource && resource.kind === 'native_workspace'" in owner
    assert "preview.open(localWorkspace, projectId())" in owner
    assert "nextChoice.dataset.gpiNextLocalDatabase" in events
    assert "authorizeDataSource('begin_local_selection', { database: localDatabase })" in events


def test_copilot_message_action_owner_renders_copy_edit_and_latest_retry() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-message-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const actions = window.EU_GUIDED_PI_MESSAGE_ACTIONS.create({{
        tr: (en, zh) => zh,
        iconHtml: name => `<i>${{name}}</i>`,
      }});
      console.log(actions.render(
        {{id: 'u1', role: 'user', text: '原问题', complete: true}},
        {{allowEdit: true, canEdit: true}}
      ).actionsHtml);
      console.log(actions.render(
        {{id: 'u1', role: 'user', text: '原问题', complete: true}},
        {{editing: true, allowEdit: true, canEdit: true}}
      ).editorHtml);
      console.log(actions.render(
        {{id: 'a1', role: 'assistant', text: '回答', complete: true}},
        {{canRetry: true, retryUserEntryId: 'entry-u1'}}
      ).actionsHtml);
      console.log(actions.render(
        {{id: 'a0', role: 'assistant', text: '旧回答', complete: true}},
        {{canRetry: false, retryUserEntryId: 'entry-u0'}}
      ).actionsHtml);
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert "data-gpi-message-copy" in completed.stdout
    assert "data-gpi-message-edit" in completed.stdout
    assert "这条之后的回答会被替换；原分支仍保留在历史中可恢复。" in completed.stdout
    assert completed.stdout.count("data-gpi-message-retry") == 1


def test_editing_an_earlier_turn_rewinds_instead_of_appending() -> None:
    """Editing message N must replace what follows N, not append at the bottom.

    Regression: the edit submit called sendText, so an edited earlier turn was
    posted as a brand-new message while the original answers stayed below it.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-message-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const calls = [];
      const rows = [
        {{id: 'u1', role: 'user', text: '第一个问题', entryId: 'entry-u1', complete: true}},
        {{id: 'a1', role: 'assistant', text: '第一个回答', complete: true}},
        {{id: 'u2', role: 'user', text: '第二个问题', entryId: 'entry-u2', complete: true}},
        {{id: 'a2', role: 'assistant', text: '第二个回答', complete: true}},
      ];
      const actions = window.EU_GUIDED_PI_MESSAGE_ACTIONS.create({{
        tr: (en, zh) => zh,
        iconHtml: name => `<i>${{name}}</i>`,
        rows: () => rows,
        setEditing: () => {{}},
        sendText: value => calls.push(['sendText', value]),
        regenerate: (entryId, text, intent, target) =>
          calls.push(['regenerate', entryId, text, intent, target]),
      }});
      const form = {{
        dataset: {{ gpiMessageEditForm: 'u1' }},
        querySelector: () => ({{ value: '  改过的第一个问题  ' }}),
      }};
      actions.handleSubmit({{
        preventDefault: () => {{}},
        target: {{ closest: sel => (sel === '[data-gpi-message-edit-form]' ? form : null) }},
      }});
      console.log(JSON.stringify(calls));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    calls = json.loads(completed.stdout)
    # It rewinds to the edited turn with the new text, targeting that turn's
    # own answer -- it does not append a fresh message at the bottom.
    assert calls == [
        ["regenerate", "entry-u1", "改过的第一个问题", "user_edited_message", "a1"]
    ]


def test_editing_host_generated_plan_action_resubmits_without_appending() -> None:
    """A reloaded host plan action keeps direct routing despite a Pi entry id."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-message-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const calls = [];
      const row = {{
        id: 'history-9', entryId: 'entry-plan-9', role: 'user',
        text: '重新生成研究计划', complete: true,
      }};
      const actions = window.EU_GUIDED_PI_MESSAGE_ACTIONS.create({{
        tr: (en, zh) => zh,
        iconHtml: name => `<i>${{name}}</i>`,
        rows: () => [row],
        setEditing: () => {{}},
        sendText: value => calls.push(['sendText', value]),
        regenerate: (...args) => calls.push(['regenerate', ...args]),
        resubmitHostGenerated: (target, text) => {{
          calls.push(['resubmitHostGenerated', target.id, text]);
          return true;
        }},
      }});
      const form = {{
        dataset: {{ gpiMessageEditForm: 'history-9' }},
        querySelector: () => ({{ value: '  重新生成研究计划  ' }}),
      }};
      actions.handleSubmit({{
        preventDefault: () => {{}},
        target: {{ closest: sel => (sel === '[data-gpi-message-edit-form]' ? form : null) }},
      }});
      console.log(JSON.stringify(calls));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert json.loads(completed.stdout) == [
        ["resubmitHostGenerated", "history-9", "重新生成研究计划"]
    ]


def test_retrying_candidate_plan_action_uses_governed_host_starter() -> None:
    """Assistant retry must not replay a plan action as ordinary chat."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-message-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const calls = [];
      const rows = [
        {{
          id: 'history-plan', entryId: 'entry-plan', role: 'user',
          text: '生成候选研究计划', complete: true,
        }},
        {{id: 'history-answer', role: 'assistant', text: '旧回答', complete: true}},
      ];
      const actions = window.EU_GUIDED_PI_MESSAGE_ACTIONS.create({{
        tr: (en, zh) => zh,
        iconHtml: name => `<i>${{name}}</i>`,
        rows: () => rows,
        host: () => null,
        canEdit: () => true,
        setEditing: () => {{}},
        renderHost: () => {{}},
        regenerate: (...args) => calls.push(['regenerate', ...args]),
        resubmitHostGenerated: (target, text) => {{
          calls.push(['resubmitHostGenerated', target.id, text]);
          return true;
        }},
      }});
      const retry = {{
        closest: selector => selector === '[data-gpi-message-retry]'
          ? retry
          : selector === '[data-gpi-message-id]'
            ? {{dataset: {{gpiMessageId: 'history-answer'}}}}
            : null,
      }};
      actions.handleClick({{target: retry}});
      console.log(JSON.stringify(calls));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert json.loads(completed.stdout) == [
        ["resubmitHostGenerated", "history-plan", "生成候选研究计划"]
    ]


def test_copilot_message_actions_are_host_wired_without_history_rewrite() -> None:
    owner = _read("js/screens-guided-pi.js")
    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    events = _read("js/screens-guided-pi-events.js")
    css = _read("css/guided-pi.css")

    assert "MODULES.require('messageActions').create" in owner
    assert "state.editingMessageId === row.id" in owner
    assert "MESSAGE_ACTIONS.handleClick(event)" in events
    assert "MESSAGE_ACTIONS.handleSubmit(event)" in events
    message_owner = _read("js/screens-guided-pi-message-actions.js")
    assert "copyText(row.text)" in message_owner
    # Editing an earlier turn rewinds to it, so the replies after it are
    # replaced instead of a new message being appended at the bottom.  The Pi
    # branch keeps the original recoverable, so this is not a history rewrite.
    assert (
        "context.regenerate(entryId, text, 'user_edited_message', followingAssistantId(id))"
        in message_owner
    )
    assert "context.regenerate(user.entryId, user.text, '', articleId(retry))" in message_owner
    # A host-generated action is replaced in place; other edits with no
    # resolvable turn identifier must still not vanish.
    assert "context.resubmitHostGenerated(row, text)" in message_owner
    assert "context.sendText(text);" in message_owner
    assert "resubmitHostGenerated: PLAN_ACTIONS.resubmitHostGenerated" in owner
    assert "regeneratePiCopilotMessage" in owner
    assert "user_entry_id: entryId" in owner
    assert "advance_after_data_source_confirmation" in owner
    assert "dataSourceContinuationTarget" not in owner
    assert "host.turnGrants().filter(action => action === 'configure')" in plan_actions
    assert "nextActions.governedPlanGrants(text, code)" in plan_actions
    assert "'replace_plan_response_preserve_study'" in plan_actions
    assert "REPLAY_GRANT_INTENTS" in plan_actions
    assert "planGrants.includes('provider_run')" in plan_actions
    assert "message: text, allowed_actions: authority.grants" in owner
    assert "turn_intent: authority.intent" in owner
    assert "regeneration_intent: regenerationIntent" in owner
    assert "interactive: row === latestAssistant && !interactionLocked && !stale" in owner
    assert ".gpi-message.user .gpi-message-actions{right:0;opacity:0}" in css
    assert ".gpi-message-editor" in css


def test_candidate_plan_can_be_explicitly_regenerated_before_data_preparation() -> None:
    """The review/data-prep gate must allow replacing, not executing, a plan."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value || '') }} }};
      eval({json.dumps(source)});
      const grants = window.EU_GUIDED_PI_NEXT_ACTIONS.governedPlanGrants(
        '重新生成研究计划', 'plan_execution_upgrade_required'
      );
      console.log(JSON.stringify(grants));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == ["provider_run"]


def test_failed_analysis_still_allows_an_explicit_fresh_plan_request() -> None:
    """Execution retry remains default while an explicit plan redo is governed."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value || '') }} }};
      eval({json.dumps(source)});
      console.log(JSON.stringify(
        window.EU_GUIDED_PI_NEXT_ACTIONS.governedPlanGrants(
          '重新生成研究计划', 'failed_pipeline_execution_retry_available'
        )
      ));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == ["provider_run"]

    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    assert "code === 'failed_pipeline_execution_retry_available'" in plan_actions
    assert "? 'failed_pipeline_requires_fresh_plan'" in plan_actions
    assert "startFormalPlanGeneration(" in plan_actions
    confirmation = _read("js/screens-guided-pi-confirmation.js")
    assert "tr('Generate a fresh research plan', '重新生成研究计划')" in confirmation


def test_failed_pipeline_never_renders_partial_cohort_decision() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-cohort-eligibility.js")
    script = f"""
      global.window = {{}};
      eval({owner!r});
      const render = window.EU_GUIDED_PI_COHORT_ELIGIBILITY.create({{
        tr: (_en, zh) => zh,
        esc: value => String(value ?? ''),
        workflow: () => ({{next_action_code: 'failed_pipeline_requires_fresh_plan'}}),
        session: () => ({{cohort_eligibility_selection: {{
          present: true,
          stated: false,
          blocker_code: 'cohort_eligibility_confirmation_required',
          options: [{{id: 'adults_all_admissions'}}],
          primary_cohort_contract: {{admission_eligibility: {{minimum_age_years: 18}}}},
        }}}}),
        busy: () => false,
        sessionIsStale: () => false,
      }}).render;
      process.stdout.write(render());
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert completed.stdout == ""


def test_failed_analysis_retry_submits_exact_resume_without_chat_roundtrip() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-replay.js")
    script = f"""
      global.window = global;
      eval({json.dumps(source)});
      let submitted = null;
      const api = {{
        loadStudyContext: async id => ({{context: {{
          id, question: 'E2 question',
          data_source: {{path: '/prepared/miiv', database: 'miiv'}},
        }}}}),
        startAgentRun: async body => {{ submitted = body; return {{job_id: 'retry-job'}}; }},
      }};
      window.EU_GUIDED_PI_REPLAY.retryFailedExecution({{
        api,
        session: {{
          binding: {{study_context_id: 'study-e2', run_id: 'run-e2'}},
          research_provider: {{provider: 'openai', credential_source: 'pi_verified'}},
        }},
      }}).then(result => console.log(JSON.stringify({{result, submitted}})));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload["result"]["job_id"] == "retry-job"
    assert payload["submitted"] == {
        "path": "/prepared/miiv",
        "study_id": "study-e2",
        "study_context_id": "study-e2",
        "question": "E2 question",
        "run_type": "full",
        "llm_provider": "openai",
        "credential_source": "pi_verified",
        "external_llm_opt_in": True,
        "engine": "research_agent_pipeline",
        "planner_start_mode": "auto",
        "execution_resume_source_run_id": "run-e2",
    }

    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    assert "await retryFailedExecution();" in plan_actions
    assert "host.sendText(confirmation.message" not in plan_actions.split(
        "confirmation.code === 'failed_pipeline_execution_retry_available'", 1
    )[1].split("if (FRESH_PLAN_CODES", 1)[0]


def test_copilot_regeneration_projects_activity_and_answer_in_place() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    source = _read("js/screens-guided-pi-regeneration.js")
    script = f"""
      global.window = {{}};
      eval({json.dumps(source)});
      const owner = window.EU_GUIDED_PI_REGENERATION;
      const rows = [
        {{id: 'u1', role: 'user', text: 'question', entryId: 'entry-u1'}},
        {{id: 'activity-1', role: 'activity', status: 'complete'}},
        {{id: 'a1', role: 'assistant', text: 'old answer', complete: true}},
        {{id: 'u2', role: 'user', text: 'later question', entryId: 'entry-u2'}},
        {{id: 'activity-2', role: 'activity', status: 'complete'}},
        {{id: 'a2', role: 'assistant', text: 'later answer', complete: true}},
      ];
      const replacement = owner.create(rows, {{
        userEntryId: 'entry-u1', targetMessageId: 'a1', startedAt: 1234,
      }});
      console.log(JSON.stringify({{
        targetMessageId: replacement.targetMessageId,
        targetActivityId: replacement.targetActivityId,
        projectedRoles: owner.visibleRows(rows, replacement).map(row => owner.project(row, replacement).role),
        projectedTexts: owner.visibleRows(rows, replacement).map(row => owner.project(row, replacement).text || ''),
        projectedStatuses: owner.visibleRows(rows, replacement).map(row => owner.project(row, replacement).status || ''),
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {
        "targetMessageId": "a1",
        "targetActivityId": "activity-1",
        "projectedRoles": ["user", "activity", "assistant"],
        "projectedTexts": ["question", "", ""],
        "projectedStatuses": ["", "running", ""],
    }
    owner = _read("js/screens-guided-pi.js")
    assert "REGENERATION.visibleRows(fullTimeline, state.regeneration)" in owner
    assert "REGENERATION.project(row, state.regeneration)" in owner
    assert "state.regeneration.message" in owner
    assert "state.regeneration.activity" in owner


def test_plan_retry_appends_a_short_host_owned_action() -> None:
    """Plan buttons submit the governed run without a model round trip."""

    shell = _read("js/screens-guided-pi.js")
    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    starter = plan_actions.split("async function startFormalPlanGeneration", 1)[1].split(
        "function governedNextChoiceGrants", 1
    )[0]
    assert "text: request.text" in starter
    assert "startAgentRun" in starter
    assert "latestPlanRequest" not in starter
    assert "replace_plan_response_preserve_study" not in starter
    assert "regenerateMessage(" not in starter
    assert "planner_start_mode: reasonCode === 'planner_checkpoint_resume_available'" in starter
    assert "? 'resume_checkpoint'" in starter
    assert ": 'fresh'" in starter
    assert "candidate_plan_only" not in starter

    # Explicit message editing still owns its separate branch-replacement
    # contract; removing replay from workflow buttons must not weaken it.
    replay = plan_actions.split("function regenerationAuthority", 1)[1].split(
        "function resubmitHostGenerated", 1
    )[0]
    assert "'replace_plan_response_preserve_study'" in plan_actions
    assert "nextActions.governedPlanGrants(text, code)" in replay
    assert "...planGrants" in replay
    assert "planGrants.includes('provider_run')" in replay

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    regeneration = _read("js/screens-guided-pi-regeneration.js")
    script = f"""
      global.window = {{}};
      eval({json.dumps(regeneration)});
      const rows = [
        {{id: 'u-data', role: 'user', text: 'use prepared data', entryId: 'entry-data'}},
        {{id: 'a-data', role: 'assistant', text: 'data bound'}},
        {{id: 'u-plan', role: 'user', text: '生成候选研究计划', entryId: 'entry-plan'}},
        {{id: 'a-plan', role: 'assistant', text: 'old plan receipt'}},
        {{id: 'u-later', role: 'user', text: 'later recovery message', entryId: 'entry-later'}},
        {{id: 'a-later', role: 'assistant', text: 'later answer'}},
      ];
      const receiptHiddenRows = rows.filter(row => row.id !== 'a-plan');
      console.log(JSON.stringify({{
        withReceipt: window.EU_GUIDED_PI_REGENERATION.latestPlanRequest(rows),
        withoutReceipt: window.EU_GUIDED_PI_REGENERATION.latestPlanRequest(receiptHiddenRows),
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {
        "withReceipt": {
            "userEntryId": "entry-plan",
            "targetMessageId": "a-plan",
        },
        "withoutReceipt": {
            "userEntryId": "entry-plan",
            "targetMessageId": "",
        },
    }


def test_agent_plan_runtime_configuration_is_compiled_without_prompt_or_user_turn() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    actions_source = _read("js/screens-guided-pi-plan-actions.js")
    script = f"""
      global.window = {{}};
      eval({actions_source!r});
      const calls = [];
      let busy = false;
      const session = {{
        session_id: 'session-agent-plan',
        binding: {{
          run_id: 'run-agent-plan',
          study_context_id: 'study-agent-plan',
          study_revision: 11,
        }},
        research_provider: {{provider: 'openai', credential_source: 'pi_verified'}},
      }};
      const host = {{
        tr: (en, zh) => zh,
        errorText: error => String(error && error.message || error),
        regeneration: {{}}, nextActions: {{}}, replay: {{}},
        session: () => session,
        workflow: () => ({{next_action_code: 'agent_plan_configuration_required'}}),
        busy: () => busy,
        sessionIsStale: () => false,
        api: () => ({{
          applyPiCopilotAgentPlanConfiguration: async (sessionId, body) => {{
            calls.push(['compile', sessionId, body]);
            return {{next_action: 'fresh_plan'}};
          }},
          loadStudyContext: async id => ({{context: {{
            id,
            question: 'Natural clinical question only',
            data_source: {{path: '/prepared/miiv'}},
          }}}}),
          startAgentRun: async body => {{
            calls.push(['plan', body]);
            return {{job_id: 'fresh-agent-plan'}};
          }},
        }}),
        projectId: () => 'project-agent-plan', turnGrants: () => [],
        render: () => {{}},
        recordHostAction: async (...args) => calls.push(['host-action', ...args]),
        watchChildJob: (...args) => calls.push(['child', ...args]),
        refreshSession: async () => calls.push(['refresh']),
        loadWorkflow: async () => calls.push(['workflow']),
        setBusy: value => {{busy = value;}},
        setError: value => calls.push(['error', value]),
        appendMessage: value => calls.push(['message', value.text]),
      }};
      const actions = window.EU_GUIDED_PI_PLAN_ACTIONS.create(host);
      actions.continueSystemOwnedPlanProgression()
        .then(() => process.stdout.write(JSON.stringify(calls)));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    calls = json.loads(completed.stdout)

    assert not any(row[0] in {"message", "send"} for row in calls)
    compile_call = next(row for row in calls if row[0] == "compile")
    assert compile_call[2] == {
        "project_id": "project-agent-plan",
        "expected_revision": 11,
        "run_id": "run-agent-plan",
    }
    plan_call = next(row for row in calls if row[0] == "plan")
    assert plan_call[1]["question"] == "Natural clinical question only"
    assert plan_call[1]["plan_revision_source_run_id"] == ""
