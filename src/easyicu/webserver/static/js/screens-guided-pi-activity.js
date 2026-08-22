/* Guided Copilot activity timeline.
   Owner: browser-safe lifecycle/tool rendering only. It never receives model
   reasoning, tool arguments, credentials, patient rows, or host paths. */
(function () {
  'use strict';

  const VISIBLE_KINDS = new Set([
    'submitted', 'agent', 'turn', 'assistant', 'tool', 'pipeline', 'retry', 'compaction',
  ]);

  function create(host) {
    const tr = host.tr;
    const esc = host.esc;
    const iconHtml = host.iconHtml;
    const resourceName = host.resourceName;
    const resourceKey = host.resourceKey;
    const resourceButton = host.resourceButton;

    function timeMs(value) {
      const parsed = Date.parse(String(value || ''));
      return Number.isFinite(parsed) ? parsed : Date.now();
    }
    function durationText(startedAt, endedAt) {
      const elapsed = Math.max(0, Number(endedAt || Date.now()) - Number(startedAt || Date.now()));
      if (elapsed < 1000) return `${elapsed} ms`;
      const seconds = elapsed / 1000;
      if (seconds < 60) {
        const value = seconds < 10 ? seconds.toFixed(1) : Math.round(seconds);
        return tr(`${value}s`, `${value} 秒`);
      }
      const minutes = Math.floor(seconds / 60);
      const remainder = Math.round(seconds % 60);
      return tr(
        remainder ? `${minutes}m ${remainder}s` : `${minutes}m`,
        remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分`,
      );
    }
    function stepDuration(step) {
      const started = Number(step && step.startedAt);
      const ended = Number(step && step.endedAt);
      return Number.isFinite(started) && Number.isFinite(ended) && ended >= started
        ? durationText(started, ended) : '';
    }
    function toolIcon(name) {
      const tool = String(name || '');
      if (/preview/.test(tool)) return 'globe';
      if (/read_project|write_project/.test(tool)) return 'file';
      if (/edit_project/.test(tool)) return 'edit';
      if (/check_project/.test(tool)) return 'check';
      if (/list_project/.test(tool)) return 'folder';
      if (/load_skill/.test(tool)) return 'wand';
      if (/update|replan/.test(tool)) return 'edit';
      if (/run$|extraction/.test(tool)) return 'play';
      if (/resume/.test(tool)) return 'refresh';
      if (/cancel/.test(tool)) return 'stop';
      if (/literature|evidence|validation|blocker|interpretation/.test(tool)) return 'shield';
      if (/workflow|manuscript|idea|artifact|plan|step/.test(tool)) return 'list';
      if (/workspace|capability/.test(tool)) return 'db';
      if (/context/.test(tool)) return 'file';
      return 'spark';
    }
    function activityIcon(step) {
      if (!step) return 'spark';
      if (step.kind === 'submitted') return 'arrow';
      if (step.kind === 'turn' || step.kind === 'retry') return 'refresh';
      if (step.kind === 'assistant') return 'wand';
      if (step.kind === 'tool') return toolIcon(step.toolName);
      if (step.kind === 'pipeline') {
        if (/artifact|report|manuscript/.test(step.step || '')) return 'file';
        if (/gate|valid|audit|evidence/.test(step.step || '')) return 'shield';
        if (/plan/.test(step.step || '')) return 'list';
        return 'play';
      }
      if (step.kind === 'compaction') return 'layers';
      if (step.kind === 'failed') return 'alert';
      if (step.kind === 'cancelled') return 'stop';
      if (step.kind === 'settled') return 'check';
      return 'spark';
    }
    function toolLabel(name, resource) {
      const labels = {
        easyicu_workspace_status: tr('Check workspace status', '检查工作区状态'),
        easyicu_list_data_sources: tr('List registered data sources', '列出已登记数据源'),
        easyicu_inspect_data_package: tr('Review data package', '审阅数据包'),
        easyicu_inspect_workflow: tr('Inspect research workflow', '检查科研流程'),
        easyicu_inspect_context: tr('Inspect study context', '读取研究配置'),
        easyicu_inspect_plan: tr('Inspect scientific plan', '读取科学计划'),
        easyicu_inspect_literature: tr('Inspect literature evidence', '读取文献证据'),
        easyicu_inspect_capability: tr('Inspect capabilities', '检查可用能力'),
        easyicu_inspect_run: tr('Inspect run status', '读取运行状态'),
        easyicu_inspect_step: tr('Inspect plan step', '读取计划步骤'),
        easyicu_inspect_validation: tr('Inspect validation', '读取验证状态'),
        easyicu_list_artifacts: tr('List run artefacts', '列出运行产物'),
        easyicu_inspect_evidence: tr('Inspect evidence', '读取证据状态'),
        easyicu_explain_blocker: tr('Explain blocker', '解释阻断原因'),
        easyicu_inspect_interpretation: tr('Interpret validated results', '解读已验证结果'),
        easyicu_inspect_manuscript: tr('Inspect manuscript draft', '读取论文草稿'),
        easyicu_update_study_context: tr('Save study setup', '保存研究配置'),
        easyicu_mine_ideas: tr('Mine research ideas', '发掘研究想法'),
        easyicu_search_literature: tr('Search PubMed literature', '检索 PubMed 文献'),
        easyicu_prepare_idea_handoff: tr('Prepare idea plan', '准备想法计划'),
        easyicu_accept_idea_handoff: tr('Accept selected idea', '接受所选想法'),
        easyicu_start_extraction: tr('Start feature extraction', '启动特征提取'),
        easyicu_run: tr('Start EasyICU run', '启动 EasyICU 运行'),
        easyicu_resume: tr('Resume EasyICU work', '恢复 EasyICU 任务'),
        easyicu_cancel: tr('Cancel EasyICU job', '取消 EasyICU 任务'),
        easyicu_request_replan: tr('Request replan', '请求重新规划'),
        easyicu_load_skill: tr('Load web-prototype skill', '加载网页原型技能'),
        easyicu_list_extensions: tr('List frozen extensions', '列出固化扩展'),
        easyicu_call_mcp_tool: tr('Call allowlisted MCP tool', '调用白名单 MCP 工具'),
        easyicu_list_project_files: tr('List project files', '列出项目文件'),
        easyicu_read_project_file: tr('Read project file', '读取项目文件'),
        easyicu_write_project_file: tr('Write project file', '写入项目文件'),
        easyicu_edit_project_file: tr('Edit project file', '编辑项目文件'),
        easyicu_check_project_file: tr('Check project file', '检查项目文件'),
        easyicu_preview_project_file: tr('Prepare web preview', '准备网页预览'),
      };
      const label = labels[String(name || '')] || String(name || tr('EasyICU tool', 'EasyICU 工具'));
      const file = resourceName(resource);
      return file ? `${label} · ${file}` : label;
    }
    function completedToolLabel(name, resource) {
      const labels = {
        easyicu_workspace_status: tr('Checked workspace status', '已检查工作区状态'),
        easyicu_list_data_sources: tr('Listed registered data sources', '已列出已登记数据源'),
        easyicu_inspect_data_package: tr('Reviewed data package', '已审阅数据包'),
        easyicu_inspect_workflow: tr('Read research workflow', '已读取科研流程'),
        easyicu_inspect_context: tr('Read study setup', '已读取研究配置'),
        easyicu_inspect_plan: tr('Read scientific plan', '已读取科学计划'),
        easyicu_inspect_literature: tr('Read literature evidence', '已读取文献证据'),
        easyicu_inspect_capability: tr('Checked capabilities', '已检查可用能力'),
        easyicu_inspect_run: tr('Read run status', '已读取运行状态'),
        easyicu_inspect_step: tr('Read plan step', '已读取计划步骤'),
        easyicu_inspect_validation: tr('Read validation', '已读取验证状态'),
        easyicu_list_artifacts: tr('Listed run artefacts', '已列出运行产物'),
        easyicu_inspect_evidence: tr('Read evidence', '已读取证据状态'),
        easyicu_explain_blocker: tr('Read blocker details', '已读取阻断原因'),
        easyicu_inspect_interpretation: tr('Organized evidence-bound interpretation', '已整理证据约束的结果解读'),
        easyicu_inspect_manuscript: tr('Read manuscript draft', '已读取论文草稿'),
        easyicu_update_study_context: tr('Saved study setup', '已保存研究配置'),
        easyicu_mine_ideas: tr('Mined research ideas', '已发掘研究想法'),
        easyicu_search_literature: tr('Searched PubMed literature', '已检索 PubMed 文献'),
        easyicu_prepare_idea_handoff: tr('Prepared idea plan', '已准备想法计划'),
        easyicu_accept_idea_handoff: tr('Accepted selected idea', '已接受所选想法'),
        easyicu_start_extraction: tr('Started feature extraction', '已启动特征提取'),
        easyicu_run: tr('Started EasyICU run', '已启动 EasyICU 运行'),
        easyicu_resume: tr('Resumed EasyICU work', '已恢复 EasyICU 任务'),
        easyicu_cancel: tr('Cancelled EasyICU job', '已取消 EasyICU 任务'),
        easyicu_request_replan: tr('Requested replan', '已请求重新规划'),
        easyicu_load_skill: tr('Loaded web-prototype skill', '已加载网页原型技能'),
        easyicu_list_extensions: tr('Listed frozen extensions', '已列出固化扩展'),
        easyicu_call_mcp_tool: tr('Called allowlisted MCP tool', '已调用白名单 MCP 工具'),
        easyicu_list_project_files: tr('Listed project files', '已列出项目文件'),
        easyicu_read_project_file: tr('Read project file', '已读取项目文件'),
        easyicu_write_project_file: tr('Wrote project file', '已写入项目文件'),
        easyicu_edit_project_file: tr('Edited project file', '已编辑项目文件'),
        easyicu_check_project_file: tr('Checked project file', '已检查项目文件'),
        easyicu_preview_project_file: tr('Prepared web preview', '已准备网页预览'),
      };
      const label = labels[String(name || '')] || tr(`Used ${toolLabel(name)}`, `已使用 ${toolLabel(name)}`);
      const file = resourceName(resource);
      return file ? `${label} · ${file}` : label;
    }
    function stepLabel(step) {
      const done = step.status === 'complete';
      const failed = step.status === 'error';
      if (step.kind === 'submitted') return tr('Message submitted to EasyICU Copilot', '消息已提交给 EasyICU 研究助手');
      if (step.kind === 'agent') return tr('Copilot workflow started', '研究助手工作流已启动');
      if (step.kind === 'turn') return done
        ? tr(`Model turn ${step.turn + 1} finished`, `模型回合 ${step.turn + 1} 已结束`)
        : tr(`Model turn ${step.turn + 1} is running`, `模型回合 ${step.turn + 1} 进行中`);
      if (step.kind === 'assistant') return done
        ? tr(`Model analysis and response phase ${step.phase} finished`, `模型分析与回复阶段 ${step.phase} 已完成`)
        : tr(`Model is analyzing and composing response phase ${step.phase}`, `模型正在分析并组织回复阶段 ${step.phase}`);
      if (step.kind === 'tool') return failed
        ? tr(`${toolLabel(step.toolName, step.resource)} returned an error`, `${toolLabel(step.toolName, step.resource)} 返回错误`)
        : done ? completedToolLabel(step.toolName, step.resource)
          : tr(`Calling ${toolLabel(step.toolName, step.resource)}`, `正在调用 ${toolLabel(step.toolName, step.resource)}`);
      if (step.kind === 'pipeline') return String(step.label || tr('EasyICU research pipeline updated', 'EasyICU 科研流程已更新'));
      if (step.kind === 'retry') {
        if (step.label) return String(step.label);
        if (step.attempt != null && step.maxAttempts != null) {
          return tr(`Retrying (${step.attempt}/${step.maxAttempts})`, `正在重试（${step.attempt}/${step.maxAttempts}）`);
        }
        return tr('Retrying after a failed attempt', '上一步未通过，正在重试');
      }
      if (step.kind === 'compaction') return done ? tr('Context compaction finished', '上下文整理已完成') : tr('Compacting context', '正在整理上下文');
      if (step.kind === 'cancelled') return tr('This turn was stopped', '本轮已停止');
      if (step.kind === 'failed') return tr('This turn failed', '本轮失败');
      if (step.kind === 'settled') return tr('This turn completed', '本轮已完成');
      return tr('Agent activity updated', 'Agent 状态已更新');
    }
    function stepPrimary(step) {
      const label = stepLabel(step);
      return step.resource ? resourceButton(step.resource, label) : `<strong>${esc(label)}</strong>`;
    }
    function stepResources(step) {
      const primary = resourceKey(step.resource);
      const resources = (Array.isArray(step.resources) ? step.resources : [])
        .filter(resource => resourceKey(resource) && resourceKey(resource) !== primary);
      if (!resources.length) return '';
      return `<div class="gpi-resource-list" aria-label="${tr('Run artifacts', '运行产物')}">${resources.map(resource => resourceButton(resource)).join('')}</div>`;
    }
    function stepRow(step) {
      const meta = [stepDuration(step), step.code, step.owner].filter(Boolean).join(' · ');
      return `<li class="${esc(step.status || 'complete')}">
        <span class="gpi-activity-step-icon" aria-hidden="true">${iconHtml(activityIcon(step), 15)}</span>
        <span class="gpi-activity-step-copy">${stepPrimary(step)}${step.text ? `<span>${esc(step.text)}</span>` : ''}${stepResources(step)}${meta ? `<small>${esc(meta)}</small>` : ''}</span>
        <span class="gpi-status-pip" aria-hidden="true"></span>
      </li>`;
    }
    function render(row) {
      const allSteps = Array.isArray(row && row.steps) ? row.steps : [];
      const visibleSteps = allSteps.filter(step => VISIBLE_KINDS.has(step.kind));
      const latest = visibleSteps[visibleSteps.length - 1] || allSteps[allSteps.length - 1];
      const running = row && row.status === 'running';
      const failed = row && (row.status === 'error' || row.status === 'cancelled');
      const kicker = tr('Activity', '执行明细');
      if (running) {
        const title = latest && latest.kind !== 'submitted'
          ? stepLabel(latest) : tr('EasyICU Copilot is preparing the next action', 'EasyICU 研究助手正在准备下一步');
        const liveSteps = visibleSteps.slice(-20);
        return `<div class="gpi-activity-running" role="status">
          <div class="gpi-activity-live">
            <span class="gpi-activity-glyph" aria-hidden="true">${iconHtml(activityIcon(latest), 15)}</span>
            <span class="gpi-activity-kicker">${esc(kicker)}</span>
            <span class="gpi-activity-title">${esc(title)}</span>
            <span class="gpi-status-pip" aria-hidden="true"></span>
          </div>
          ${liveSteps.length ? `<ol>${liveSteps.map(stepRow).join('')}</ol>` : ''}
        </div>`;
      }
      const toolSteps = visibleSteps.filter(step => step.kind === 'tool');
      const pipelineSteps = visibleSteps.filter(step => step.kind === 'pipeline');
      const completedTitle = row.displayTitle || (row.childJobId
        ? tr('EasyICU research task finished', 'EasyICU 科研任务已结束')
        : toolSteps.length === 1 ? stepLabel(toolSteps[0])
          : toolSteps.length === 2
            ? toolSteps.map(step => completedToolLabel(step.toolName, step.resource)).join(tr(' and ', '、'))
            : toolSteps.length > 2
              ? tr(`Used ${toolSteps.length} EasyICU tools`, `已使用 ${toolSteps.length} 个 EasyICU 工具`)
              : tr('Answered without using tools', '仅回答，未执行操作'));
      const title = failed ? tr('This turn needs attention', '本轮需要处理') : completedTitle;
      const open = failed || row.expanded === true ? 'open' : '';
      return `<details class="gpi-activity ${failed ? 'error' : 'complete'}" ${open}>
        <summary>
          <span class="gpi-activity-glyph" aria-hidden="true">${iconHtml(failed ? 'alert' : activityIcon(toolSteps[0] || pipelineSteps[0] || latest), 15)}</span>
          <span class="gpi-disclosure" aria-hidden="true">${iconHtml('chevron', 14)}</span>
          <span class="gpi-activity-kicker">${esc(kicker)}</span>
          <span class="gpi-activity-title">${esc(title)}</span>
          <span class="gpi-activity-meta">${esc(tr(`${visibleSteps.length} steps`, `${visibleSteps.length} 个步骤`))}${row.durationKnown === false ? '' : ` · ${esc(durationText(row.startedAt, row.endedAt))}`}</span>
        </summary>
        <div class="gpi-activity-body">
          ${visibleSteps.length ? `<ol>${visibleSteps.map(stepRow).join('')}</ol>` : ''}
          <p>${tr('Lifecycle facts and EasyICU receipts only — private chain-of-thought is never displayed.', '这里只显示生命周期事实和 EasyICU 回执，不展示模型的私有思维链。')}</p>
        </div>
      </details>`;
    }
    function focusLatest(rows) {
      const activities = rows.filter(row => row && row.role === 'activity');
      activities.forEach(row => { if (row.status !== 'running') row.expanded = false; });
      const latest = activities[activities.length - 1];
      if (latest && latest.status !== 'running') latest.expanded = true;
      return rows;
    }

    return Object.freeze({ focusLatest, render, stepLabel, timeMs });
  }

  window.EU_GUIDED_PI_ACTIVITY = Object.freeze({ create });
})();
