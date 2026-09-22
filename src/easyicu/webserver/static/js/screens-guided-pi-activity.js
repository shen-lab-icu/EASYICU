/* Guided Copilot activity timeline.
   Owner: browser-safe lifecycle/tool rendering. It shows the model's own
   reasoning summary as trace rows (bounded and sanitized upstream) and never
   receives tool arguments, credentials, patient rows, or host paths. */
(function () {
  'use strict';

  // Match the conversation contract used by research workspaces: the trace is
  // a record of actions a researcher can understand and inspect. Transport
  // acknowledgements, agent startup, model phases, and context maintenance
  // remain in the persisted receipt, but do not become fake "work" rows.
  const VISIBLE_KINDS = new Set(['tool', 'pipeline', 'retry', 'thinking']);
  const DURATION_KINDS = new Set(['assistant', 'tool', 'pipeline', 'retry', 'compaction', 'thinking']);

  function create(host) {
    const tr = host.tr;
    const esc = host.esc;
    const iconHtml = host.iconHtml;
    const resourceName = host.resourceName;
    const resourceKey = host.resourceKey;
    const resourceButton = host.resourceButton;
    const publicText = typeof host.publicText === 'function' ? host.publicText : value => String(value || '');
    let liveClock = 0;
    let liveClockRoot = null;

    function timeMs(value) {
      const parsed = Date.parse(String(value || ''));
      return Number.isFinite(parsed) ? parsed : Date.now();
    }
    function durationText(startedAt, endedAt) {
      const elapsed = Math.max(0, Number(endedAt || Date.now()) - Number(startedAt || Date.now()));
      if (elapsed < 100) return tr('<0.1s', '<0.1 秒');
      const seconds = elapsed / 1000;
      if (seconds < 60) {
        const value = seconds.toFixed(1);
        return tr(`${value}s`, `${value} 秒`);
      }
      const roundedSeconds = Math.round(seconds);
      const minutes = Math.floor(roundedSeconds / 60);
      const remainder = roundedSeconds % 60;
      return tr(
        remainder ? `${minutes}m ${remainder}s` : `${minutes}m`,
        remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分`,
      );
    }
    function updateLiveClock() {
      if (!liveClockRoot || !liveClockRoot.isConnected) {
        if (liveClock) clearInterval(liveClock);
        liveClock = 0;
        liveClockRoot = null;
        return;
      }
      liveClockRoot.querySelectorAll('[data-gpi-live-elapsed]').forEach(node => {
        node.textContent = durationText(Number(node.dataset.gpiLiveElapsed), Date.now());
      });
    }
    function syncLiveClock(root, running) {
      liveClockRoot = root || null;
      updateLiveClock();
      if (running && !liveClock) liveClock = setInterval(updateLiveClock, 250);
      if (!running && liveClock) {
        clearInterval(liveClock);
        liveClock = 0;
      }
    }
    function appendPublicDelta(activity, delta) {
      if (!activity || !delta) return;
      const step = activity.steps.slice().reverse()
        .find(item => item.kind === 'assistant' && item.status === 'running');
      if (!step) return;
      const addition = String(delta);
      step.publicChars = Number(step.publicChars || 0) + addition.length;
      step.publicText = (String(step.publicText || '') + addition).slice(-320);
    }
    function startTurn(activity, at) {
      if (!activity) return;
      const turn = activity.steps.filter(item => item.kind === 'turn').length;
      activity.steps.push({ id: `turn-${turn}`, kind: 'turn', turn, status: 'running', at, startedAt: at });
    }
    function finishTurn(activity, at) {
      if (!activity) return;
      const turn = activity.steps.slice().reverse()
        .find(item => item.kind === 'turn' && item.status === 'running');
      if (turn) { turn.status = 'complete'; turn.endedAt = at; }
    }
    /* The reasoning summary arrives as short bold headlines separated by
       blank lines. The first headline names the row; the whole summary is
       shown under it in the model's own words, whatever language they are in. */
    function reasoningHeadline(text) {
      const value = publicText(String(text || ''));
      const bold = value.match(/\*\*([^*\n]{1,160})\*\*/);
      if (bold) return bold[1].trim();
      const line = value.split(/\n+/).map(item => item.trim()).find(Boolean) || '';
      return line.replace(/[*_`#>]/g, '').trim().slice(0, 160);
    }
    /* The row label already carries the first headline, so the body starts
       after it; a summary that is only a headline renders as a single line. */
    function reasoningHtml(text) {
      const value = publicText(String(text || '')).trim();
      if (!value) return '';
      const paragraphs = value.split(/\n{2,}/).map(paragraph => paragraph.trim()).filter(Boolean);
      const headline = reasoningHeadline(text);
      if (paragraphs.length && headline && paragraphs[0].replace(/\*\*/g, '').trim() === headline) paragraphs.shift();
      return paragraphs.map(paragraph => {
        const safe = esc(paragraph).replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
        return `<p>${safe}</p>`;
      }).join('');
    }
    function stepDuration(step) {
      if (step && step.durationKnown === false) return '';
      if (!DURATION_KINDS.has(String(step && step.kind || ''))) return '';
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
      if (step.kind === 'thinking') return 'spark';
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
        easyicu_list_data_sources: tr('Check the EasyICU data-source catalog', '检查 EasyICU 数据源目录'),
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
        easyicu_prepare_demo_source: tr('Download and prepare official demo data', '下载并准备官方 Demo 数据'),
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
        easyicu_list_data_sources: tr('Checked the EasyICU data-source catalog', '已检查 EasyICU 数据源目录'),
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
        easyicu_prepare_demo_source: tr('Started official demo preparation', '已启动官方 Demo 准备'),
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
    /* Research Agent emits diagnostic owner steps such as provider, runtime,
       context, and audit. They remain available in the persisted job receipt,
       but the conversation projects them into researcher-facing stages. */
    function pipelineStage(value) {
      const step = String(value || '').toLowerCase();
      if (step === 'submitted') return 'submitted';
      if (['provider', 'research_pipeline', 'run', 'runtime'].includes(step)) return 'setup';
      if (['data_foundation', 'cohort', 'context', 'audit', 'extraction', 'data'].includes(step)) return 'inputs';
      if (['hypothesis', 'literature', 'evidence'].includes(step)) return 'evidence';
      if (['planning', 'plan', 'scientific_review'].includes(step)) return 'plan';
      if (['step', 'coder', 'runner', 'analysis'].includes(step)) return 'analysis';
      if (['figure', 'visual_qa'].includes(step)) return 'figure';
      if (['writer', 'latex', 'manuscript', 'report'].includes(step)) return 'report';
      if (step === 'terminal') return 'terminal';
      if (['event_stream', 'cancel_requested'].includes(step)) return 'attention';
      return 'progress';
    }

    function pipelineStageLabel(stage, status, fallback) {
      const done = status === 'complete';
      if (stage === 'setup') return done
        ? tr('Research-task setup is ready', '研究计划生成环境已准备')
        : tr('Preparing the research-task setup', '正在准备研究计划生成环境');
      if (stage === 'inputs') return done
        ? tr('Research question, data scope, and study context checked', '已核对研究问题、数据范围与研究上下文')
        : tr('Checking the research question, data scope, and study context', '正在核对研究问题、数据范围与研究上下文');
      if (stage === 'evidence') return done
        ? tr('Planning evidence and hypotheses organized', '已整理计划所需的研究依据与假设')
        : tr('Organizing planning evidence and hypotheses', '正在整理计划所需的研究依据与假设');
      if (stage === 'plan') return done
        ? tr('Candidate research plan generated and contract-checked', '候选研究计划已生成并完成结构校验')
        : tr('Generating and checking the candidate research plan', '正在生成并校验候选研究计划');
      if (stage === 'analysis') return done
        ? tr('Approved analysis steps completed', '已完成批准的分析步骤')
        : tr('Running the approved analysis steps', '正在执行批准的分析步骤');
      if (stage === 'figure') return done
        ? tr('Figure generation and checks finished', '图件生成与检查流程已结束')
        : tr('Regenerating and checking the analysis figures', '正在重新生成并核查分析图件');
      if (stage === 'report') return done
        ? tr('Manuscript generation and checks finished; see the result verdict', '稿件生成与检查流程已结束；是否通过请看结果审阅')
        : tr('Regenerating the evidence-bound article and manuscript exports', '正在重新生成证据绑定文章与稿件导出');
      if (stage === 'progress') return done
        ? tr('Research-task progress updated', '研究任务进度已更新')
        : tr('Research task is progressing', '研究任务正在推进');
      return String(fallback || tr('Research-task status updated', '研究任务状态已更新'));
    }

    /* Planning events reach the browser as job progress rows whose typed
       `planning_unit` / `retry_phase` fields may be absent (only the label
       survives the job projection). Recover both from the label so the tally
       and the row text do not depend on which transport carried the event. */
    function planningEventFacts(event) {
      const message = String(event && (event.message || event.label) || '').toLowerCase();
      let unit = String(event && event.planning_unit || '');
      let phase = String(event && event.retry_phase || '');
      // Only the progressive planner's own labels are recovered here; the
      // legacy "plan draft n/m" sentences keep their dedicated wording below.
      if (!unit && !phase) {
        if (message.includes('study structure')) unit = 'structure';
        else if (message.includes('cohort and analysis rules')) unit = 'rules';
        else if (message.includes('executable plan step')) unit = 'step';
        if (unit) {
          if (message.includes('passed validation')) phase = 'accepted';
          else if (message.includes('did not satisfy') || message.includes('retrying')) phase = 'rejected';
          else if (message.includes('validation attempt')) phase = 'started';
        }
      }
      return { unit, phase };
    }

    function pipelineEventLabel(event) {
      const type = String((event && event.type) || '');
      if (type === 'start') return tr('EasyICU research pipeline started', 'EasyICU 科研流程已启动');
      if (type === 'cancel_requested') return tr('Cancellation requested', '已请求取消任务');
      const stage = pipelineStage(event && event.step);
      if (stage === 'plan') {
        const current = Number(event && event.current);
        const total = Number(event && event.total);
        const count = Number.isFinite(current) && Number.isFinite(total)
          ? `${current}/${total}` : '';
        const facts = planningEventFacts(event);
        const planningUnit = facts.unit || 'plan';
        const retryPhase = facts.phase;
        const unit = {
          structure: tr('study structure', '研究结构'),
          rules: tr('cohort and analysis rules', '队列与分析规则'),
          step: tr('current executable step', '当前可执行步骤'),
          plan: tr('candidate plan', '候选计划'),
        }[planningUnit] || tr('candidate plan', '候选计划');
        if (retryPhase === 'rejected') {
          return tr(
            `${unit} ${count || 'attempt'} did not pass validation; EasyICU is correcting it automatically`,
            `${unit} ${count || '本次'} 未通过校验；EasyICU 正在自动修正`,
          );
        }
        if (retryPhase === 'accepted') {
          return tr(
            `${unit} ${count || ''} passed validation`.trim(),
            `${unit} ${count || ''} 已通过校验`.trim(),
          );
        }
        if (retryPhase === 'started') {
          return tr(
            `Validating ${unit} ${count || ''}`.trim(),
            `正在校验${unit} ${count || ''}`.trim(),
          );
        }
        const message = String(event && (event.message || event.label) || '').toLowerCase();
        if (message.includes('did not satisfy the scientific contract')) {
          return tr(
            `Candidate plan ${count || 'draft'} did not pass the scientific contract; EasyICU is revising it`,
            `候选计划 ${count || '草案'} 未通过科学合同；EasyICU 正在自动修订`,
          );
        }
        if (message.includes('passed contract validation')) {
          return tr(
            `Candidate plan ${count || 'draft'} passed contract validation`,
            `候选计划 ${count || '草案'} 已通过科学合同校验`,
          );
        }
        if (message.includes('generating plan draft')) {
          return tr(
            `Generating candidate plan ${count || 'draft'}`,
            `正在生成候选计划 ${count || '草案'}`,
          );
        }
      }
      return pipelineStageLabel(stage, String(event && event.status || 'running'));
    }

    /* Live tally for the collapsed plan row: what the planner has already
       validated and which validation is running now. */
    function planningProgressText(progress) {
      if (!progress || typeof progress !== 'object') return '';
      const passed = [];
      if (progress.structurePassed) passed.push(tr('study structure', '研究结构'));
      if (progress.rulesPassed) passed.push(tr('cohort and analysis rules', '队列与分析规则'));
      const steps = Number(progress.validatedSteps) || 0;
      if (steps > 0) passed.push(tr(`${steps} executable-step validation${steps === 1 ? '' : 's'}`, `${steps} 次可执行步骤校验`));
      const parts = [];
      if (passed.length) parts.push(tr(`Validated: ${passed.join(', ')}`, `已通过校验：${passed.join('、')}`));
      const retries = Number(progress.retries) || 0;
      if (retries > 0) parts.push(tr(`${retries} automatic correction${retries === 1 ? '' : 's'}`, `自动修正 ${retries} 次`));
      const current = String(progress.current || '').trim();
      if (current) parts.push(tr(`Now: ${current}`, `当前：${current}`));
      return parts.join(' · ');
    }

    function projectPipelineSteps(steps, running) {
      const source = Array.isArray(steps) ? steps : [];
      const hasPlanStage = source.some(step => step.kind === 'pipeline'
        && ['planning', 'plan', 'scientific_review'].includes(String(step.step || '').toLowerCase()));
      const projected = [];
      const stageIndexes = new Map();
      source.forEach(step => {
        if (step.kind !== 'pipeline') { projected.push(step); return; }
        const sourceStage = pipelineStage(step.step);
        let stage = sourceStage;
        if (stage === 'terminal' && hasPlanStage) stage = 'plan';
        const fallbackAllowed = ['submitted', 'terminal', 'attention'].includes(sourceStage);
        // Each intermediate validation reports "complete"; while the job is
        // still running the collapsed plan row must keep its running label
        // rather than announce a finished plan after the first passed step.
        const stageStatus = running && stage === 'plan' && step.status === 'complete'
          ? 'running' : String(step.status || '');
        const next = {
          ...step,
          id: `pipeline-stage-${stage}`,
          step: stage,
          label: pipelineStageLabel(
            fallbackAllowed ? sourceStage : stage,
            stageStatus,
            fallbackAllowed ? step.label : '',
          ),
          // Artifact navigation belongs to the result/confirmation card. The
          // activity list is lifecycle progress, not a second artifact menu.
          resource: null,
          resources: [],
        };
        if (stageIndexes.has(stage)) {
          projected[stageIndexes.get(stage)] = next;
        } else {
          stageIndexes.set(stage, projected.length);
          projected.push(next);
        }
      });
      return orderPipelineStages(projected);
    }
    /* A reloaded page rebuilds the timeline from a bounded job snapshot and
       then replays the full event stream, so later stages can arrive first.
       Present lifecycle stages in pipeline order; rows that are not stages
       (tool calls, retries) stay attached to the stage that preceded them. */
    const STAGE_ORDER = ['submitted', 'setup', 'inputs', 'evidence', 'plan', 'analysis', 'figure', 'report', 'progress', 'attention', 'terminal'];
    function orderPipelineStages(rows) {
      let previous = -1;
      const keyed = rows.map((row, index) => {
        const isStage = row.kind === 'pipeline' && String(row.id || '').startsWith('pipeline-stage-');
        const order = isStage ? STAGE_ORDER.indexOf(String(row.step || '')) : -1;
        if (isStage && order >= 0) previous = order;
        return { row, index, key: isStage && order >= 0 ? order : previous };
      });
      keyed.sort((a, b) => (a.key - b.key) || (a.index - b.index));
      return keyed.map(item => item.row);
    }

    function isVisibleOperation(step) {
      if (!step || !VISIBLE_KINDS.has(step.kind)) return false;
      if (step.kind !== 'pipeline') return true;
      // Submission and terminal envelopes describe transport state rather than
      // a scientific action. The actual stage rows and any attention event stay
      // visible, while the envelope remains available in the run receipt.
      return !['submitted', 'terminal'].includes(pipelineStage(step.step));
    }

    function stepLabel(step) {
      const done = step.status === 'complete';
      const failed = step.status === 'error';
      if (step.kind === 'submitted') return tr('Message submitted to EasyICU Copilot', '消息已提交给 EasyICU 研究助手');
      if (step.kind === 'agent') return tr('Copilot workflow started', '研究助手工作流已启动');
      if (step.kind === 'turn') return done
        ? tr(`Model turn ${step.turn + 1} finished`, `模型回合 ${step.turn + 1} 已结束`)
        : tr(`Model turn ${step.turn + 1} is running`, `模型回合 ${step.turn + 1} 进行中`);
      if (step.kind === 'assistant') {
        if (failed) return tr('The model response phase did not complete', '模型回复阶段未完成');
        if (done && step.publicChars) return tr(`Public response phase ${step.phase} finished`, `公开回复阶段 ${step.phase} 已输出`);
        if (done && step.stopReason === 'toolUse') return tr(`Model processing phase ${step.phase} finished`, `模型处理阶段 ${step.phase} 已完成`);
        if (done) return tr(`Model response phase ${step.phase} finished`, `模型回复阶段 ${step.phase} 已完成`);
        if (step.publicChars) return tr(`Streaming public response phase ${step.phase}`, `正在流式输出公开回复阶段 ${step.phase}`);
        return tr(`Preparing the next visible action ${step.phase}`, `正在准备下一步可见操作 ${step.phase}`);
      }
      if (step.kind === 'thinking') {
        const headline = reasoningHeadline(step.text);
        if (headline) return headline;
        return done ? tr('Thought it through', '已完成思考') : tr('Thinking…', '正在思考…');
      }
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
    function operationPayload(step, label) {
      const resources = [step.resource].concat(Array.isArray(step.resources) ? step.resources : []).filter(Boolean);
      const safeResources = resources.slice(0, 12).map(resource => ({
        kind: String(resource.kind || '').slice(0, 80),
        artifact: String(resource.artifact || '').slice(0, 240),
        filename: String(resource.filename || '').slice(0, 240),
        title: String(resource.title || resource.label || '').slice(0, 240),
        label: String(resource.label || '').slice(0, 240),
        run_id: String(resource.run_id || '').slice(0, 160),
        sha256: String(resource.sha256 || '').slice(0, 64),
      }));
      return encodeURIComponent(JSON.stringify({
        id: String(step.id || `${step.kind || 'step'}-${label}`).slice(0, 240),
        title: label,
        kind: String(step.kind || 'operation').slice(0, 80),
        icon: String(activityIcon(step) || 'play').slice(0, 40),
        status: String(step.status || 'complete').slice(0, 40),
        detail: step.kind === 'thinking' ? String(step.text || '').slice(0, 4000) : localizedStepText(step),
        duration: stepDuration(step),
        toolName: String(step.toolName || step.step || '').slice(0, 160),
        resources: safeResources,
      }));
    }
    /* One trace row is one button: icon, action, elapsed time, chevron. The
       whole row opens the operation workbench, as in the reference UI. */
    function stepPrimary(step) {
      const label = stepLabel(step);
      const operation = operationPayload(step, label);
      const thinking = step.kind === 'thinking';
      const detail = thinking ? '' : localizedStepText(step);
      const running = step.status === 'running';
      const reasoning = thinking
        ? `<span class="gpi-activity-reasoning${running ? ' is-streaming' : ''}" data-gpi-thinking-stream="${esc(step.id || '')}">${reasoningHtml(step.text)}</span>`
        : '';
      const publicStream = step.kind === 'assistant' && running && step.publicText
        ? `<span class="gpi-activity-public-stream"><em>${tr('Public output', '公开输出')}</em>${esc(step.publicText)}<i aria-hidden="true"></i></span>`
        : '';
      const meta = stepDuration(step);
      const started = Number(step.startedAt);
      const duration = running && Number.isFinite(started)
        ? `<span class="gpi-activity-step-duration" data-gpi-live-elapsed="${started}">${iconHtml('clock', 12)}${esc(durationText(started))}</span>`
        : meta
          ? `<span class="gpi-activity-step-duration">${iconHtml('clock', 12)}${esc(meta)}</span>`
          : '<span class="gpi-activity-step-duration" aria-hidden="true"></span>';
      const icon = running
        ? '<span class="gpi-running-spinner" aria-hidden="true"></span>'
        : iconHtml(activityIcon(step), 16);
      return `<button type="button" class="gpi-activity-step-open" data-gpi-operation="${esc(operation)}" aria-label="${esc(label)}">
        <span class="gpi-activity-step-icon" aria-hidden="true">${icon}</span>
        <span class="gpi-activity-step-copy"><strong>${esc(label)}</strong>${publicStream}${detail ? `<span>${esc(detail)}</span>` : ''}${reasoning}</span>
        ${duration}
        <span class="gpi-disclosure" aria-hidden="true">${iconHtml('chevron', 14)}</span>
      </button>`;
    }
    function stepResources(step) {
      const seen = new Set();
      const resources = [step.resource].concat(Array.isArray(step.resources) ? step.resources : [])
        .filter(resource => {
          const key = resourceKey(resource);
          if (!key || seen.has(key)) return false;
          seen.add(key);
          return true;
        });
      if (!resources.length) return '';
      return `<div class="gpi-resource-list" aria-label="${tr('Run artifacts', '运行产物')}">${resources.map(resource => resourceButton(resource)).join('')}</div>`;
    }
    function localizedStepText(step) {
      const value = String(step && step.text || '').trim();
      if (!value) return '';
      const containsChinese = /[\u3400-\u9fff]/.test(value);
      return window.EU_LANG === 'zh'
        ? (containsChinese ? value : '')
        : (containsChinese ? '' : value);
    }
    function stepRow(step) {
      // Raw validator codes and Python/Node owner paths are diagnostics, not
      // user-facing research progress. Keep them in persisted receipts while
      // projecting only elapsed time in the ordinary activity timeline.
      return `<li class="${esc(step.status || 'complete')} kind-${esc(step.kind || 'operation')}">${stepPrimary(step)}${stepResources(step)}</li>`;
    }
    /* The model's interim narration ("I'll check the data source next") is
       part of the trace, not a separate reply: it sits between the rows it
       explains, after the tool call that preceded it. */
    function narrationRow(html) {
      return html ? `<li class="gpi-activity-narration">${html}</li>` : '';
    }
    function traceListHtml(steps, narration) {
      const segments = (Array.isArray(narration) ? narration : [])
        .filter(item => item && item.html)
        .map(item => ({ after: Number.isFinite(Number(item.afterSteps)) ? Number(item.afterSteps) : Number.POSITIVE_INFINITY, html: item.html }))
        .sort((left, right) => left.after - right.after);
      const items = [];
      let tools = 0;
      let cursor = 0;
      const drain = limit => {
        while (cursor < segments.length && segments[cursor].after <= limit) {
          items.push(narrationRow(segments[cursor].html));
          cursor += 1;
        }
      };
      drain(0);
      steps.forEach(step => {
        items.push(stepRow(step));
        if (step.kind === 'tool') { tools += 1; drain(tools); }
      });
      while (cursor < segments.length) { items.push(narrationRow(segments[cursor].html)); cursor += 1; }
      return items.length ? `<ol>${items.join('')}</ol>` : '';
    }
    function render(row, options) {
      const allSteps = Array.isArray(row && row.steps) ? row.steps : [];
      const running = row && row.status === 'running';
      const visibleSteps = projectPipelineSteps(allSteps.filter(isVisibleOperation), running);
      const narration = options && Array.isArray(options.narration) ? options.narration : [];
      const latest = visibleSteps[visibleSteps.length - 1];
      const failed = row && (row.status === 'error' || row.status === 'failed' || row.status === 'cancelled');
      const kicker = tr('Show traces', '查看执行过程');
      if (running) {
        const title = row.runningTitle || (latest
          ? stepLabel(latest) : tr('EasyICU Copilot is preparing the next action', 'EasyICU 研究助手正在准备下一步'));
        const note = String(row.runningNote || '').trim() || tr(
          'Still running. You can wait here; EasyICU will ask when review or confirmation is needed.',
          '任务仍在进行。请在这里等待；需要审阅或确认时 EasyICU 会明确提示。',
        );
        return `<div class="gpi-activity-running" role="status" aria-live="polite" aria-busy="true">
          <div class="gpi-activity-live">
            <span class="gpi-running-spinner" aria-hidden="true"></span>
            <span class="gpi-activity-kicker">${esc(tr('In progress', '正在进行'))}</span>
            <span class="gpi-activity-title"><strong>${esc(title)}</strong></span>
            <span class="gpi-activity-elapsed" data-gpi-live-elapsed="${Number(row.startedAt || Date.now())}">${esc(durationText(row.startedAt))}</span>
          </div>
          ${traceListHtml(visibleSteps, narration)}
          <p class="gpi-activity-note">${esc(note)}</p>
        </div>`;
      }
      // Biomni-style traces are evidence of actual work. A plain response with
      // only submit/start/model lifecycle events should read as a plain response
      // instead of exposing an empty or misleading execution disclosure.
      if (!visibleSteps.length && !failed) return '';
      const title = failed ? tr('Execution needs attention', '执行过程需要处理') : kicker;
      const traceMeta = `${tr(`${visibleSteps.length} steps`, `${visibleSteps.length} 个步骤`)}${row.durationKnown === false ? '' : ` · ${tr(`total ${durationText(row.startedAt, row.endedAt)}`, `总耗时 ${durationText(row.startedAt, row.endedAt)}`)}`}`;
      const longTrace = visibleSteps.length + narration.length > 5;
      // Finished failures expose their status in the summary but keep verbose
      // diagnostic receipts collapsed. Persisted rows from older builds may
      // still carry expanded=true, so terminal rendering must not trust it.
      return `<details class="gpi-activity ${failed ? 'error' : 'complete'}">
        <summary aria-label="${esc(`${title}. ${traceMeta}`)}">
          <span class="gpi-disclosure" aria-hidden="true">${iconHtml('chevron', 14)}</span>
          <span class="gpi-activity-kicker">${esc(title)}</span>
          <span class="gpi-activity-meta" aria-hidden="true">${esc(traceMeta)}</span>
        </summary>
        <div class="gpi-activity-body${longTrace ? ' is-clipped' : ''}">
          ${traceListHtml(visibleSteps, narration)}
          ${longTrace ? `<button type="button" class="gpi-trace-expand" data-gpi-trace-expand aria-expanded="false">${iconHtml('external', 12)}<span>${esc(tr('Expand all', '展开全部'))}</span></button>` : ''}
          <p class="sr-only">${tr('Reasoning summaries, lifecycle facts, and EasyICU receipts — tool arguments, credentials, and patient rows are never displayed.', '这里展示模型的思考摘要、生命周期事实和 EasyICU 回执；工具参数、凭据和患者行级数据不会展示。')}</p>
        </div>
      </details>`;
    }
    /* A finished turn collapses. Its summary line already carries what a
       researcher needs -- which tools ran and how long the turn took -- while
       the expanded body is sub-second lifecycle detail written for debugging.
       Leaving the newest turn open put two screens of "已读取科研流程 0.6 秒"
       between the question and the answer. A running turn still expands, so
       progress stays visible; a failed summary remains clickable when someone
       needs its diagnostic receipts. */
    function focusLatest(rows) {
      rows.forEach(row => {
        if (row && row.role === 'activity' && row.status !== 'running') {
          row.expanded = false;
        }
      });
      return rows;
    }

    /* One reply turn reads as the reference UI does: the model's opening
       sentence, the traces it produced (with its interim narration between
       the rows), then the answer. `renderSegment(row, 'intro' | 'narration')`
       renders the text-only segments; without it every assistant row keeps
       its full message rendering after the traces. */
    function renderTimeline(rows, renderRow, renderSegment) {
      let pending = [];
      let turnTexts = [];
      const output = [];
      const segmentsSupported = typeof renderSegment === 'function';

      function traceHtml(narration) {
        if (!pending.length) return '';
        const rendered = pending
          .map((row, index) => ({
            row,
            html: renderRow(row, index === pending.length - 1 && narration.length ? { narration } : undefined),
          }))
          .filter(item => item.html);
        pending = [];
        if (!rendered.length) return '';
        if (rendered.length === 1) {
          const html = `<div class="gpi-turn-trace">${rendered[0].html}</div>`;
          return html;
        }
        // Keep the newest terminal failure -- or the turn still running --
        // visible beside the user's next decision. Earlier attempts stay in
        // the compact history disclosure.
        const latest = rendered[rendered.length - 1];
        const latestFailed = ['failed', 'error', 'cancelled', 'running'].includes(String(latest.row.status || ''));
        const history = latestFailed ? rendered.slice(0, -1) : rendered.slice();
        const rowsHtml = history.map(item => item.html).join('');
        const summary = tr('Show traces', '查看执行过程');
        const historyHtml = !history.length ? '' : history.length === 1
          ? `<div class="gpi-turn-trace">${rowsHtml}</div>`
          : `<details class="gpi-execution-history gpi-turn-traces"><summary>${esc(summary)}</summary><div class="gpi-turn-trace-list">${rowsHtml}</div></details>`;
        return `${historyHtml}${latestFailed ? `<div class="gpi-turn-trace">${latest.html}</div>` : ''}`;
      }

      function tracedTurn() {
        return pending.length > 0 && pending.some(row => Array.isArray(row.steps)
          && row.steps.some(step => step && step.kind === 'tool'));
      }

      function flush() {
        const texts = turnTexts;
        turnTexts = [];
        if (!segmentsSupported || texts.length < 2 || !tracedTurn()) {
          const html = traceHtml([]);
          if (html) output.push(html);
          texts.forEach(row => output.push(renderRow(row)));
          return;
        }
        const final = texts[texts.length - 1];
        const earlier = texts.slice(0, -1);
        const intro = Number(earlier[0].afterSteps) === 0 ? earlier.shift() : null;
        const narration = earlier.map(row => ({ afterSteps: row.afterSteps, html: renderSegment(row, 'narration') }));
        if (intro) output.push(renderSegment(intro, 'intro'));
        const html = traceHtml(narration);
        if (html) output.push(html);
        output.push(renderRow(final));
      }

      rows.forEach(row => {
        // A running turn groups the same way, so the opening sentence, the
        // live rows, and the streaming answer keep their places while the
        // model works instead of re-flowing when the turn settles.
        if (row && row.role === 'activity') {
          if (turnTexts.length) flush();
          pending.push(row);
          return;
        }
        // The persisted timeline already places a completed operation trace
        // before the assistant result it produced. Preserve that order so the
        // conversation reads: optional plan -> actual work -> final answer.
        if (row && row.role === 'assistant' && pending.length) {
          turnTexts.push(row);
          return;
        }
        flush();
        output.push(renderRow(row));
      });
      flush();
      return output.join('');
    }

    return Object.freeze({ appendPublicDelta, durationText, finishTurn, focusLatest, pipelineEventLabel, planningEventFacts, planningProgressText, reasoningHtml, render, renderTimeline, startTurn, stepLabel, syncLiveClock, timeMs });
  }

  window.EasyICU.guidedPi.declare('activity', { create });
})();
