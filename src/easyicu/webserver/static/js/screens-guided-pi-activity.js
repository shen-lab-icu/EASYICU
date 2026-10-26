/* Guided Copilot activity timeline.
   Owner: browser-safe lifecycle/tool rendering. It shows the model's own
   reasoning duration as trace rows and never
   receives tool arguments, credentials, patient rows, or host paths. */
(function () {
  'use strict';

  // Match the conversation contract used by research workspaces: the trace is
  // a record of actions a researcher can understand and inspect. Transport
  // acknowledgements, agent startup, model phases, and context maintenance
  // remain in the persisted receipt, but do not become fake "work" rows.
  const VISIBLE_KINDS = new Set(['tool', 'pipeline', 'retry', 'thinking']);
  const DURATION_KINDS = new Set(['assistant', 'tool', 'pipeline', 'retry', 'compaction', 'thinking']);
  // Traces a reader expanded keep that state across live re-renders.
  const expandedTraces = new Set();
  function traceExpanded(key, expanded) {
    const value = String(key || '').slice(0, 240);
    if (!value) return false;
    if (expanded === true) expandedTraces.add(value);
    else if (expanded === false) expandedTraces.delete(value);
    return expandedTraces.has(value);
  }

  function create(host) {
    const tr = host.tr;
    const esc = host.esc;
    const iconHtml = host.iconHtml;
    const resourceName = host.resourceName;
    const resourceKey = host.resourceKey;
    const resourceButton = host.resourceButton;
    let liveClock = 0;
    let liveClockRoot = null;

    function timeMs(value) {
      const parsed = Date.parse(String(value || ''));
      return Number.isFinite(parsed) ? parsed : Date.now();
    }
    function durationText(startedAt, endedAt) {
      const elapsed = Math.max(0, Number(endedAt || Date.now()) - Number(startedAt || Date.now()));
      if (elapsed < 100) return tr('<0.1 s', '<0.1 秒');
      const seconds = elapsed / 1000;
      if (seconds < 60) {
        const value = seconds.toFixed(1);
        return tr(`${value} s`, `${value} 秒`);
      }
      const roundedSeconds = Math.round(seconds);
      const minutes = Math.floor(roundedSeconds / 60);
      const remainder = roundedSeconds % 60;
      return tr(
        remainder ? `${minutes} min ${remainder} s` : `${minutes} min`,
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
      return '';
    }
    // A row shows a kind icon only when its kind is known.
    function activityIcon(step) {
      if (!step) return '';
      if (step.icon) return step.icon;
      // A plan step's icon says how it ran: a registered standard method or
      // code written for this study.
      if (step.step === 'plan_step' && EXECUTION_ICONS[step.execution]) return EXECUTION_ICONS[step.execution];
      if (step.kind === 'submitted') return 'arrow';
      if (step.kind === 'turn' || step.kind === 'retry') return 'refresh';
      if (step.kind === 'assistant') return 'wand';
      if (step.kind === 'thinking') return 'spark';
      if (step.kind === 'tool') return toolIcon(step.toolName);
      if (step.kind === 'pipeline') {
        if (/artifact|report|manuscript/.test(step.step || '')) return 'file';
        if (/gate|valid|audit|evidence/.test(step.step || '')) return 'shield';
        if (/plan/.test(step.step || '')) return 'list';
        if (String(step.step || '').startsWith('provider_')) return 'refresh';
        const stage = STAGE_ORDER.includes(step.step) ? step.step : pipelineStage(step.step);
        return stage === 'progress' ? '' : 'play';
      }
      if (step.kind === 'compaction') return 'layers';
      if (step.kind === 'failed') return 'alert';
      if (step.kind === 'cancelled') return 'stop';
      if (step.kind === 'settled') return 'check';
      return '';
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
        easyicu_state_target_trial: tr('State the target trial', '说明目标试验'),
        easyicu_quick_answer: tr('Prepare a quick answer', '准备快速回答'),
        easyicu_propose_multi_database: tr('Prepare the same study on other databases', '准备在其他数据库上运行同一研究'),
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
      const label = labels[String(name || '')] || tr('EasyICU tool', 'EasyICU 工具');
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
        easyicu_state_target_trial: tr('Stated the target trial for compiling', '已提交目标试验编译'),
        easyicu_quick_answer: tr('Prepared a quick answer', '已准备快速回答'),
        easyicu_propose_multi_database: tr('Prepared the other-database review', '已准备其他数据库的审阅'),
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
      if (['data_foundation', 'cohort', 'context', 'audit', 'extraction', 'data', 'source', 'snapshot', 'cache'].includes(step)) return 'inputs';
      if (['hypothesis', 'literature', 'evidence'].includes(step)) return 'evidence';
      if (['planning', 'plan', 'scientific_review', 'contract'].includes(step)) return 'plan';
      if (['step', 'coder', 'runner', 'analysis', 'runner_repair'].includes(step)) return 'analysis';
      if (['figure', 'visual_qa'].includes(step)) return 'figure';
      if (['writer', 'latex', 'manuscript', 'report'].includes(step)) return 'report';
      if (step === 'terminal') return 'terminal';
      if (step === 'resume') return 'resume';
      if (step === 'report_repair') return 'report_repair';
      if (['event_stream', 'cancel_requested', 'pause', 'human_review', 'scientific_action_gap'].includes(step)) return 'attention';
      return 'progress';
    }

    function pipelineStageLabel(stage, status, fallback) {
      const done = status === 'complete';
      if (status === 'error') return tr('Research task stopped; review the failure details', '研究任务已停止，请查看失败原因');
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
      if (stage === 'resume') return done
        ? tr('Reused the steps this approved run had already completed', '复用了这次获批运行已完成的步骤')
        : tr('Checking which approved steps already completed', '正在核对已完成的步骤');
      if (stage === 'report_repair') return done
        ? tr('Report rewritten from the sealed results; no analysis re-run', '已基于封存结果重写报告，未重跑分析')
        : tr('Rewriting the report from the sealed results', '正在基于封存结果重写报告');
      if (stage === 'progress') return done
        ? tr('Research-task progress updated', '研究任务进度已更新')
        : tr('Research task is progressing', '研究任务正在推进');
      if (stage === 'attention') return fallback || tr('Needs attention', '需处理');
      return String(fallback || tr('Research-task status updated', '研究任务状态已更新'));
    }

    /* An approved plan step reports "Step n/N started|complete: <step_id>."
       and its coder/runner events name the same step ("... for <step_id>.").
       Each plan step is its own trace row, named for a reader, with how it
       ran as its detail; the step id stays in the persisted receipt. */
    const PLAN_STEP_EVENT = /^Step\s+(\d+)\/(\d+)\s+(started|complete|failed contract checks)\b:?\s*([A-Za-z0-9_.-]*)/i;
    const PLAN_SUB_STEP_EVENT = /\bfor\s+([A-Za-z0-9_.-]+?)\.?$/;
    const STEP_NAMES = {
      cohort_accounting: ['Cohort and denominators', '队列与分母核算'],
      cohort_definition: ['Cohort definition', '队列定义'],
      baseline_context: ['Baseline characteristics', '基线特征'],
      table_one: ['Baseline characteristics', '基线特征'],
      exposure_outcome_distribution: ['Group proportions and outcomes', '分组比例与结局'],
      measurement_audit: ['Measurement and missingness audit', '测量与缺失审计'],
      measurement_quality: ['Measurement quality audit', '测量质量审计'],
      descriptive_context_figure: ['Result figure', '结果图'],
      cohort_accounting_figure: ['Cohort flow figure', '队列流程图'],
      data_quality_figure: ['Data-quality figure', '数据质量图'],
      article_report: ['Article report', '文章报告'],
      article_display_report: ['Article figures and tables', '文章图表'],
      robustness_replay: ['Robustness re-analysis', '稳健性重分析'],
      robustness_replay_figure: ['Robustness figure', '稳健性图'],
      functional_form_sensitivity: ['Functional-form sensitivity', '函数形式敏感性分析'],
    };
    const STEP_TOKENS = {
      cohort: ['cohort', '队列'], accounting: ['accounting', '核算'], baseline: ['baseline', '基线'],
      context: ['context', '背景'], exposure: ['exposure', '暴露'], outcome: ['outcome', '结局'],
      distribution: ['distribution', '分布'], measurement: ['measurement', '测量'], audit: ['audit', '核查'],
      quality: ['quality', '质量'], missingness: ['missingness', '缺失'], missing: ['missing', '缺失'],
      figure: ['figure', '图'], data: ['data', '数据'], descriptive: ['descriptive', '描述性'],
      primary: ['primary', '主要'], model: ['model', '模型'], logistic: ['logistic', 'Logistic'],
      cox: ['Cox', 'Cox'], sensitivity: ['sensitivity', '敏感性分析'], robustness: ['robustness', '稳健性'],
      subgroup: ['subgroup', '亚组'], spline: ['spline', '样条'], landmark: ['landmark', 'landmark'],
      time: ['time', '时间'], varying: ['varying', '时变'], association: ['association', '关联'],
      adjusted: ['adjusted', '校正'], absolute: ['absolute', '绝对'], risk: ['risk', '风险'],
      trend: ['trend', '趋势'], report: ['report', '报告'], article: ['article', '文章'],
      survival: ['survival', '生存'], analysis: ['analysis', '分析'], flow: ['flow', '流程'],
      mortality: ['mortality', '死亡'], stage: ['stage', '分期'], aki: ['AKI', 'AKI'], kdigo: ['KDIGO', 'KDIGO'],
      trajectory: ['trajectory', '轨迹'], phenotype: ['phenotype', '表型'], prediction: ['prediction', '预测'],
      calibration: ['calibration', '校准'], discrimination: ['discrimination', '区分度'], table: ['table', '表'],
    };
    /* A plan step's row says what it did. A registered standard executor
       names its own action; otherwise a known step id, or a step id ending
       in a kind word, gives one. The step's model-written purpose stays in
       the plan. The grouping variable and the outcome get named here once
       the plan carries their reader names (w-2). */
    const EXECUTION_ICONS = { standard: 'layers', generated: 'edit', fallback: 'shield', host: 'db' };
    const STEP_VERBS = {
      build: ['Build the', '生成'], draw: ['Draw the', '绘制'], summarize: ['Summarize the', '汇总'],
      audit: ['Audit the', '检查'], run: ['Run the', '运行'], fit: ['Fit the', '拟合'],
    };
    const EXECUTOR_ACTIONS = {
      grouped_table_one: ['build', 'baseline characteristics table by exposure group', '基线特征表（按暴露分组）'],
      cohort_flow_figure: ['draw', 'cohort selection flow chart', '队列筛选流程图'],
      exposure_outcome_distribution: ['summarize', 'outcomes by exposure group', '各组的结局'],
      exposure_outcome_distribution_figure: ['draw', 'outcome distribution by exposure group', '各组结局分布图'],
      descriptive_cohort_summary: ['summarize', 'cohort characteristics', '队列特征'],
      missingness_audit: ['audit', 'measurements and missing data', '测量与缺失情况'],
      missingness_measurement_audit: ['audit', 'measurements and missing data', '测量与缺失情况'],
      prevalence_outcome_figure: ['draw', 'prevalence and outcome figure', '患病比例与结局图'],
      prevalence_mortality_figure: ['draw', 'prevalence and mortality figure', '患病比例与死亡率图'],
      descriptive_result_figure: ['draw', 'result figure', '结果图'],
      composite_descriptive_figure: ['draw', 'combined descriptive figure', '描述性组合图'],
      measurement_missingness_figure: ['draw', 'missing-data figure', '缺失情况图'],
      missingness_measurement_figure: ['draw', 'missing-data figure', '缺失情况图'],
      audit_panel: ['draw', 'data-quality audit panel', '数据质量核查图'],
      robustness_figure: ['draw', 'robustness figure', '稳健性分析图'],
      robustness: ['run', 'robustness re-analysis', '稳健性重分析'],
      absolute_risk_context: ['summarize', 'absolute risks', '绝对风险'],
      grouped_descriptive_distribution: ['summarize', 'distribution by group', '各组分布'],
      continuous_outcome_summary: ['summarize', 'continuous outcomes by group', '各组的连续结局'],
      association_model_grid: ['fit', 'adjusted association models', '调整后的关联模型'],
      adjusted_association_estimates: ['fit', 'adjusted association model', '调整后的关联模型'],
      adjusted_association_figure: ['draw', 'adjusted association figure', '调整后关联图'],
      association_overview_figure: ['draw', 'association overview figure', '关联概览图'],
      survival_primary_cox: ['fit', 'primary Cox model', '主要 Cox 模型'],
      static_prediction_model: ['fit', 'prediction model', '预测模型'],
    };
    const STEP_ACTIONS = {
      cohort_accounting: ['Count the cohort and its denominators', '核算队列与分母'],
      baseline_context: ['Build the baseline characteristics table', '生成基线特征表'],
      table_one: ['Build the baseline characteristics table', '生成基线特征表'],
      exposure_outcome_distribution: ['Summarize group proportions and outcomes', '汇总分组比例与结局'],
      measurement_audit: ['Audit measurements and missing data', '检查测量与缺失情况'],
      measurement_quality: ['Audit measurement quality', '检查测量质量'],
      descriptive_context_figure: ['Draw the result figure', '绘制结果图'],
      cohort_accounting_figure: ['Draw the cohort flow chart', '绘制队列流程图'],
      data_quality_figure: ['Draw the data-quality figure', '绘制数据质量图'],
      robustness_replay: ['Run the robustness re-analysis', '运行稳健性重分析'],
      robustness_replay_figure: ['Draw the robustness figure', '绘制稳健性图'],
      functional_form_sensitivity: ['Run the functional-form sensitivity analysis', '运行函数形式敏感性分析'],
    };
    const LAST_TOKEN_VERBS = { figure: 'draw', table: 'build', model: 'fit', distribution: 'summarize' };
    function verbPhrase(verb, noun) {
      const words = STEP_VERBS[verb];
      const english = /^[A-Z][a-z]/.test(noun) ? noun.charAt(0).toLowerCase() + noun.slice(1) : noun;
      return tr(`${words[0]} ${english}`, `${words[1]}${noun}`);
    }
    function executorName(key) {
      const action = EXECUTOR_ACTIONS[String(key || '')];
      return action ? tr(action[1], action[2]) : '';
    }
    /* With the run's own reader names (the trace's group_variable_label and
       outcome_label, launch-b2) a grouped step names its grouping and its
       outcome: 「汇总按血糖分组的院内死亡」. Without them the title stays
       generic; nothing is inferred from a column code. */
    const QUALIFIED_NOUNS = {
      grouped_table_one: (g) => g && [`baseline characteristics table by ${g.en}`, `基线特征表（按${g.zh}）`],
      exposure_outcome_distribution: (g, o) => g && o && [`${o.en} by ${g.en}`, `按${g.zh}的${o.zh}`],
      exposure_outcome_distribution_figure: (g, o) => g && o && [`${o.en} distribution by ${g.en}`, `按${g.zh}的${o.zh}分布图`],
      continuous_outcome_summary: (g, o) => g && o && [`${o.en} by ${g.en}`, `按${g.zh}的${o.zh}`],
      grouped_descriptive_distribution: (g) => g && [`distribution by ${g.en}`, `按${g.zh}的分布`],
    };
    function readerLabel(value) {
      if (!value || typeof value !== 'object') return null;
      const clean = item => (typeof item === 'string' ? item.replace(/\s+/g, ' ').trim().slice(0, 80) : '');
      const zh = clean(value.zh), en = clean(value.en);
      return zh || en ? { zh: zh || en, en: en || zh } : null;
    }
    function readerLabels(event) {
      const group = readerLabel(event && event.group_variable_label);
      const outcome = readerLabel(event && event.outcome_label);
      return { ...(group ? { groupLabel: group } : {}), ...(outcome ? { outcomeLabel: outcome } : {}) };
    }
    function qualifiedNoun(key, row) {
      const build = QUALIFIED_NOUNS[key];
      const words = build ? build(row && row.groupLabel, row && row.outcomeLabel) : null;
      return words ? tr(words[0], words[1]) : '';
    }
    function planStepTitle(row) {
      const executor = String(row && row.executor || '');
      const action = EXECUTOR_ACTIONS[executor];
      if (action) return verbPhrase(action[0], qualifiedNoun(executor, row) || tr(action[1], action[2]));
      const id = String(row && row.stepId || '').replace(/^\d+_/, '').toLowerCase();
      if (STEP_ACTIONS[id]) return qualifiedNoun(id, row) ? verbPhrase('summarize', qualifiedNoun(id, row)) : tr(...STEP_ACTIONS[id]);
      const verb = LAST_TOKEN_VERBS[id.split(/[_.-]+/).pop()];
      const noun = planStepName(id);
      return verb && noun !== tr('Analysis step', '分析步骤') ? verbPhrase(verb, noun) : '';
    }
    function planStepName(stepId) {
      const id = String(stepId || '').replace(/^\d+_/, '').toLowerCase();
      if (!id) return tr('Analysis step', '分析步骤');
      if (STEP_NAMES[id]) return tr(...STEP_NAMES[id]);
      const tokens = id.split(/[_.-]+/).filter(Boolean);
      if (tokens.length && tokens.every(token => STEP_TOKENS[token])) {
        const words = tokens.map(token => STEP_TOKENS[token]);
        return tr(words.map(word => word[0]).join(' '), words.map(word => word[1]).join(''));
      }
      return tr('Analysis step', '分析步骤');
    }
    function planStepFacts(event) {
      if (!['step', 'resume'].includes(String(event && event.step || '').toLowerCase())) return null;
      const label = String(event && (event.label || event.message) || '').trim();
      const match = PLAN_STEP_EVENT.exec(label);
      const stepId = String(event.step_id || (match && match[4]) || '').replace(/\.$/, '');
      if (!stepId) return null;
      const skipped = /^Skipped (?:step already completed by pre-execution|completed step from prior run):/.test(label);
      const reused = /^Skipped completed step from prior run:/.test(label);
      const phase = skipped ? 'precompleted' : event.action === 'step_skipped' ? 'skipped'
        : ['error', 'failed', 'blocked', 'skipped'].includes(event.status) ? 'failed'
          : event.status === 'complete' || (match && /^complete/i.test(match[3])) ? 'complete'
            : match && /^failed/i.test(match[3]) ? 'failed' : 'started';
      const facts = { index: Number(event.current_step || (match && match[1])) || null,
        total: Number(event.total_steps || (match && match[2])) || null, phase, stepId, ...(reused ? { reused: true } : {}), ...readerLabels(event) };
      // The run the step belongs to addresses its detail; never a path.
      if (RUN_ID.test(String(event.run_id || ''))) facts.runId = String(event.run_id);
      return facts;
    }
    const RUN_ID = /^[A-Za-z][A-Za-z0-9_.-]{0,159}$/;
    function planStepLabel(facts) {
      if (!facts.index || !facts.total) return planStepName(facts.stepId);
      return tr(`Step ${facts.index}/${facts.total}: ${planStepName(facts.stepId)}`,
        `第 ${facts.index}/${facts.total} 步：${planStepName(facts.stepId)}`);
    }
    function planStepRow(facts, at, options) {
      const opts = options || {};
      const status = ['complete', 'precompleted'].includes(facts.phase) ? 'complete'
        : facts.phase === 'failed' ? 'error' : facts.phase === 'skipped' ? 'skipped' : 'running';
      return {
        id: `pipeline-plan-step-${facts.index || facts.stepId}`, kind: 'pipeline', step: 'plan_step', stepId: facts.stepId,
        label: planStepLabel(facts), status, at,
        ...(facts.index && facts.total ? { stepIndex: facts.index, stepTotal: facts.total } : {}),
        ...(facts.runId ? { runId: facts.runId } : {}),
        ...(facts.phase === 'started' ? { startedAt: at } : { endedAt: at }),
        ...(facts.phase === 'precompleted' ? { precompleted: true } : {}),
        ...(facts.reused ? { reused: true } : {}),
        ...(facts.phase === 'skipped' ? { skipped: true } : {}),
        ...(opts.durationKnown === false ? { durationKnown: false } : {}),
        ...(facts.groupLabel ? { groupLabel: facts.groupLabel } : {}),
        ...(facts.outcomeLabel ? { outcomeLabel: facts.outcomeLabel } : {}),
      };
    }
    function pipelineTime(value) {
      const at = Date.parse(String(value || ''));
      return Number.isFinite(at) ? at : null;
    }
    function finishPipelineSteps(activity, at, status) {
      activity.steps.forEach(row => {
        if (row.kind !== 'pipeline' || row.status !== 'running') return;
        row.status = status || 'complete';
        if (at != null) row.endedAt = at;
      });
    }
    // One reducer for SSE, hydrated jobs and host-action transcript replay.
    // Receipt times describe server work; a reload never starts a new clock.
    function notePipelineEvent(activity, event) {
      if (!event || event.type === 'end') return null;
      const actionRow = noteActionEvent(activity, event);
      if (actionRow !== undefined) return actionRow;
      const facts = planStepFacts(event);
      if (!facts && notePlanSubStep(activity.steps, planSubStepFacts(event))) {
        closeRunningRows(activity, pipelineTime(event.timestamp), '', true);
        return null;
      }
      const at = pipelineTime(event.timestamp);
      const start = pipelineTime(event.started_at);
      const end = pipelineTime(event.ended_at);
      const step = String(event.step || event.type || 'pipeline');
      const system = step.startsWith('provider_');
      const kind = step === 'planning' && planningEventFacts(event).phase === 'rejected' ? 'retry' : 'pipeline';
      const id = system ? `pipeline-${step}-${event.current || event.seq || 0}` : kind === 'retry'
        ? `pipeline-plan-retry-${planningEventFacts(event).unit || 'plan'}-${event.current || 0}` : `pipeline-${step}`;
      const next = facts ? planStepRow(facts, at, { durationKnown: at != null }) : {
        id, kind, step, label: pipelineEventLabel(event), at, actor: event.actor,
        ...(system ? { current: boundedCount(event.current), total: boundedCount(event.total) } : {}),
        status: ['error', 'failed', 'cancelled', 'blocked'].includes(event.status)
          || step === 'provider_continuation_refused' ? 'error'
          : event.status === 'complete' || (system && step !== 'provider_recovery_wait') ? 'complete' : 'running',
      };
      const previous = activity.steps.find(row => row.id === next.id);
      closeRunningRows(activity, at, next.id, false);
      const started = previous && previous.durationKnown !== false && previous.startedAt != null
        ? previous.startedAt : start != null ? start : next.status === 'running' ? at : null;
      Object.assign(next, { startedAt: started, durationKnown: started != null && at != null });
      if (next.status !== 'running') next.endedAt = end != null ? end : at;
      else if (previous) delete previous.endedAt;
      if (previous) Object.assign(previous, next); else activity.steps.push(next);
      return previous || next;
    }
    // A running span ends when the next event of the run arrives. Coder and
    // runner events inside a plan step end only typed action rows.
    function closeRunningRows(activity, at, exceptId, actionsOnly) {
      activity.steps.forEach(row => {
        if (row.kind !== 'pipeline' || row.step === 'plan_step' || row.id === exceptId || row.status !== 'running') return;
        if (actionsOnly && row.step !== 'action') return;
        row.status = 'complete';
        if (at != null) row.endedAt = at;
      });
    }

    /* Actions the host names by a closed code (run_step_trace.trace_action)
       are rows of their own with their own times: planning validations,
       data preparation, the literature search, and a repair or an error
       inside a plan step. A completion-only event has no measured duration. */
    const ACTION_STAGES = {
      runtime_validated: 'setup', context_built: 'inputs', exposure_groups_staged: 'inputs',
      cohort_materialized: 'inputs', cohort_verified: 'inputs', trajectory_staged: 'inputs',
      cohort_audit: 'inputs', literature_build: 'evidence', plan_validation: 'plan', plan_ready: 'plan',
      literature_deferred: 'evidence',
      repair: 'analysis', repair_retry: 'analysis', deterministic_repair: 'analysis',
      repair_failed: 'analysis', step_error: 'analysis', figure_audit: 'figure',
    };
    const SPAN_ACTIONS = new Set(['plan_validation', 'literature_build', 'figure_audit', 'repair', 'repair_retry']);
    const ACTION_ICONS = {
      runtime_validated: 'gear', context_built: 'file', exposure_groups_staged: 'layers', cohort_materialized: 'db',
      cohort_verified: 'db', trajectory_staged: 'db', cohort_audit: 'shield', literature_build: 'search',
      literature_deferred: 'search',
      plan_validation: 'shield', plan_ready: 'list', repair: 'refresh', repair_retry: 'refresh',
      deterministic_repair: 'refresh', repair_failed: 'alert', step_error: 'alert', figure_audit: 'eye',
    };
    function boundedCount(value, upper) {
      const number = Number(value);
      return Number.isInteger(number) && number >= 0 && number <= (upper || 100000) ? number : null;
    }
    function actionFieldsOf(event) {
      const text = (value, limit) => String(value == null ? '' : value).slice(0, limit);
      const code = value => (/^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(String(value || '')) ? String(value) : '');
      return {
        planningUnit: code(event.planning_unit), retryPhase: code(event.retry_phase),
        current: boundedCount(event.current), total: boundedCount(event.total),
        attempt: boundedCount(event.attempt), repairKind: code(event.repair_kind),
        queryCount: boundedCount(event.query_count), recordCount: boundedCount(event.record_count),
        queries: (Array.isArray(event.queries) ? event.queries : []).slice(0, 4).map(value => text(value, 200)).filter(Boolean),
        sourceLabel: text(event.source_label, 160), sourceDatabase: text(event.source_database, 64),
        groupVariableCount: boundedCount(event.group_variable_count), planStepCount: boundedCount(event.plan_step_count),
        stayCount: boundedCount(event.stay_count, 10000000), variableCount: boundedCount(event.variable_count, 10000000),
        stepId: /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}$/.test(String(event.step_id || '')) ? String(event.step_id) : '',
        ...(RUN_ID.test(String(event.run_id || '')) ? { runId: String(event.run_id) } : {}),
      };
    }
    function actionRowId(event) {
      const action = String(event.action);
      if (action === 'plan_validation') return `pipeline-action-plan_validation-${event.planning_unit || 'plan'}-${boundedCount(event.current) || 0}`;
      if (ACTION_STAGES[action] === 'analysis') {
        return `pipeline-action-${action}-${event.step || ''}-${event.step_id || ''}-${event.repair_kind || ''}-${boundedCount(event.attempt) || 0}`;
      }
      return `pipeline-action-${action}`;
    }
    // Returns the action's row, or undefined when the event names no action.
    function noteActionEvent(activity, event) {
      const action = String(event && event.action || '');
      if (!ACTION_STAGES[action]) return undefined;
      const at = pipelineTime(event.timestamp);
      const start = pipelineTime(event.started_at);
      const end = pipelineTime(event.ended_at);
      const id = actionRowId(event);
      const span = SPAN_ACTIONS.has(action) && !(action === 'plan_validation' && event.retry_phase !== 'started')
        && !(event.status === 'complete' && start != null && end != null);
      const failed = ['error', 'failed', 'blocked'].includes(String(event.status || ''))
        || ['repair_failed', 'step_error'].includes(action)
        || (action === 'plan_validation' && event.retry_phase === 'rejected');
      const previous = activity.steps.find(row => row.id === id);
      closeRunningRows(activity, at, id, false);
      const row = previous || { id, kind: 'pipeline', step: 'action', action, stage: ACTION_STAGES[action], icon: ACTION_ICONS[action] };
      Object.assign(row, actionFieldsOf(event), { at, status: failed ? 'error' : span ? 'running' : 'complete' });
      if (span) {
        row.startedAt = previous && previous.startedAt != null ? previous.startedAt : start != null ? start : at;
        delete row.endedAt;
        row.durationKnown = row.startedAt != null;
      } else if (previous && previous.startedAt != null && SPAN_ACTIONS.has(action)) {
        // The verdict of a started span (a validation attempt) ends it.
        row.endedAt = end != null ? end : at;
        row.durationKnown = row.endedAt != null;
      } else {
        row.endedAt = end != null ? end : at;
        row.durationKnown = SPAN_ACTIONS.has(action) && start != null && end != null;
        if (row.durationKnown) row.startedAt = start;
      }
      if (!previous) activity.steps.push(row);
      return row;
    }
    function actionLabel(row) {
      const attempt = Number(row.attempt) > 0 ? tr(` (attempt ${row.attempt})`, `（第 ${row.attempt} 次）`) : '';
      const repairs = {
        runtime: tr('The run failed; repairing the code', '运行失败，修复代码'),
        contract: tr('A contract check failed; repairing the code', '合同校验未通过，修复代码'),
        concept: tr('A concept check failed; repairing the code', '概念校验未通过，修复代码'),
        figure_layout: tr('Repairing the figure layout', '修复图件版式'),
      };
      const labels = {
        runtime_validated: tr('Check the execution environment', '检查执行环境'),
        context_built: tr('Build the research context', '构建研究上下文'),
        exposure_groups_staged: tr('Form the exposure groups', '划分暴露分组'),
        cohort_materialized: tr('Build the analysis cohort', '生成分析队列'),
        cohort_verified: tr('Verify the prepared cohort', '核验已准备的队列'),
        trajectory_staged: tr('Prepare the trajectory input', '准备轨迹输入'),
        cohort_audit: tr('Check the initial cohort', '核查初始队列'),
        literature_build: tr('Search the literature and build the hypotheses', '检索文献并构建研究假设'),
        literature_deferred: tr('Literature is part of a full article run; skipped for this first result',
          '文献检索属于完整论文流程，本次首轮结果已跳过'),
        plan_ready: Number(row.planStepCount) > 0
          ? tr(`Analysis plan ready: ${row.planStepCount} steps`, `分析计划就绪：共 ${row.planStepCount} 步`)
          : tr('Analysis plan ready', '分析计划就绪'),
        repair: (repairs[row.repairKind] || tr('Repairing the code', '修复代码')) + attempt,
        repair_retry: tr('The repair gave no usable code; retrying it', '修复未产出可用代码，重试修复') + attempt,
        deterministic_repair: tr('Applied a deterministic repair', '应用确定性修复'),
        repair_failed: tr('The repair failed', '修复失败'),
        step_error: tr('This step hit an error', '这一步运行出错'),
        figure_audit: tr('Check the generated figures', '检查生成的图件'),
      };
      if (row.action === 'plan_validation') {
        return pipelineEventLabel({ step: 'planning', planning_unit: row.planningUnit, retry_phase: row.retryPhase, current: row.current, total: row.total });
      }
      return labels[row.action] || tr('Research-task status updated', '研究任务状态已更新');
    }
    function actionDetail(row) {
      if (row.action === 'literature_build' && row.queryCount != null) {
        return row.recordCount != null
          ? tr(`${row.queryCount} queries · ${row.recordCount} records`, `${row.queryCount} 条检索式 · ${row.recordCount} 条记录`)
          : tr(`${row.queryCount} queries`, `${row.queryCount} 条检索式`);
      }
      return '';
    }
    // Typed inputs of a host action, for the workbench's Input section.
    function actionFacts(row) {
      const facts = [];
      const add = (label, value) => { if (value != null && value !== '') facts.push({ label, value: String(value) }); };
      const units = { structure: tr('Study structure', '研究结构'), rules: tr('Cohort and analysis rules', '队列与分析规则'),
        step: tr('Current executable step', '当前可执行步骤'), plan: tr('Candidate plan', '候选计划') };
      if (row.action === 'plan_validation') {
        add(tr('Checked', '校验对象'), units[row.planningUnit] || units.plan);
        add(tr('Attempt', '尝试'), row.current && row.total ? `${row.current}/${row.total}` : '');
      }
      if (row.action === 'literature_build') {
        add(tr('Queries', '检索式数'), row.queryCount);
        add(tr('Records found', '检索到的记录'), row.recordCount);
        (row.queries || []).forEach(query => add(tr('Query', '检索式'), query));
      }
      if (row.action === 'context_built') {
        add(tr('Data source', '数据源'), [row.sourceLabel, row.sourceDatabase ? `(${row.sourceDatabase})` : ''].filter(Boolean).join(' '));
        add(tr('ICU stays', 'ICU 入住数'), row.stayCount);
        add(tr('Variables built', '构建的变量数'), row.variableCount);
      }
      if (row.action === 'exposure_groups_staged') add(tr('Grouping variables', '分组变量数'), row.groupVariableCount);
      if (row.action === 'plan_ready') add(tr('Plan steps', '计划步骤数'), row.planStepCount);
      if (['repair', 'repair_retry'].includes(row.action)) add(tr('Repair attempt', '修复次数'), row.attempt);
      return facts.slice(0, 12);
    }

    // How a coder/runner event says the step ran, for the step row's detail.
    function planSubStepFacts(event) {
      const step = String(event && event.step || '').toLowerCase();
      if (!['coder', 'runner'].includes(step)) return null;
      const label = String(event && (event.label || event.message) || '').trim();
      const match = PLAN_SUB_STEP_EVENT.exec(label);
      const action = String(event.action || '');
      // A finished run's audit rows name the step only by typed coordinates.
      const typedStep = action && /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}$/.test(String(event.step_id || '')) ? String(event.step_id) : '';
      if (!match && !typedStep) return null;
      const execution = action === 'standard_executor' ? 'standard' : action === 'code_generation' ? 'generated'
        : action === 'fallback_code' ? 'fallback'
          : action === 'run' ? ({ standard: 'standard', fallback: 'fallback' }[event.run_kind] || 'generated') : '';
      const typed = {
        ...(execution ? { execution } : {}),
        ...(action === 'standard_executor' && /^[a-z][a-z0-9_]{0,79}$/.test(String(event.executor || '')) ? { executor: String(event.executor) } : {}),
        ...(RUN_ID.test(String(event.run_id || '')) ? { runId: String(event.run_id) } : {}),
        ...readerLabels(event),
      };
      const text = /^Running standard executor script/i.test(label) || (action === 'run' && event.run_kind === 'standard')
        ? tr('Ran the standard executor script; no model call.', '运行标准执行脚本，不调用模型。')
        : /^Using (?:planner-scoped|digest-bound)\b/i.test(label) || /deterministic standard executor/i.test(label)
          ? tr('Rendered by the deterministic renderer the plan names.', '由计划指定的确定性渲染器生成。')
          : (/generat|draft|writ/i.test(label) && step === 'coder') || action === 'code_generation'
            ? tr('The model wrote the analysis code for this step.', '由模型为这一步编写分析代码。')
            : '';
      return { stepId: match ? match[1] : typedStep, text, ...typed };
    }
    // Attach a coder/runner event to its plan-step row. False when no row
    // names that step, so the caller can keep its generic analysis row.
    function notePlanSubStep(steps, facts) {
      if (!facts) return false;
      const row = (Array.isArray(steps) ? steps : []).slice().reverse()
        .find(item => item && item.step === 'plan_step' && item.stepId === facts.stepId);
      if (!row) return false;
      if (facts.text) row.text = facts.text;
      if (facts.execution) row.execution = facts.execution;
      if (facts.executor) row.executor = facts.executor;
      if (facts.runId && !row.runId) row.runId = facts.runId;
      if (facts.groupLabel) row.groupLabel = facts.groupLabel;
      if (facts.outcomeLabel) row.outcomeLabel = facts.outcomeLabel;
      return true;
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
      if (event && String(event.step || '').startsWith('provider_')
        && !['provider_continuation_refused', 'provider_auto_resume', 'provider_recovery_wait'].includes(event.step))
        return ['error', 'failed', 'cancelled', 'blocked'].includes(event.status)
          ? tr('The model service reported a problem. Review the failure details.', '模型服务出现问题，请查看失败原因。')
          : event.status === 'running' ? tr('Checking the model service.', '正在检查模型服务。')
            : tr('Model service status updated.', '模型服务状态已更新。');
      if (event && event.step === 'provider_continuation_refused') return host.runFailureText(event.reason || event.reason_code);
      if (event && event.step === 'provider_auto_resume' && event.restart_stage === 'metadata_planning') return tr(
        'System action: provider recovered; regenerating the candidate plan with the saved study requirements',
        '系统操作：模型服务已恢复，正在按已保存的研究要求重新生成候选计划',
      );
      if (event && event.step === 'provider_auto_resume') return tr(
        'System action: provider recovered; automatically continuing from the verified checkpoint',
        '系统操作：模型服务已恢复，正在从已验证的检查点自动继续',
      );
      if (event && event.step === 'provider_recovery_wait') return tr(
        'System action: waiting for the model provider to recover within the job budget; then the run will continue automatically',
        '系统操作：正在任务预算内等待模型服务恢复，恢复后将自动继续',
      );
      if (type === 'start') return tr('EasyICU research pipeline started', 'EasyICU 科研流程已启动');
      if (type === 'cancel_requested' || event && event.step === 'cancel_requested') return tr('Cancellation requested; waiting for the task to stop.', '已请求取消，正在等待任务停止。');
      if (event && event.step === 'pause') return tr('The task is paused.', '任务已暂停。');
      if (event && event.step === 'human_review') return tr('The study needs manual review.', '这项研究需要人工核对。');
      if (event && event.step === 'scientific_action_gap') return tr('The study cannot continue with the current settings. Review the outstanding issues.', '当前设置下研究无法继续，请查看待处理问题。');
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
        && ['planning', 'plan', 'scientific_review', 'contract'].includes(String(step.step || '').toLowerCase()));
      const projected = [];
      const stageIndexes = new Map();
      source.forEach(step => {
        if (step.kind !== 'pipeline' || ['plan_step', 'action'].includes(step.step) || String(step.step || '').startsWith('provider_')) { projected.push(step); return; }
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
          const previous = projected[stageIndexes.get(stage)];
          next.startedAt = previous.startedAt == null ? next.startedAt : next.startedAt == null ? previous.startedAt : Math.min(previous.startedAt, next.startedAt);
          next.endedAt = next.endedAt == null ? previous.endedAt : next.endedAt;
          next.durationKnown = next.startedAt != null && (next.status === 'running' || next.endedAt != null);
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
    const STAGE_ORDER = ['submitted', 'setup', 'resume', 'inputs', 'evidence', 'plan', 'analysis', 'figure', 'report', 'report_repair', 'progress', 'attention', 'terminal'];
    function orderPipelineStages(rows) {
      let previous = -1;
      const keyed = rows.map((row, index) => {
        const isStage = row.kind === 'pipeline' && String(row.id || '').startsWith('pipeline-stage-');
        const order = isStage ? STAGE_ORDER.indexOf(String(row.step || '')) : -1;
        if (isStage && order >= 0) previous = order;
        // Plan-step rows are the analysis stage, in the order they ran; an
        // action row belongs to its own stage.
        if (row.kind === 'pipeline' && row.step === 'plan_step') return { row, index, key: STAGE_ORDER.indexOf('analysis') };
        if (row.kind === 'pipeline' && row.step === 'action') return { row, index, key: STAGE_ORDER.indexOf(row.stage) };
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
      if (step.kind === 'thinking') return tr('Thought', '思考');
      if (step.kind === 'tool') return failed
        ? tr(`${toolLabel(step.toolName, step.resource)} returned an error`, `${toolLabel(step.toolName, step.resource)} 返回错误`)
        : done ? completedToolLabel(step.toolName, step.resource)
          : tr(`Calling ${toolLabel(step.toolName, step.resource)}`, `正在调用 ${toolLabel(step.toolName, step.resource)}`);
      if (step.kind === 'pipeline' && step.step === 'action') return actionLabel(step);
      if (step.kind === 'pipeline' && step.step === 'plan_step') return planStepTitle(step) || String(step.label || planStepName(step.stepId));
      if (step.kind === 'pipeline') return String(step.label || tr('EasyICU research pipeline updated', 'EasyICU 科研流程已更新'));
      if (step.kind === 'retry') {
        if (step.label) return String(step.label);
        if (step.attempt != null && step.maxAttempts != null) {
          return tr(`Model service unavailable; retrying (${step.attempt}/${step.maxAttempts})`, `模型服务暂时不可用，正在重试（${step.attempt}/${step.maxAttempts}）`);
        }
        return tr('Retrying after a failed attempt', '上一步未通过，正在重试');
      }
      if (step.kind === 'compaction') return done ? tr('Context compaction finished', '上下文整理已完成') : tr('Compacting context', '正在整理上下文');
      if (step.kind === 'cancelled') return tr('This turn was stopped', '本轮已停止');
      if (step.kind === 'failed') return tr('This turn failed', '本轮失败');
      if (step.kind === 'settled') return tr('This turn completed', '本轮已完成');
      return tr('Agent activity updated', 'Agent 状态已更新');
    }
    function operationPayload(step, label, cause) {
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
      // A plan step is addressed by its run and step ids, server-issued
      // coordinates the workbench uses to read the code it ran.
      const planStep = step.kind === 'pipeline' && step.step === 'plan_step';
      return encodeURIComponent(JSON.stringify({
        id: String(step.id || `${step.kind || 'step'}-${label}`).slice(0, 240),
        title: label,
        kind: String(step.kind || 'operation').slice(0, 80),
        icon: String(activityIcon(step) || '').slice(0, 40),
        status: String(step.status || 'complete').slice(0, 40),
        detail: step.kind === 'thinking' ? '' : localizedStepText(step),
        duration: stepDuration(step),
        toolName: step.kind === 'tool' ? toolLabel(step.toolName) : '',
        resources: safeResources,
        ...(planStep && step.runId && step.stepId ? { run: String(step.runId).slice(0, 160), step: String(step.stepId).slice(0, 160) } : {}),
        ...(planStep ? {
          execution: String(step.execution || (step.precompleted ? 'host' : '')).slice(0, 40),
          executor: String(step.executor || '').slice(0, 80),
          executorName: executorName(step.executor),
          stepIndex: Number(step.stepIndex) || 0,
          stepTotal: Number(step.stepTotal) || 0,
        } : {}),
        ...(step.step === 'action' ? { facts: actionFacts(step) } : {}),
        ...(cause ? { cause: String(cause).slice(0, 400) } : {}),
      }));
    }
    /* One trace row is one button: icon, action, elapsed time, chevron. The
       whole row opens the operation workbench, as in the reference UI. */
    function stepPrimary(step, cause) {
      const label = stepLabel(step);
      const operation = operationPayload(step, label, cause);
      const thinking = step.kind === 'thinking';
      const detail = thinking ? '' : localizedStepText(step);
      const running = step.status === 'running';
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
      const kindIcon = activityIcon(step);
      const icon = running
        ? '<span class="gpi-running-spinner" aria-hidden="true"></span>'
        : kindIcon ? iconHtml(kindIcon, 16) : '';
      // A failed row's one-line cause, from the error-text owner.
      const causeHtml = cause ? `<span class="gpi-activity-step-cause" title="${esc(cause)}">${esc(cause)}</span>` : '';
      return `<button type="button" class="gpi-activity-step-open" data-gpi-operation="${esc(operation)}" aria-label="${esc(label)}">
        <span class="gpi-activity-step-icon" aria-hidden="true">${icon}</span>
        <span class="gpi-activity-step-copy"><strong>${esc(label)}${thinking ? ' ·' : ''}</strong>${publicStream}${detail ? `<span>${esc(detail)}</span>` : ''}${causeHtml}</span>
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
      if (step.kind === 'thinking') return '';
      if (step.precompleted && step.reused) return tr('Completed by an earlier attempt; reused', '上一次尝试已完成，直接复用');
      if (step.precompleted) return tr('Already completed during preparation; skipped', '已在准备阶段完成，跳过');
      if (step.skipped) return tr('Skipped: a step it depends on failed', '已跳过：所依赖的步骤失败');
      if (step.step === 'action') return actionDetail(step);
      if (step.kind === 'tool') {
        const receipt = step.receipt;
        if (step.status === 'complete' && step.toolName === 'easyicu_update_study_context'
          && receipt && receipt.study_family === 'descriptive_epidemiology') {
          const words = (Array.isArray(receipt.question_evidence) ? receipt.question_evidence : []).slice(0, 4).join(' / ');
          if (words) return tr(`Saved the study configuration: a descriptive study, as the question asks ('${words}')`,
            `保存了研究配置：按题目定为描述性研究（题目问“${words}”）`);
        }
        // Other families and partial updates have no public field-level receipt.
        if (step.status === 'complete' && step.toolName === 'easyicu_update_study_context') return step.code === 'study_context_updated'
          ? tr('This update is saved. Review the study setup for any remaining fields.', '本次修改已保存；请在研究设置中核对仍需填写的内容。')
          : tr('Review the study setup; this record does not list the updated fields.', '请核对研究设置；这条记录未列出修改的具体内容。');
        return '';
      }
      const value = String(step && step.text || '').trim();
      if (!value) return '';
      const containsChinese = /[\u3400-\u9fff]/.test(value);
      return window.EU_LANG === 'zh'
        ? (containsChinese ? value : '')
        : (containsChinese ? '' : value);
    }
    function stepRow(step, cause) {
      // Raw validator codes and Python/Node owner paths are diagnostics, not
      // user-facing research progress. Keep them in persisted receipts while
      // projecting only elapsed time in the ordinary activity timeline.
      return `<li class="${esc(step.status || 'complete')} kind-${esc(step.kind || 'operation')}">${stepPrimary(step, cause)}${stepResources(step)}</li>`;
    }
    // The host's own sentences (screens-guided-pi-job-narration.js); an
    // isolated owner test without it renders every row as a row.
    function jobNarration() {
      const registry = window.EasyICU && window.EasyICU.guidedPi;
      return registry && typeof registry.optional === 'function' ? registry.optional('jobNarration') : null;
    }
    // A muted line between rows, written by the host from a job event.
    function hostLineRow(text) {
      return `<li class="gpi-activity-narration gpi-trace-note"><span>${esc(text)}</span></li>`;
    }
    /* A failed row says why in one line, from the error-text owner, only when
       the job's own failure code names the stop. A finished job whose earlier
       step errors were repaired has no failure code, so no cause is shown. */
    function failureCauses(steps, row) {
      const causes = new Map();
      const code = String(row && row.failureCode || '');
      if (!code || typeof host.runFailureText !== 'function') return causes;
      const failed = steps.filter(step => step.status === 'error' && step.kind !== 'tool'
        && !String(step.step || '').startsWith('provider_'));
      const cause = String(host.runFailureText(code) || '').trim();
      if (failed.length && cause) causes.set(failed[failed.length - 1], cause);
      return causes;
    }
    /* The model's interim narration ("I'll check the data source next") is
       part of the trace, not a separate reply: it sits between the rows it
       explains, after the tool call that preceded it. */
    function narrationRow(html) {
      return html ? `<li class="gpi-activity-narration">${html}</li>` : '';
    }
    function traceListHtml(steps, narration, causes) {
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
      const owner = jobNarration();
      const lineState = {};
      steps.forEach(step => {
        const line = owner ? owner.lineFor(step, lineState, tr) : null;
        if (line && line.text) items.push(hostLineRow(line.text));
        if (!(line && line.replaces && line.text)) items.push(stepRow(step, causes && causes.get(step)));
        if (step.kind === 'tool') { tools += 1; drain(tools); }
      });
      while (cursor < segments.length) { items.push(narrationRow(segments[cursor].html)); cursor += 1; }
      return items.length ? `<ol>${items.join('')}</ol>` : '';
    }
    function expandButton(expanded) {
      return `<button type="button" class="gpi-trace-expand" data-gpi-trace-expand aria-expanded="${expanded ? 'true' : 'false'}">${iconHtml('external', 12)}<span>${esc(expanded ? tr('Collapse', '收起') : tr('Expand all', '展开全部'))}</span></button>`;
    }
    function render(row, options) {
      const allSteps = Array.isArray(row && row.steps) ? row.steps : [];
      const running = row && row.status === 'running';
      const visibleSteps = projectPipelineSteps(allSteps.filter(isVisibleOperation), running);
      const narration = options && Array.isArray(options.narration) ? options.narration : [];
      const latest = visibleSteps[visibleSteps.length - 1];
      const failed = row && (row.status === 'error' || row.status === 'failed' || row.status === 'cancelled');
      const kicker = tr('Show traces', '查看执行过程');
      // One host sentence before a job's trace says what EasyICU will do.
      const owner = jobNarration();
      const openingText = owner && row ? owner.opening(row.jobCode, tr) : '';
      const opening = openingText ? `<p class="gpi-turn-opening">${esc(openingText)}</p>` : '';
      const rowCount = visibleSteps.filter(step => !(owner && owner.replaces(step))).length;
      const longTrace = visibleSteps.length + narration.length > 5;
      const traceKey = String(row && (row.childJobId || row.id) || '');
      const expanded = longTrace && traceExpanded(traceKey);
      const bodyClass = `${longTrace ? ' is-clipped' : ''}${expanded ? ' is-expanded' : ''}`;
      const keyAttr = traceKey ? ` data-gpi-trace-key="${esc(traceKey)}"` : '';
      const failureExplained = Boolean(options && options.failureExplained);
      const causes = failureExplained ? new Map() : failureCauses(visibleSteps, row);
      if (running) {
        const title = row.runningTitle || (latest
          ? stepLabel(latest) : tr('EasyICU Copilot is preparing the next action', 'EasyICU 研究助手正在准备下一步'));
        const note = String(row.runningNote || '').trim() || tr(
          'Still running. You can wait here; EasyICU will ask when review or confirmation is needed.',
          '任务仍在进行。请在这里等待；需要审阅或确认时 EasyICU 会明确提示。',
        );
        // While running the list stays open; its window keeps the newest row in view.
        const list = traceListHtml(visibleSteps, narration, causes);
        return `${opening}<div class="gpi-activity-running" role="status" aria-live="polite" aria-busy="true">
          <div class="gpi-activity-live">
            <span class="gpi-running-spinner" aria-hidden="true"></span>
            <span class="gpi-activity-kicker">${esc(tr('In progress', '正在进行'))}</span>
            <span class="gpi-activity-title"><strong>${esc(title)}</strong></span>
            <span class="gpi-activity-elapsed" data-gpi-live-elapsed="${Number(row.startedAt || Date.now())}">${esc(durationText(row.startedAt))}</span>
          </div>
          ${list ? `<div class="gpi-activity-body gpi-trace-live${bodyClass}"${keyAttr}><div class="gpi-trace-window">${list}</div>${longTrace ? expandButton(expanded) : ''}</div>` : ''}
          <p class="gpi-activity-note">${esc(note)}</p>
        </div>`;
      }
      // Biomni-style traces are evidence of actual work. A plain response with
      // only submit/start/model lifecycle events should read as a plain response
      // instead of exposing an empty or misleading execution disclosure.
      if (!visibleSteps.length && !failed) return '';
      const title = failed ? tr('Execution needs attention', '执行过程需要处理') : kicker;
      // A job that stopped before any step has no row to carry its cause; it
      // is said under the summary, or the stop reads as nothing at all.
      const unplacedCause = failed && !failureExplained && !causes.size && row && row.failureCode
        && typeof host.runFailureText === 'function'
        ? String(host.runFailureText(row.failureCode) || '').trim() : '';
      const traceMeta = `${tr(`${rowCount} steps`, `${rowCount} 个步骤`)}${row.durationKnown === false ? '' : ` · ${tr(`total ${durationText(row.startedAt, row.endedAt)}`, `总耗时 ${durationText(row.startedAt, row.endedAt)}`)}`}`;
      // Finished failures expose their status in the summary but keep verbose
      // diagnostic receipts collapsed. Persisted rows from older builds may
      // still carry expanded=true, so terminal rendering must not trust it.
      return `${opening}<details class="gpi-activity ${failed ? 'error' : 'complete'}">
        <summary aria-label="${esc(`${title}. ${traceMeta}`)}">
          <span class="gpi-disclosure" aria-hidden="true">${iconHtml('chevron', 14)}</span>
          <span class="gpi-activity-kicker">${esc(title)}</span>
          <span class="gpi-activity-meta" aria-hidden="true">${esc(traceMeta)}</span>
        </summary>
        <div class="gpi-activity-body${bodyClass}"${keyAttr}>
          ${traceListHtml(visibleSteps, narration, causes)}
          ${longTrace ? expandButton(expanded) : ''}
          <p class="sr-only">${tr('Reasoning summaries, lifecycle facts, and EasyICU receipts — tool arguments, credentials, and patient rows are never displayed.', '这里展示模型的思考摘要、生命周期事实和 EasyICU 回执；工具参数、凭据和患者行级数据不会展示。')}</p>
        </div>
      </details>${unplacedCause ? `<p class="gpi-activity-note">${esc(unplacedCause)}</p>` : ''}`;
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

    return Object.freeze({ appendPublicDelta, durationText, executorName, finishTurn, focusLatest, notePlanSubStep, pipelineEventLabel, planStepFacts, planStepRow, planStepTitle, planSubStepFacts, planningEventFacts, planningProgressText, notePipelineEvent, finishPipelineSteps, pipelineTime, render, renderTimeline, startTurn, stepLabel, syncLiveClock, timeMs });
  }

  window.EasyICU.guidedPi.declare('activity', { create, traceExpanded });
})();
