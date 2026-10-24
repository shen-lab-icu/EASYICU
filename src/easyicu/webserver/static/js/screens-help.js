/* Owner: Help/Get Started route. */
/* Screens: Get Started (tutorial) + AI Assistant.
   Get Started orients new users to the workflow; Assistant is the
   evidence-bound sidebar AI. Assistant messages give workflow guidance and
   plan structure only — never invented scientific results. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  const { esc } = window.EU_HTML;

  S.llmEgressDisclosure = Object.freeze({
    noticeKey: 'physionet_llm_data_handling_declaration',
    sent: Object.freeze([
      'user_messages', 'analysis_plans_and_code', 'schema_and_concept_metadata',
      'aggregate_statistics_with_small_cell_suppression', 'structured_diagnostics',
      'project_artifacts_after_privacy_projection',
    ]),
  });

  /* ---------------- GET STARTED on a hosted instance ----------------
     A clinician reaches EasyICU on the hospital server: the administrator
     prepared the data and the model, and the conversation is the way in.
     The local page's promises (runs on your machine, connect a folder, bring
     your own token) are not true there, so the hosted page says what is.
     Facts come from their owners: deployment and prepared databases from
     /api/capabilities, the administrator from the hosted owner. */
  function hostedOwner() {
    const owner = window.EasyICU?.guidedPi?.optional('hosted');
    return owner && owner.active() ? owner : null;
  }
  const DATABASE_NAMES = { miiv: 'MIMIC-IV', eicu: 'eICU-CRD', aumc: 'AmsterdamUMCdb', hirid: 'HiRID', mimic: 'MIMIC-III', sic: 'SICdb' };
  function preparedDatabases() {
    const counts = (((window.EU_CAPABILITIES || {}).registered_sources || {}).by_database) || {};
    return Object.keys(DATABASE_NAMES).filter(key => Number(counts[key]) > 0).map(key => DATABASE_NAMES[key]);
  }
  function hostedRail(owner) {
    const contact = owner.supportContact();
    return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Where to go', '常用入口')}</span></div>
        <div class="col gap-6" style="font-size:12.5px;color:var(--ink-3);">
          <a class="nav-item" data-nav="guided" style="height:30px;"><span class="ico">${icon('agent', 15)}</span>${t('Research conversation', '研究对话')}</a>
          <a class="nav-item" data-nav="patient" style="height:30px;"><span class="ico">${icon('patient', 15)}</span>${t('Patient Review', '患者审阅')}</a>
        </div>
        ${contact ? `<div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Administrator', '管理员')}</div><div style="font-size:12.5px;color:var(--ink-2);">${esc(contact)}</div>` : ''}
      </div>`;
  }
  function hostedRender(owner) {
    const contact = owner.supportContact();
    const databases = preparedDatabases();
    const steps = [
      [t('Say what you want to study', '说出研究问题'), t('Describe the patients, the factor you care about and the outcome, in your own words. If something is missing, EasyICU asks.', '用自己的话说出研究人群、关心的因素和结局。缺少什么，EasyICU 会追问。')],
      [t('Confirm the data', '确认数据'), t('EasyICU picks a suitable database and asks you to confirm it; data for the study is prepared only after you do.', 'EasyICU 会选出合适的数据库并请你确认；确认后才会为这项研究准备数据。')],
      [t('Review the analysis plan', '审阅分析计划'), t('Before any analysis runs you see the whole plan: the study population, the variables and the statistical methods. It runs only after you approve it.', '分析开始前，你会看到完整的计划：研究人群、变量和统计方法。你批准后才会运行。')],
      [t('Read the results', '查看结果'), t('Results come as tables, figures and a written explanation, and every number comes from the run\'s own record. The result files open and download in the conversation.', '结果以表格、图和文字说明呈现，每个数字都来自这次运行的记录。结果文件可以在对话里查看和下载。')],
    ];
    const questions = [
      [t('Where is the data?', '数据在哪里？'), t('The research data is kept on the hospital server, prepared and managed by the administrator, and the analysis runs there too.', '研究数据保存在医院服务器上，由管理员准备和管理；分析也在服务器上运行。')],
      // Patient-level rows stay on the server (user decision 2026-10-22).
      [t('Can I download the data?', '可以下载数据吗？'), t('The tables, figures and written results of the research conversation can be downloaded. Patient-level data, including your extractions, stays on the hospital server and cannot be downloaded.', '研究对话里的表格、图和文字结果可以下载；患者级数据（包括你抽取的数据）只保存在医院服务器上，不能下载。')],
      [t('Which databases are ready?', '这里有哪些数据？'), databases.length
        ? t(`Ready on this server: ${databases.join(', ')}. For another database, contact the administrator.`, `这台服务器上已准备好：${databases.join('、')}。需要其他数据库，请联系管理员。`)
        : t('No database is ready on this server yet. Please contact the administrator.', '这台服务器上还没有准备好的数据库，请联系管理员。')],
      [t('Does it use an AI model?', '会用到 AI 模型吗？'), t('Yes. The research conversation runs on a model the administrator set up. It receives your messages, the analysis plan and its code, variable names, error details, and summary statistics in which very small counts are hidden. Patient-level rows are never sent. For credentialed databases such as MIMIC-IV and eICU, a model is used only when the administrator has declared that it runs on this server or that its service keeps no data, as the PhysioNet data use agreement requires.', '会。研究对话由管理员配置的模型支持。发给模型的有：你的消息、分析计划及其代码、变量名称、错误详情，以及隐藏了极小计数的汇总统计。患者的逐行记录不会发给模型。对于 MIMIC-IV、eICU 等需授权的数据库，只有在管理员声明该模型运行在本服务器上、或其服务不保留数据时才会使用，这是 PhysioNet 数据使用协议的要求。')],
      [t('Can the results guide a patient\'s treatment?', '结果可以用于临床诊疗吗？'), t('No. EasyICU results are for research. They do not replace clinical judgement for an individual patient.', '不能。EasyICU 的结果用于科研，不能替代对个体患者的临床判断。')],
      [t('Do I need to set up a model or an API key?', '需要自己配置模型或 API key 吗？'), t('No. The administrator sets up the model for everyone.', '不需要，模型由管理员统一配置。')],
      [t('What if something goes wrong?', '遇到问题怎么办？'), t(`At the top of the research conversation, open “Having a problem?” and copy the problem reference for the administrator${contact ? ` (${contact})` : ''}. It holds only the codes of the project, conversation and run, never patient information.`, `在研究对话顶部点「遇到问题」，复制问题编号发给管理员${contact ? `（${contact}）` : ''}。编号只包含项目、对话和运行的代号，不含患者信息。`)],
    ];
    return `
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">${t('Get started', '快速上手')}</div>
        <h1 style="margin-top:6px;">${t('One conversation, one ICU study', '用一次对话完成一项 ICU 研究')}</h1>
        <p class="lead">${t('EasyICU runs on the hospital server, where the administrator has prepared the data and the model. Say what you want to study; the conversation takes you through the rest, step by step.', 'EasyICU 运行在医院服务器上，数据和模型都由管理员准备好。你只需要说出想研究的问题，其余步骤会在对话里一步步带你完成。')}</p>
      </div>

      <div class="guide-hero" data-help-hosted>
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;">
          <div style="max-width:60ch;">
            <div class="row gap-10" style="align-items:center;">
              <div class="mode-mark accent" style="width:34px;height:34px;">${icon('agent', 17)}</div>
              <div style="font-size:17px;font-weight:600;letter-spacing:-0.01em;">${t('Start with a question', '从一个问题开始')}</div>
            </div>
            <p style="font-size:12.75px;color:var(--ink-3);line-height:1.6;margin:12px 0 0;">${t('For example: “In eICU, is the highest lactate on the first ICU day associated with in-hospital death?” Plain words are fine; the conversation also offers example questions to start from.', '例如：「在 eICU 中，入 ICU 第一天的最高乳酸与院内死亡是否相关？」用平常的话说就行，对话里也有现成的示例问题可以直接选用。')}</p>
          </div>
          <div class="col gap-8" style="flex:none;">
            <button class="btn primary lg" data-nav="guided" type="button">${t('Start a research conversation', '开始研究对话')} ${icon('arrow', 15)}</button>
          </div>
        </div>
      </div>

      <div class="sec-stack"><div class="lbl">${t('Four steps', '四个步骤')}</div><h2>${t('How a study moves forward', '一项研究怎样推进')}</h2></div>
      <div class="card pad">
        <div class="guide-steps">
          ${steps.map(([title, desc], i) => `
            <div class="guide-step ${i === 0 ? 'accent' : ''}">
              <div class="gs-num">${i + 1}</div>
              <div>
                <div class="gs-t">${title}</div>
                <div class="gs-d">${desc}</div>
              </div>
            </div>`).join('')}
        </div>
      </div>

      <div class="sec-stack"><div class="lbl">${t('Good to know', '你可能关心')}</div><h2>${t('Common questions', '常见问题')}</h2></div>
      <div class="faq">
        ${questions.map(([q, a], i) => `
          <div class="faq-item ${i === 0 ? 'open' : ''}">
            <button class="faq-q">${q}<span class="fq-chev">${icon('chevron', 14)}</span></button>
            <div class="faq-a">${esc(a)}</div>
          </div>`).join('')}
      </div>`;
  }

  /* ---------------- GET STARTED ---------------- */
  S.tutorial = {
    section: 'tutorial', nav: 'tutorial',
    crumbs: ['Home', 'Get Started'],
    // The module shell's Help aside on a hosted instance (app.js).
    contextAside() {
      const hosted = hostedOwner();
      const contact = hosted && hosted.supportContact();
      return contact ? `<div class="euh-context-contact"><span>${t('Administrator', '管理员')}</span><strong>${esc(contact)}</strong></div>` : '';
    },
    actionHtml() {
      if (hostedOwner()) return `<button class="btn primary" data-nav="guided" type="button">${t('Start a research conversation', '开始研究对话')}</button>`;
      return `<button class="btn primary" data-help-startdemo type="button">${icon('play', 13)} ${t('Start demo', '开始演示')}</button>`;
    },
    rail() {
      const hosted = hostedOwner();
      if (hosted) return hostedRail(hosted);
      const workflow = [
        [t('Extract', '抽取'), 'extract', 'extraction'],
        [t('Review', '审阅'), 'patient', 'patient'],
        [t('Benchmark', '基准'), 'benchmark', 'crossdb'],
        [t('Agent', 'Agent'), 'agent', 'agent'],
      ];
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('The workflow', '工作流程')}</span></div>
        <div class="col gap-6" style="font-size:12.5px;color:var(--ink-3);">
          ${workflow.map(([label, ic, nav]) =>
            `<a class="nav-item" data-nav="${nav}" style="height:30px;"><span class="ico">${icon(ic, 15)}</span>${label}</a>`).join('')}
        </div>
        <div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Shortcuts', '快捷键')}</div>
        <div class="col" style="font-size:12px;">
          <div class="shortcut-row"><span style="color:var(--ink-3);">${t('Open EasyICU Copilot', '打开 EasyICU 研究助手')}</span><span class="keys"><span class="kbd">⌘</span><span class="kbd">K</span></span></div>
          <div class="shortcut-row"><span style="color:var(--ink-3);">${t('Switch section', '切换区域')}</span><span class="keys"><span class="kbd">1</span>–<span class="kbd">5</span></span></div>
          <div class="shortcut-row"><span style="color:var(--ink-3);">${t('Toggle language', '切换语言')}</span><span class="keys"><span class="kbd">L</span></span></div>
        </div>
      </div>`;
    },
    render() {
      const hosted = hostedOwner();
      if (hosted) return hostedRender(hosted);
      const steps = [
        [
          '1',
          t('Frame the question', '确定研究问题'),
          t(
            'Pick a data mode — Demo generates reproducible mock data, Real connects a local export folder (nothing is ever uploaded) — then pin down what you’re asking: an outcome, a time window, and a comparator.',
            '先选择数据模式：演示模式生成可复现的模拟数据，真实模式连接本地导出文件夹（不会上传任何内容）；然后确定结局、时间窗和比较对象。',
          ),
          [[t('Configure source', '配置数据源'), 'extraction'], [t('Talk it through', '对话式引导'), 'guided']],
          true,
        ],
        [
          '2',
          t('Extract & verify the data', '抽取并核验数据'),
          t(
            'One click runs the recommended extraction — cohort, coverage-audited feature modules, and a reproducible export. Need more control? Open Customize to set every detail. Nothing runs on incomplete data.',
            '一键运行推荐抽取：队列、覆盖率审计过的特征模块和可复现导出。需要更多控制时，可以进入自定义配置；数据不完整时不会继续运行。',
          ),
          [[t('Open extraction', '打开数据抽取'), 'extraction']],
          false,
        ],
        [
          '3',
          t('Review & explore', '审阅与探索'),
          t(
            'Inspect patient-level tables and time series, step up to cohort contrasts and SOFA reclassification, then benchmark one cohort definition across two or more ICU databases.',
            '查看单患者表格和时间序列，再做队列对比、SOFA 重分类，并把同一个队列定义放到两个或更多 ICU 数据库中做基准比较。',
          ),
          [[t('Patient Review', '患者审阅'), 'patient'], [t('Cohort Statistics', '队列统计'), 'cohort'], [t('Cross-database comparison', '跨库对比'), 'crossdb']],
          false,
        ],
        [
          '4',
          t('Analyze & draft', '分析与撰稿'),
          t(
            'Plan, run, and review an auditable pipeline with the Research Agent. The manuscript draft stays locked until every evidence check passes and you confirm.',
            '使用 Research Agent 规划、运行并审阅可审计流程。只有证据检查全部通过并由你确认后，手稿草稿才会进入下一步。',
          ),
          [[t('Open conversation', '打开研究对话'), 'agent'], [t('Start Guided Copilot', '开始研究引导'), 'guided']],
          false,
        ],
      ];
      const questions = [
        [
          t('Is any patient data uploaded?', '会上传患者数据吗？'),
          t(
            'Extraction, review, and analysis run on your machine. If you enable model calls, your messages, analysis plans and code, variable names, error details, and summary statistics with very small counts hidden may be sent to the configured model. The data tools block patient-level rows from model requests. Do not paste patient records into messages.',
            '抽取、审阅和分析在本机运行。开启模型调用后，你的消息、分析计划及代码、变量名称、错误详情，以及隐藏了极小计数的汇总统计可能发送给已配置的模型。数据工具会阻止患者逐行记录进入模型请求；请勿在消息中粘贴患者记录。',
          ),
        ],
        [
          t('What exactly is Demo Mode?', '演示模式是什么？'),
          t(
            'A reproducible synthetic dataset — by default 10 ICU stays over 24 hours across 19 feature modules. It produces no scientific findings; every number is a seeded placeholder so you can learn the interface safely.',
            '一个可复现的合成数据集：默认 10 个 ICU stay、24 小时、19 个特征模块。它不产生科学发现，所有数字都是带种子的占位结果，方便你安全熟悉界面。',
          ),
        ],
        [
          t('Why is the manuscript draft locked?', '为什么手稿草稿是锁定的？'),
          t(
            'Drafting is a deliberate second stage. The agent only writes after each evidence check passes — denominators resolved, coverage above threshold, tables reproducing from the manifest — and after a reviewer signs off. Claims stay traceable to logged artifacts.',
            '撰稿是明确的第二阶段。只有分母、覆盖率、表格复现等证据检查通过，并且人工确认后，Agent 才会继续写作。每个 claim 都要能追溯到日志化 artifact。',
          ),
        ],
        [
          t('Which databases are supported?', '支持哪些数据库？'),
          t(
            'Six standardized ICU sources: MIMIC-IV, eICU-CRD, AmsterdamUMCdb, HiRID, MIMIC-III, and SICdb. EasyICU detects known export layouts when you select a local folder.',
            '目前支持六个标准化 ICU 数据源：MIMIC-IV、eICU-CRD、AmsterdamUMCdb、HiRID、MIMIC-III 和 SICdb。选择本地文件夹后，EasyICU 会识别已知导出布局。',
          ),
        ],
        [
          t('Do I need API tokens?', '需要 API token 吗？'),
          t(
            'Not for Demo Mode or any extraction, review, or benchmark work — those never call a model. Tokens only apply if you connect an external model endpoint for the Research Agent.',
            '演示模式、数据抽取、审阅和跨库对比都不需要 token，这些不会调用模型。只有你为 Research Agent 连接外部模型端点时才需要 token。',
          ),
        ],
      ];
      return `
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">${t('Get started', '快速上手')}</div>
        <h1 style="margin-top:6px;">${t('A quiet, reviewable path from data to draft', '从数据到草稿，一条安静、可审阅的路径')}</h1>
        <p class="lead">${t('EasyICU runs entirely on your machine. Start with reproducible demo data to learn the flow, then point it at local ICU exports when you’re ready.', 'EasyICU 完全在你的机器上运行。你可以先用可复现的演示数据熟悉流程，准备好后再连接本地 ICU 导出。')}</p>
      </div>

      <div class="guide-hero">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;">
          <div style="max-width:60ch;">
            <div class="row gap-10" style="align-items:center;">
              <div class="mode-mark accent" style="width:34px;height:34px;">${icon('flask', 17)}</div>
              <div style="font-size:17px;font-weight:600;letter-spacing:-0.01em;">${t('New here? Take the 2-minute demo tour', '第一次使用？先跑 2 分钟演示')}</div>
            </div>
            <p style="font-size:12.75px;color:var(--ink-3);line-height:1.6;margin:12px 0 0;">${t('No tokens, no setup, no patient data. The demo generates 10 mock ICU stays so every screen, table, and review check is fully explorable before you connect anything real.', '不需要 token、不需要配置、不需要患者数据。演示会生成 10 个模拟 ICU stay，让你在连接真实数据前先完整探索每个页面、表格和审阅检查。')}</p>
          </div>
          <div class="col gap-8" style="flex:none;">
            <button class="btn primary lg" data-help-startdemo type="button">${icon('play', 15)} ${t('Start demo', '开始演示')}</button>
          </div>
        </div>
      </div>

      <div class="sec-stack"><div class="lbl">${t('The four stages', '四个阶段')}</div><h2>${t('How a study moves through EasyICU', '一个研究如何在 EasyICU 中推进')}</h2></div>
      <div class="card pad">
        <div class="guide-steps">
          ${steps.map(([n, title, desc, actions, acc]) => `
            <div class="guide-step ${acc ? 'accent' : ''}">
              <div class="gs-num">${n}</div>
              <div>
                <div class="gs-t">${title}</div>
                <div class="gs-d">${desc}</div>
                <div class="gs-actions">
                  ${actions.map(([lab, nav], i) => `<button class="btn ${i === 0 ? 'primary' : ''} sm" data-nav="${nav}">${lab} ${icon('arrow', 13)}</button>`).join('')}
                </div>
              </div>
            </div>`).join('')}
        </div>
      </div>

      <div class="sec-stack"><div class="lbl">${t('Good to know', '你可能关心')}</div><h2>${t('Common questions', '常见问题')}</h2></div>
      <div class="faq">
        ${questions.map(([q, a], i) => `
          <div class="faq-item ${i === 0 ? 'open' : ''}">
            <button class="faq-q">${q}<span class="fq-chev">${icon('chevron', 14)}</span></button>
            <div class="faq-a">${a}</div>
          </div>`).join('')}
      </div>`;
    },
    afterRender(root) {
      root.querySelectorAll('.faq-q').forEach(q => {
        q.addEventListener('click', () => q.closest('.faq-item').classList.toggle('open'));
      });
      // 'Start demo' must deliver a demo: switch the data mode first (with the
      // usual work-protection confirm), and only navigate once the switch
      // actually applied — a cancelled confirm stays on Get Started.
      root.querySelectorAll('[data-help-startdemo]').forEach(b => b.addEventListener('click', () => {
        if (window.EU_DATA !== 'demo' && window.setDataMode) {
          window.setDataMode('demo', { onapply: () => { location.hash = '#extraction'; } });
          return;
        }
        location.hash = '#extraction';
      }));
    },
  };

})();
