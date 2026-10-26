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
  function preparedDatabases() {
    const counts = (((window.EU_CAPABILITIES || {}).registered_sources || {}).by_database) || {};
    const names = window.EU_DATABASE_NAMES;
    return names.keys().filter(key => Number(counts[key]) > 0).map(key => names.name(key));
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

  /* ---------------- GET STARTED on a local install ----------------
     The clinician's own computer: their model, their PhysioNet data and the
     public demos. Conversation first; the data workspace modules stay in the
     rail. Facts come from their owners: the demo catalogue
     (demo_source_contracts.py), the model connection and its credentialed-data
     declaration (screens-guided-pi-provider*.js), the conversation's source
     offer (screens-guided-pi-data-consent.js). */
  const LOCAL_EXAMPLES = Object.freeze([
    ['In the eICU demo, how does ICU length of stay differ across ICU types?',
      '在 eICU demo 中，不同 ICU 类型的患者，ICU 住院时长有什么差别？'],
    ['In the eICU demo, how much higher is the in-hospital mortality risk of patients whose highest lactate in the first 24 ICU hours is 4 mmol/L or more, compared with below 2 mmol/L? Adjust for age.',
      'eICU demo 中，入 ICU 24 小时内最高乳酸 ≥4 mmol/L 与 <2 mmol/L 的患者相比，院内死亡风险高多少？调整年龄。'],
    ['Across all ICU stays in the eICU demo, what is the in-hospital mortality of men and of women?',
      '在 eICU demo 全部 ICU 入住中，男性和女性的院内死亡率分别是多少？'],
  ]);
  function localRender() {
    const setup = [
      [t('Connect a model', '连接模型'),
        t('The research conversation uses the model you connect to read your question and draft the plan. The first time you open it, choose a service, such as DeepSeek, an OpenAI-compatible service or a model running on this computer, and paste its API key. Data extraction, patient review, cohort statistics and cross-database comparison need no model.',
          '研究对话用你连接的模型来理解问题、拟定计划。第一次打开研究对话时，选择一个模型服务（如 DeepSeek、OpenAI 兼容服务，或在本机运行的模型），填入它的 API 凭据。数据抽取、患者审阅、队列统计和跨库对比不需要模型。')],
      [t('Try it on the public demo', '先用公开 Demo 试一遍'),
        t('Name the “eICU demo” in your question. EasyICU offers to download the official eICU demo (130.6 MB) from PhysioNet, checks it and prepares it on this computer. The public demo needs no data use agreement. Click a question below to start a conversation with it.',
          '在问题里写上「eICU demo」，EasyICU 会提出从 PhysioNet 下载官方 eICU Demo（130.6 MB），校验后在本机准备好。公开 Demo 不需要签数据使用协议。点下面的问题，会用它开始一次对话。')],
      [t('Use your own database', '接入你自己的数据库'),
        t('If you have access to MIMIC-IV or eICU-CRD, choose that database in the conversation and select the folder you downloaded from PhysioNet; EasyICU prepares it on this computer. The first preparation of a full database takes a while; later studies reuse it. The first time a study uses these data, EasyICU asks whether the model runs on this computer or its service keeps no data. The PhysioNet data use agreement requires it, and without it EasyICU does not use the model for these data.',
          '已获得 MIMIC-IV 或 eICU-CRD 使用资格的，在研究对话里选择这个数据库，再选中你从 PhysioNet 下载的文件夹，EasyICU 会在本机准备好。完整数据库第一次准备要花较长时间，之后的研究会复用。研究第一次用到这些数据时，EasyICU 会问你：模型是否在本机运行，或服务是否不保留数据。这是 PhysioNet 数据使用协议的要求；没有说明时，EasyICU 不会用这个模型处理这些数据。')],
    ];
    const steps = [
      [t('Say what you want to study', '说出研究问题'), t('Describe the patients, the factor you care about and the outcome, in your own words. If something is missing, EasyICU asks.', '用自己的话说出研究人群、关心的因素和结局。缺什么，EasyICU 会问你。')],
      [t('Confirm the data', '确认数据'), t('EasyICU proposes the data for the question and asks you to confirm it; data for the study is prepared only after you do.', 'EasyICU 会为问题选出数据并请你确认；确认后才准备这项研究要用的数据。')],
      [t('Review the analysis plan', '审阅分析计划'), t('Before any analysis runs you see the whole plan: the study population, the variables and the statistical methods. It runs only after you approve it.', '分析开始前，你会看到完整的计划：研究人群、变量和统计方法。你批准后才会运行。')],
      [t('Read the results', '查看结果'), t('Results come as tables, figures and a written explanation, and every number comes from the run\'s own record. Ask follow-up questions in the same conversation.', '结果以表格、图和文字说明呈现，每个数字都来自这次运行的记录。可以在同一个对话里继续追问。')],
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
        t('Which studies does this version run?', '这个版本能做哪些研究？'),
        t(
          'Descriptive, association, prediction, survival, subtype or trajectory, and causal studies (target trial emulation) on MIMIC-III, MIMIC-IV, eICU, HiRID, AmsterdamUMCdb and SICdb. What a study can do depends on what the database records: when it lacks something the question needs, the plan states the substitute and runs only after you approve it, or the conversation says what is missing before planning. None of the six has yet passed acceptance through a normal real-data run; the Data map marks each database.',
          'MIMIC-III、MIMIC-IV、eICU、HiRID、AmsterdamUMCdb、SICdb 上的描述、关联、预测、生存分析、亚型或轨迹，以及因果研究（目标试验模拟）。能做到哪一步取决于该库记录了哪些信息：缺少问题需要的信息时，计划会写明替代口径，你批准后才运行；无法替代时，对话会在拟定计划前说明缺少什么。目前六个库都还没有通过正常入口的真实数据运行验收，数据地图逐库标明。',
        ),
      ],
      [
        t('Can the results guide a patient\'s treatment?', '结果可以用于临床诊疗吗？'),
        t('No. EasyICU results are for research. They do not replace clinical judgement for an individual patient.', '不能。EasyICU 的结果用于科研，不能替代对个体患者的临床判断。'),
      ],
      [
        t('What is a “Public demo” in the data workspace?', '数据工作台里的「公开 Demo」是什么？'),
        t(
          'A small official release of deidentified real records from PhysioNet (MIMIC-IV, eICU-CRD, MIMIC-III). Once prepared on this computer it goes through the same conversion, mapping and export as your own data, and its row says “Public demo”.',
          'PhysioNet 官方发布的小规模去标识化真实记录（MIMIC-IV、eICU-CRD、MIMIC-III）。在本机准备好后，和你自己的数据走同样的转换、映射和导出，数据行上标着「公开 Demo」。',
        ),
      ],
      [
        t('What if something goes wrong?', '遇到问题怎么办？'),
        t(
          'At the top of the research conversation, open “Having a problem?” and copy the problem reference; under Settings → About you can export diagnostics. Send them to the colleague who gave you EasyICU.',
          '在研究对话顶部打开「遇到问题」，复制问题编号；在「设置 → 关于」里可以导出诊断。把它们发给为你提供 EasyICU 的同事。',
        ),
      ],
    ];
    return `
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">${t('Get started', '快速上手')}</div>
        <h1 style="margin-top:6px;">${t('One conversation, one ICU study', '用一次对话完成一项 ICU 研究')}</h1>
        <p class="lead">${t('EasyICU runs on your computer, with your model and your data. Say what you want to study; the conversation takes you through the rest, step by step.', 'EasyICU 在你的电脑上运行，用你连接的模型和你的数据。说出想研究的问题，对话会一步步带你完成后面的事。')}</p>
      </div>
      <div class="guide-hero">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;">
          <div style="max-width:60ch;">
            <div class="row gap-10" style="align-items:center;">
              <div class="mode-mark accent" style="width:34px;height:34px;">${icon('agent', 17)}</div>
              <div style="font-size:17px;font-weight:600;letter-spacing:-0.01em;">${t('See a whole study first', '先看一项完整的研究')}</div>
            </div>
            <p style="font-size:12.75px;color:var(--ink-3);line-height:1.6;margin:12px 0 0;">${t('A read-only replay of a real MIMIC-IV run, from the question to the results. It needs no model and changes nothing.', '回放一次真实的 MIMIC-IV 运行，从提问到结果。只读，不需要模型，也不会改动你的项目。')}</p>
          </div>
          <div class="col gap-8" style="flex:none;">
            <button class="btn primary lg" data-help-workflow-demo type="button">${icon('play', 15)} ${t('View the workflow demo', '查看流程演示')}</button>
          </div>
        </div>
      </div>
      <div class="sec-stack"><div class="lbl">${t('First time', '第一次使用')}</div><h2>${t('Three things to set up', '三件事')}</h2></div>
      <div class="card pad">
        <div class="guide-steps">
          ${setup.map(([title, desc], i) => `
            <div class="guide-step ${i === 0 ? 'accent' : ''}">
              <div class="gs-num">${i + 1}</div>
              <div>
                <div class="gs-t">${title}</div>
                <div class="gs-d">${desc}</div>
                ${i === 0 ? `<div class="gs-actions"><button class="btn primary sm" data-nav="guided" type="button">${t('Open conversation', '打开研究对话')} ${icon('arrow', 13)}</button></div>` : ''}
                ${i === 1 ? `<div class="gs-actions" style="flex-direction:column;align-items:flex-start;">${LOCAL_EXAMPLES.map(([en, zh]) => `<button class="btn sm" data-help-question="${esc(t(en, zh))}" type="button">${esc(t(en, zh))}</button>`).join('')}</div>` : ''}
              </div>
            </div>`).join('')}
        </div>
      </div>
      <div class="sec-stack"><div class="lbl">${t('Four steps', '四个步骤')}</div><h2>${t('How a study moves forward', '一项研究怎样推进')}</h2></div>
      <div class="card pad">
        <div class="guide-steps">
          ${steps.map(([title, desc], i) => `
            <div class="guide-step">
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
      return `<button class="btn primary" data-nav="guided" type="button">${t('Start a research conversation', '开始研究对话')}</button>`;
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
      return localRender();
    },
    afterRender(root) {
      root.querySelectorAll('.faq-q').forEach(q => {
        q.addEventListener('click', () => q.closest('.faq-item').classList.toggle('open'));
      });
      // The workflow replay and the example questions open in the research
      // conversation. Each leaves a one-shot request it reads on mount: the
      // replay (screens-guided-pi.js) and the question draft it already takes
      // from the Skill Hub.
      root.querySelectorAll('[data-help-workflow-demo]').forEach(b => b.addEventListener('click', () => {
        try { window.sessionStorage.setItem('easyicu.help.workflowDemo', '1'); } catch (_) {}
        location.hash = '#guided';
      }));
      root.querySelectorAll('[data-help-question]').forEach(b => b.addEventListener('click', () => {
        try { window.sessionStorage.setItem('easyicu.skillHub.question', b.getAttribute('data-help-question') || ''); } catch (_) {}
        location.hash = '#guided';
      }));
    },
  };

})();
