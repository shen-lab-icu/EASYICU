/* Screens: Get Started (tutorial) + AI Assistant.
   Get Started orients new users to the workflow; Assistant is the
   evidence-bound sidebar AI. Assistant messages give workflow guidance and
   plan structure only — never invented scientific results. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  /* ---------------- GET STARTED ---------------- */
  S.tutorial = {
    section: 'tutorial', nav: 'tutorial',
    crumbs: ['Home', 'Get Started'],
    actionHtml() {
      return `<button class="btn primary" data-help-startdemo type="button">${icon('play', 13)} ${t('Start demo', '开始演示')}</button>`;
    },
    rail() {
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
          <div class="shortcut-row"><span style="color:var(--ink-3);">${t('Open Page guide', '打开页面指南')}</span><span class="keys"><span class="kbd">⌘</span><span class="kbd">K</span></span></div>
          <div class="shortcut-row"><span style="color:var(--ink-3);">${t('Switch section', '切换区域')}</span><span class="keys"><span class="kbd">1</span>–<span class="kbd">5</span></span></div>
          <div class="shortcut-row"><span style="color:var(--ink-3);">${t('Toggle language', '切换语言')}</span><span class="keys"><span class="kbd">L</span></span></div>
        </div>
      </div>`;
    },
    render() {
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
          [[t('Open Agent Projects', '打开研究项目'), 'agent'], [t('Start Guided Copilot', '开始研究引导'), 'guided']],
          false,
        ],
      ];
      const questions = [
        [
          t('Is any patient data uploaded?', '会上传患者数据吗？'),
          t(
            'No. EasyICU is local-first and the guarantee is enforced — extraction, review, and analysis all run on your machine. The only thing that can ever leave (and only if you explicitly enable it) is the Research Agent’s plan text, never patient rows.',
            '不会。EasyICU 是 local-first，抽取、审阅和分析都在你的机器上运行。只有在你显式开启时，Research Agent 的计划文本才可能离开本机；患者行级数据不会上传。',
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
