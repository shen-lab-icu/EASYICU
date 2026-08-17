/* Guided Copilot progressive extraction + study-design stepper.
   Owner file for the "Prepare data inside Copilot" inline card. screens-guided.js
   owns state (guidedExtract / guidedDesign), event wiring, and the extraction job;
   this module owns render-only HTML and the study-design vocabulary (outcome /
   time-window / comparator presets + their resolvers), passed a ctx object.

   Progressive disclosure: instead of one bubble with the whole classic form, the
   card walks source -> cohort -> study design -> modules -> export -> review, one
   step at a time, so a researcher makes decisions in the order they actually think
   about them (question -> population -> endpoint/window/comparator -> features ->
   where it lands -> confirm). */
(function () {
  'use strict';

  /* ---- study-design vocabulary (owned here) ---- */
  const OUTCOME_PRESETS = [
    ['inhosp_mortality', 'In-hospital mortality', '院内死亡'],
    ['mortality_28d', '28-day mortality', '28 天死亡'],
    ['icu_mortality', 'ICU mortality', 'ICU 死亡'],
    ['icu_los', 'ICU length of stay', 'ICU 住院时长'],
    ['aki_onset', 'AKI onset', 'AKI 发生'],
    ['vent_free', 'Ventilator-free days', '无呼吸机天数'],
    ['custom', 'Custom endpoint…', '自定义结局…'],
  ];
  const WINDOW_PRESETS = [
    ['first_24h', 'First 24 hours', '前 24 小时', 24],
    ['first_48h', 'First 48 hours', '前 48 小时', 48],
    ['first_72h', 'First 72 hours', '前 72 小时', 72],
    ['whole_stay', 'Whole stay (30-day cap)', '全程（30 天上限）', 24 * 30],
  ];
  const COMPARATOR_PRESETS = [
    ['none', 'Single cohort (descriptive)', '单队列（描述性）'],
    ['exposure', 'Split by exposure group', '按暴露分组比较'],
    ['marker', 'Model with vs. without a marker', '含/不含某标志物建模'],
    ['custom', 'Custom comparison…', '自定义比较…'],
  ];

  function label(presets, key, t) {
    const row = presets.find(r => r[0] === key);
    return row ? (t ? t(row[1], row[2]) : row[1]) : '';
  }
  function resolveOutcome(design, t) {
    if (!design || !design.outcome) return '';
    if (design.outcome === 'custom') return String(design.outcomeCustom || '').trim();
    return label(OUTCOME_PRESETS, design.outcome, t);
  }
  function resolveComparator(design, t) {
    if (!design || !design.comparator) return '';
    if (design.comparator === 'custom') return String(design.comparatorCustom || '').trim();
    return label(COMPARATOR_PRESETS, design.comparator, t);
  }
  function windowHours(design) {
    const key = design && design.window ? design.window : 'whole_stay';
    const row = WINDOW_PRESETS.find(r => r[0] === key);
    return row ? row[3] : 24 * 30;
  }
  function windowLabel(design, t) {
    const key = design && design.window ? design.window : 'whole_stay';
    return label(WINDOW_PRESETS, key, t) || (t ? t('Whole stay (30-day cap)', '全程（30 天上限）') : 'Whole stay');
  }

  const STEPS = [
    ['source', 'Source', '数据源'],
    ['cohort', 'Cohort', '队列'],
    ['design', 'Design', '研究设计'],
    ['modules', 'Modules', '特征模块'],
    ['export', 'Export', '导出'],
    ['review', 'Review', '确认'],
  ];
  const STEP_ORDER = STEPS.map(s => s[0]);

  function stepIndicator(ctx, current) {
    const t = ctx.t;
    const curIdx = STEP_ORDER.indexOf(current);
    return `<div class="gdx-steps" role="tablist">${STEPS.map(([id, en, zh], i) => {
      const state = i < curIdx ? 'done' : i === curIdx ? 'on' : 'ahead';
      return `<button type="button" class="gdx-step ${state}" data-gx-goto-step="${id}" role="tab" aria-selected="${i === curIdx}">
        <span class="gdx-step-n">${i < curIdx ? ctx.icon('check', 10, 3) : (i + 1)}</span>${t(en, zh)}
      </button>`;
    }).join('<span class="gdx-step-sep"></span>')}</div>`;
  }

  /* one-line recap of decisions the user already made */
  function recap(ctx) {
    const t = ctx.t, ex = ctx.ex, design = ctx.design;
    const scan = ex.scan || {};
    const bits = [];
    if (scan.ok) bits.push(`${ctx.esc(scan.db || 'local')} · ${ctx.fmtInt(scan.tables, 'n/a')} ${t('tables', '表')}`);
    const cohortRow = ctx.cohortPresets.find(c => c[0] === ex.cohort);
    if (cohortRow) bits.push(t(cohortRow[1], cohortRow[2]));
    const outcome = resolveOutcome(design, t);
    if (outcome) bits.push(outcome);
    bits.push(windowLabel(design, t));
    if (ex.modules.length) bits.push(`${ex.modules.length} ${t('modules', '模块')}`);
    if (!bits.length) return '';
    return `<div class="gdx-recap">${bits.map(b => `<span>${ctx.esc(b)}</span>`).join('')}</div>`;
  }

  /* ---- per-step bodies ---- */
  function bodySource(ctx) {
    const t = ctx.t, ex = ctx.ex;
    const scan = ex.scan || {};
    const sourceMeta = scan.ok
      ? `${ctx.esc(scan.db || 'Unknown')} · ${ctx.esc(scan.source || 'source')} · ${ctx.fmtInt(scan.tables, 'n/a')} ${t('tables', '表')} · ${ctx.fmtInt(scan.modules, 'n/a')} ${t('modules', '模块')}`
      : t('No path is prefilled because every machine is different. Paste or choose a local ICU folder, then analyze it.', '不会预填路径，因为每台电脑都不同。请粘贴或选择本机 ICU 文件夹，然后识别目录。');
    return `
      <div class="gdx-body">
        <div class="gdx-steplead">${t('Where is your ICU data? I scan it locally first — nothing is uploaded.', '你的 ICU 数据在哪里？我会先在本地识别目录，不会上传任何数据。')}</div>
        <div class="gdx-source ${ctx.ready ? '' : 'blocked'}">
          <span>${ctx.icon(ctx.ready ? 'check' : 'shield', 12)}</span>
          <div><strong>${t('Local source', '本地数据源')}</strong><small>${sourceMeta}</small></div>
        </div>
        <div class="gdx-pathrow">
          <label>
            <span>${t('Data folder path', '数据文件夹路径')}</span>
            <input data-gx-path value="${ctx.attr(ex.path || '')}" placeholder="${ctx.attr(t('Paste or browse to a local ICU folder', '粘贴或选择本机 ICU 文件夹'))}" />
          </label>
          <button type="button" class="btn primary" data-gx-analyze ${ex.scanning ? 'disabled' : ''}>${ctx.icon('search', 13)} ${t('Analyze folder', '识别目录')}</button>
        </div>
        ${scan.ok && scan.source === 'module' ? `<div class="gdx-note warn">${t('This folder is already an EasyICU export. Register it for review instead of extracting again.', '该文件夹已是 EasyICU 导出。应注册后审阅，无需再次抽取。')}</div>` : ''}
      </div>`;
  }
  function bodyCohort(ctx) {
    const t = ctx.t, ex = ctx.ex;
    return `
      <div class="gdx-body">
        <div class="gdx-steplead">${t('Which patients are your denominator? This defines every downstream rate.', '哪些患者构成你的分母？它决定后续所有比率。')}</div>
        <div class="gdx-presets">
          ${ctx.cohortPresets.map(([key, en, zh, den, dzh]) => `
            <button type="button" class="gdx-preset ${ex.cohort === key ? 'on' : ''}" data-gx-cohort="${ctx.attr(key)}">
              <strong>${t(en, zh)}</strong><span>${t(den, dzh)}</span>
            </button>`).join('')}
        </div>
      </div>`;
  }
  function bodyDesign(ctx) {
    const t = ctx.t, design = ctx.design;
    const seg = (presets, cur, attr, custom, customAttr, placeholder) => `
      <div class="gdx-seg wrap" role="group">
        ${presets.map(([key, en, zh]) => `<button type="button" class="${cur === key ? 'on' : ''}" data-${attr}="${key}">${t(en, zh)}</button>`).join('')}
      </div>
      ${cur === 'custom' ? `<input class="gdx-custom" data-${customAttr} value="${ctx.attr(custom || '')}" placeholder="${ctx.attr(placeholder)}" />` : ''}`;
    return `
      <div class="gdx-body">
        <div class="gdx-steplead">${t('What is the study actually testing? I record the endpoint, window, and comparison so the analysis has a clear target — not a vague aim.', '这个研究到底在检验什么？我会记录结局、观察窗和比较方式，让后续分析有明确目标，而不是含糊的方向。')}</div>
        <div class="gdx-field">
          <div class="gdx-label">${t('Primary outcome / endpoint', '主要结局 / 终点')}</div>
          ${seg(OUTCOME_PRESETS, design.outcome, 'gx-outcome', design.outcomeCustom, 'gx-outcome-custom', t('e.g. 90-day mortality, delirium onset', '例如 90 天死亡、谵妄发生'))}
          ${!design.outcome ? `<div class="gdx-hint">${t('Pick or type the endpoint your question is really about.', '选择或输入你的问题真正关心的终点。')}</div>` : ''}
        </div>
        <div class="gdx-field">
          <div class="gdx-label">${t('Observation window', '观察窗')}</div>
          <div class="gdx-seg wrap" role="group">
            ${WINDOW_PRESETS.map(([key, en, zh]) => `<button type="button" class="${design.window === key ? 'on' : ''}" data-gx-window="${key}">${t(en, zh)}</button>`).join('')}
          </div>
          <div class="gdx-hint">${t('The window caps how much of each stay is pulled (hours after ICU admission). Whole-stay uses a 30-day cap.', '观察窗限定每次住院取多长（ICU 入科后的小时数）。全程使用 30 天上限。')}</div>
        </div>
        <div class="gdx-field">
          <div class="gdx-label">${t('Comparison', '比较方式')}</div>
          ${seg(COMPARATOR_PRESETS, design.comparator, 'gx-comparator', design.comparatorCustom, 'gx-comparator-custom', t('e.g. vasopressor early vs late', '例如 早期 vs 晚期升压药'))}
          <div class="gdx-hint">${t('Optional. A single descriptive cohort is fine; a comparison sharpens the analysis question.', '可选。单个描述性队列也可以；设定比较会让分析问题更清晰。')}</div>
        </div>
      </div>`;
  }
  function bodyModules(ctx) {
    const t = ctx.t, ex = ctx.ex;
    const concepts = ctx.selectedConcepts;
    const expanded = !!ex.modulesExpanded;
    const grid = (rows) => `<div class="gdx-modgrid">${rows.map(([key, en, zh]) => {
      const on = ex.modules.includes(key);
      return `<button type="button" class="gdx-module ${on ? 'on' : ''}" data-gx-module="${ctx.attr(key)}">
        <span class="mk">${on ? ctx.icon('check', 10, 3) : ''}</span><strong>${t(en, zh)}</strong><span>${ctx.moduleConceptCount(key)}</span>
      </button>`;
    }).join('')}</div>`;
    /* [id, en, zh, is-core] — the concept count column was removed; the
       catalog is the only source for it. */
    const coreRows = ctx.modules.filter(m => m[3]);
    const extraRows = ctx.modules.filter(m => !m[3]);
    const extraOn = extraRows.filter(m => ex.modules.includes(m[0])).length;
    return `
      <div class="gdx-body">
        <div class="gdx-steplead">${t('Which features do you need? I default to the Core 6 that most ICU studies use, then let you add more.', '你需要哪些特征？我默认给出多数 ICU 研究都会用的核心 6 个，再让你按需增加。')}</div>
        <div class="gdx-row">
          <div><div class="gdx-label">${t('Core modules', '核心模块')}</div><small>${ex.modules.length} ${t('modules', '模块')} · ${concepts} ${t('concepts', '概念')}</small></div>
          <div class="gdx-tools">
            <button type="button" class="btn sm" data-gx-module-set="core">${ctx.icon('refresh', 12)} ${t('Reset to Core 6', '重置为核心 6')}</button>
          </div>
        </div>
        ${grid(coreRows)}
        <button type="button" class="gdx-more" data-gx-modules-expand>${ctx.icon(expanded ? 'chevdown' : 'chevron', 13)} ${expanded ? t('Hide advanced modules', '收起进阶模块') : t(`Add more modules (${extraRows.length})`, `添加更多模块（${extraRows.length}）`)}${extraOn && !expanded ? ` · ${extraOn} ${t('on', '已选')}` : ''}</button>
        ${expanded ? `<div class="gdx-tools end"><button type="button" class="btn sm" data-gx-module-set="all">${ctx.icon('check', 12)} ${t('Select all', '全选')}</button><button type="button" class="btn sm" data-gx-module-set="none">${ctx.icon('close', 12)} ${t('Clear', '清空')}</button></div>${grid(extraRows)}` : ''}
      </div>`;
  }
  function bodyExport(ctx) {
    const t = ctx.t, ex = ctx.ex;
    const capOn = ex.maxPatients === 500;
    return `
      <div class="gdx-body">
        <div class="gdx-steplead">${t('How should the prepared data be written, and where should it land?', '准备好的数据应以什么格式写出、保存到哪里？')}</div>
        <div class="gdx-field">
          <div class="gdx-label">${t('Export format', '导出格式')}</div>
          <div class="gdx-seg" role="group" aria-label="Export format">
            ${['parquet', 'csv', 'excel'].map(fmt => `<button type="button" class="${ex.format === fmt ? 'on' : ''}" data-gx-format="${fmt}">${fmt === 'parquet' ? 'Parquet' : fmt.toUpperCase()}</button>`).join('')}
          </div>
          <div class="gdx-hint">${t('Parquet is the default. Each run creates a timestamped folder with README.md and _manifest.json.', '默认 Parquet。每次运行创建带时间戳的文件夹，并写入 README.md 和 _manifest.json。')}</div>
        </div>
        <div class="gdx-field">
          <div class="gdx-label">${t('Export destination', '导出位置')}</div>
          <input data-gx-exportdir value="${ctx.attr(ex.exportDir || '')}" placeholder="${ctx.attr(t('Local folder for the export (leave blank to use the default output folder)', '导出的本地文件夹（留空则使用默认输出目录）'))}" />
          <div class="gdx-hint">${ex.exportDir ? t('A timestamped run folder is created inside this folder.', '将在该文件夹内创建带时间戳的运行子文件夹。') : t('Blank = the app writes to its default local output folder. Nothing leaves your machine either way.', '留空 = 写入应用默认本地输出目录。无论哪种都不会离开你的电脑。')}</div>
        </div>
        <div class="gdx-field">
          <div class="gdx-label">${t('Cohort size', '队列规模')}</div>
          <div class="gdx-seg" role="group" aria-label="Cohort size">
            <button type="button" class="${capOn ? 'on' : ''}" data-gx-max="500">${t('500 safety cap', '500 安全上限')}</button>
            <button type="button" class="${ex.maxPatients === null ? 'on' : ''}" data-gx-max="all">${t('All stays', '全量 stays')}</button>
          </div>
          <div class="gdx-hint">${capOn
            ? t('The cap keeps this first run fast — it extracts the first 500 stays only. Switch to All stays for the full cohort; on a large database that can take much longer.', '该上限让首次运行更快 —— 只抽取前 500 例住院。需要完整队列时切换到“全量”；在大型数据库上会明显更慢。')
            : t('All stays extracts the full cohort. On a large database this can take much longer and write more data.', '“全量”会抽取完整队列。在大型数据库上会明显更慢、写出更多数据。')}</div>
        </div>
      </div>`;
  }
  function bodyReview(ctx) {
    const t = ctx.t, ex = ctx.ex, design = ctx.design;
    const scan = ex.scan || {};
    const cohortRow = ctx.cohortPresets.find(c => c[0] === ex.cohort);
    const outcome = resolveOutcome(design, t);
    const comparator = resolveComparator(design, t);
    const row = (k, v) => `<div class="gdx-sumrow"><span>${ctx.esc(k)}</span><strong>${v ? ctx.esc(v) : `<em class="gdx-muted">${t('not set', '未设置')}</em>`}</strong></div>`;
    return `
      <div class="gdx-body">
        <div class="gdx-steplead">${t('Here is the study I will prepare. Edit any step above, then run the local extraction.', '这是我将要准备的研究。可点上方任意步骤修改，然后运行本地抽取。')}</div>
        <div class="gdx-summary">
          ${row(t('Source', '数据源'), scan.ok ? `${scan.db || 'local'} · ${scan.source || ''}` : '')}
          ${row(t('Cohort', '队列'), cohortRow ? t(cohortRow[1], cohortRow[2]) : ex.cohort)}
          ${row(t('Outcome', '结局'), outcome)}
          ${row(t('Window', '观察窗'), windowLabel(design, t))}
          ${row(t('Comparison', '比较'), comparator)}
          ${row(t('Modules', '模块'), `${ex.modules.length} · ${ctx.selectedConcepts} ${t('concepts', '概念')}`)}
          ${row(t('Format', '格式'), (ex.format || 'parquet').toUpperCase())}
          ${row(t('Destination', '导出位置'), ex.exportDir ? ctx.compactPath(ex.exportDir) : t('default output folder', '默认输出目录'))}
          ${row(t('Cohort size', '队列规模'), ex.maxPatients === null ? t('All stays', '全量') : t('500 safety cap', '500 安全上限'))}
        </div>
      </div>`;
  }

  function bodyFor(ctx, step) {
    switch (step) {
      case 'source': return bodySource(ctx);
      case 'cohort': return bodyCohort(ctx);
      case 'design': return bodyDesign(ctx);
      case 'modules': return bodyModules(ctx);
      case 'export': return bodyExport(ctx);
      case 'review': return bodyReview(ctx);
      default: return bodySource(ctx);
    }
  }

  function footer(ctx, step) {
    const t = ctx.t, ex = ctx.ex;
    const idx = STEP_ORDER.indexOf(step);
    const back = idx > 0
      ? `<button type="button" class="btn" data-gx-step-back>${ctx.icon('back', 13)} ${t('Back', '上一步')}</button>`
      : '';
    if (step === 'review') {
      const canRun = ctx.ready && ex.modules.length && !ex.running;
      return `<div class="gdx-actions">
        ${back}
        <button type="button" class="btn primary" data-gx-run ${canRun ? '' : 'disabled'}>${ctx.icon('play', 13)} ${t('Run extraction here', '在这里开始抽取')}</button>
        ${ex.scan && ex.scan.source === 'module' ? `<button type="button" class="btn primary" data-gx-use-export>${ctx.icon('check', 13)} ${t('Register this export', '注册这个导出')}</button>` : ''}
        <button type="button" class="btn" data-open="extraction">${t('Advanced classic settings', '打开高级经典设置')}</button>
      </div>
      ${!ctx.ready ? `<div class="gdx-hint">${t('Analyze an extraction-ready local folder on the Source step before running.', '运行前请在“数据源”步骤识别一个可直接抽取的本地文件夹。')}</div>` : ''}`;
    }
    // source step gates Continue on a ready scan (or a module export -> register path)
    const isModuleExport = ex.scan && ex.scan.source === 'module';
    const sourceBlocked = step === 'source' && !ctx.ready && !isModuleExport;
    const nextId = STEP_ORDER[idx + 1];
    const nextLabel = nextId === 'review' ? t('Review & run', '确认并运行') : t('Continue', '继续');
    return `<div class="gdx-actions">
      ${back}
      ${isModuleExport && step === 'source'
        ? `<button type="button" class="btn primary" data-gx-use-export>${ctx.icon('check', 13)} ${t('Register this export', '注册这个导出')}</button>`
        : `<button type="button" class="btn primary" data-gx-step-next ${sourceBlocked ? 'disabled' : ''}>${nextLabel} ${ctx.icon('chevron', 13)}</button>`}
      <button type="button" class="btn" data-open="extraction">${t('Advanced classic settings', '打开高级经典设置')}</button>
    </div>
    ${sourceBlocked ? `<div class="gdx-hint">${t('Paste a local ICU folder and analyze it to continue.', '粘贴本机 ICU 文件夹并识别目录后继续。')}</div>` : ''}`;
  }

  function renderDone(ctx) {
    const t = ctx.t, ex = ctx.ex, r = ex.result || {};
    return `
      <div class="gdx-body">
        <div class="gdx-status ok"><span>${ctx.icon('check', 12)}</span><div><strong>${ctx.esc(ctx.statusText)}</strong>${ex.jobId ? `<small>job ${ctx.esc(ex.jobId)}</small>` : ''}</div></div>
        <div class="gdx-result">
          <span>${t('Output folder', '输出文件夹')}</span>
          <code>${ctx.esc(ctx.compactPath(r.out_dir || r.path || ''))}</code>
          <span>${t('Rows', '行数')}</span><strong>${ctx.fmtInt(r.total_rows, 'n/a')}</strong>
          <span>${t('Files', '文件')}</span><strong>${ctx.fmtInt(r.files_written || r.files, 'n/a')}</strong>
        </div>
      </div>
      <div class="gdx-actions">
        <button type="button" class="btn" data-open="patient">${t('Review export', '审阅导出结果')}</button>
        <button type="button" class="btn primary" data-guided-goal="run_agent">${t('Continue to Agent preflight', '继续 Agent 预检')}</button>
      </div>`;
  }

  function render(ctx) {
    const t = ctx.t, ex = ctx.ex;
    const step = ex.result ? null : (ex.step || 'source');
    const head = `
      <div class="gdx-head">
        <span class="gdx-ico">${ctx.icon('extract', 15)}</span>
        <div>
          <strong>${t('Prepare data inside Copilot', '在 Copilot 内准备/抽取数据')}</strong>
          <span>${t('Same backend as Classic Data Extraction — walked one decision at a time.', '复用经典数据抽取同一个后端 —— 一次只做一个决定。')}</span>
        </div>
      </div>`;
    if (ex.result) {
      return `<div class="gd-x-card" data-guided-extraction-card>${head}${renderDone(ctx)}</div>`;
    }
    const statusBlock = (ex.running || ex.error || ex.scanError)
      ? `<div class="gdx-status ${ex.error || ex.scanError ? 'bad' : ''}"><span>${ctx.icon(ex.error || ex.scanError ? 'x' : 'shield', 12)}</span><div><strong>${ctx.esc(ctx.statusText)}</strong>${ex.jobId ? `<small>job ${ctx.esc(ex.jobId)}</small>` : ''}</div></div>`
      : '';
    const bar = ex.running ? `<div class="gdx-bar"><span style="width:${ctx.progressPct}%"></span></div>` : '';
    return `
      <div class="gd-x-card" data-guided-extraction-card>
        ${head}
        ${stepIndicator(ctx, step)}
        ${recap(ctx)}
        ${bodyFor(ctx, step)}
        ${statusBlock}
        ${bar}
        ${footer(ctx, step)}
      </div>`;
  }

  window.EU_GUIDED_EXTRACT = {
    render,
    STEP_ORDER,
    OUTCOME_PRESETS,
    WINDOW_PRESETS,
    COMPARATOR_PRESETS,
    resolveOutcome,
    resolveComparator,
    windowHours,
    windowLabel,
  };
})();
