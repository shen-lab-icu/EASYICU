/* Screen: Idea Mining — first-class discovery workflow.
   Local-first Stage67: user-supplied metadata/excerpt -> idea ledger ->
   dictionary/export feasibility -> pre-experiment -> Agent handoff plan. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  let srcType = 'manual';
  let mining = false;
  let resolving = false;
  let priorArting = false;
  let handoffing = false;
  let projectCreating = false;
  let loadingRun = null;
  let err = null;
  let result = null;
  let sourceResolved = null;
  let priorArt = null;
  let projectSeed = null;
  let selectedRunId = null;
  let selectedRecordKey = null;
  let history = null;
  let planEdits = '';
  let draft = {};
  let activeStep = 'source';

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }
  function fmt(v) {
    if (v == null || v === '') return '—';
    if (typeof v === 'number') return Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });
    return esc(v);
  }
  function pct(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(100, n));
  }
  function pctLabel(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return '—';
    return `${Number(n).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
  }
  function coverageTone(v) {
    const n = pct(v);
    if (n < 50) return 'risk';
    if (n < 90) return 'warn';
    return 'ok';
  }
  function activeSourceLine() {
    const src = window.EU_SOURCES && window.EU_SOURCES.activeSource ? window.EU_SOURCES.activeSource() : null;
    if (!src) return 'No active export selected';
    const s = src.summary || {};
    return `${src.label || src.database || 'Local export'} · ${fmt(s.stays)} entities · ${fmt(s.modules)} modules`;
  }
  function activeSourceShortLine() {
    const src = window.EU_SOURCES && window.EU_SOURCES.activeSource ? window.EU_SOURCES.activeSource() : null;
    if (!src) return t('No export', '无导出');
    const s = src.summary || {};
    const label = src.label || src.database || 'Local export';
    return `${label} · ${fmt(s.stays || 0)} · ${fmt(s.modules || 0)}`;
  }
  function activeStepLabel() {
    if (activeStep === 'ledger') return t('Ledger', '台账');
    if (activeStep === 'evidence') return t('Feasibility', '可行性');
    if (activeStep === 'handoff') return t('Handoff', '交接');
    return t('Source', '来源');
  }
  function repaint() {
    if (window.__euRender) window.__euRender();
  }
  function inputVal(root, sel) {
    const el = root.querySelector(sel);
    return el ? el.value.trim() : '';
  }
  function inputValOr(root, sel, key) {
    const el = root.querySelector(sel);
    return el ? el.value.trim() : fieldValue(key);
  }
  function fieldValue(key) {
    return draft && draft[key] != null ? String(draft[key]) : '';
  }
  function collectPayload(root) {
    const opt = root.querySelector('#ideaNetworkOptIn');
    draft = {
      source_type: srcType,
      topic: inputValOr(root, '#ideaTopic', 'topic'),
      excerpt: inputValOr(root, '#ideaExcerpt', 'excerpt'),
      title: inputValOr(root, '#ideaTitle', 'title'),
      journal: inputValOr(root, '#ideaJournal', 'journal'),
      year: inputValOr(root, '#ideaYear', 'year'),
      doi: inputValOr(root, '#ideaDoi', 'doi'),
      url: inputValOr(root, '#ideaUrl', 'url'),
      allow_network: opt ? !!opt.checked : !!(draft && draft.allow_network),
    };
    return draft;
  }
  async function postLocalJSON(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      let detail = '';
      try {
        const payload = await res.json();
        const d = payload && payload.detail;
        detail = typeof d === 'string' ? d : (d && (d.error || JSON.stringify(d)));
      } catch (e) {}
      throw new Error(path + ' -> HTTP ' + res.status + (detail ? ' · ' + detail : ''));
    }
    return res.json();
  }
  function loadIdeaRunApi(body) {
    if (window.EU_API && window.EU_API.loadIdeaRun) return window.EU_API.loadIdeaRun(body);
    return postLocalJSON('/api/ideas/run', body || {});
  }
  function sourceTabs() {
    const rows = [
      ['manual', 'Manual idea', '手动输入', 'Paste a topic, article metadata, or excerpt.'],
      ['url', 'Article metadata', '文章元数据', 'Paste title/abstract/excerpt; the URL is citation metadata until live fetch is enabled.'],
      ['pdf', 'PDF excerpt', 'PDF 摘录', 'Paste bounded text now; full PDF parsing comes later.'],
      ['frontier', 'Frontier prompt', '前沿主题', 'Describe journals/topics now; live journal search needs opt-in.'],
    ];
    return `
      <div class="modeswitch ideas-source-switch" data-ideas-tabs>
        ${rows.map(r => `<button class="${srcType === r[0] ? 'on' : ''}" data-idea-src="${r[0]}" title="${esc(r[3])}">${t(r[1], r[2])}</button>`).join('')}
      </div>`;
  }
  function modeTitle() {
    if (srcType === 'url') return t('Article metadata seed', '文章元数据种子');
    if (srcType === 'pdf') return t('PDF excerpt seed', 'PDF 摘录种子');
    if (srcType === 'frontier') return t('Frontier topic seed', '前沿主题种子');
    return t('Manual idea seed', '手动想法种子');
  }
  function modePrimaryLabel() {
    if (srcType === 'frontier') return t('Frontier topic / journal scope', '前沿主题 / 期刊范围');
    if (srcType === 'url') return t('Question this article suggests', '这篇文章启发的问题');
    if (srcType === 'pdf') return t('Question from this excerpt', '这段摘录启发的问题');
    return t('Idea / research question', '想法 / 研究问题');
  }
  function modeExcerptLabel() {
    if (srcType === 'frontier') return t('Reasoning notes or review theme', '推理备注或 review 主题');
    if (srcType === 'pdf') return t('Bounded PDF excerpt', '有界 PDF 摘录');
    if (srcType === 'url') return t('Abstract / quoted trigger sentence', '摘要 / 触发原文句子');
    return t('Source quote or rationale sentence', '来源引用或触发句');
  }
  function modePlaceholder() {
    if (srcType === 'frontier') return 'e.g. ICU long-term outcomes after septic shock, editorials/reviews in Intensive Care Medicine';
    if (srcType === 'url') return 'e.g. What ICU-database study could test this trial or review insight?';
    if (srcType === 'pdf') return 'e.g. The excerpt suggests a measurable ICU exposure, outcome, or subgroup.';
    return 'e.g. Vasopressor-first resuscitation and mortality among adult septic shock ICU patients';
  }
  function sourceModeHint() {
    if (srcType === 'url') {
      return t('Article URL mode is metadata-only for now: paste the title, abstract/excerpt, or rationale sentence. The app will not fetch the URL until the opt-in adapter is connected.', '文章链接模式目前仅作元数据：请粘贴标题、摘要/摘录或触发句。接入 opt-in adapter 前，应用不会自动抓取链接。');
    }
    if (srcType === 'pdf') {
      return t('PDF mode currently accepts pasted bounded excerpts. Full PDF parsing and upload handling are still blocked behind a separate parser/opt-in stage.', 'PDF 模式目前只接受粘贴的有界摘录。完整 PDF 解析和上传处理仍需单独 parser/opt-in 阶段。');
    }
    if (srcType === 'frontier') {
      return t('Frontier mode currently creates a local plan from your prompt. Live journal/review search and prior-art checking require explicit network/provider opt-in.', '前沿主题模式目前只基于你的描述生成本地计划。真实期刊/review 检索和 prior-art 检查需要明确网络/provider opt-in。');
    }
    return t('Manual mode creates a local, evidence-bound idea ledger from the text you provide.', '手动模式会根据你提供的文本生成本地、证据绑定的 idea 台账。');
  }
  function validatePayload(payload) {
    const hasTopic = !!payload.topic;
    const hasExcerpt = !!payload.excerpt;
    const hasTitle = !!payload.title;
    if (srcType === 'url' && !payload.url) return t('Paste the article URL, then add a title, abstract, excerpt, or rationale sentence.', '请先粘贴文章链接，并补充标题、摘要、摘录或触发句。');
    if (srcType === 'url' && !(hasTopic || hasExcerpt || hasTitle)) return t('This version does not fetch the URL automatically. Add a title, abstract, excerpt, or rationale sentence before mining.', '当前版本不会自动抓取链接。请先补充标题、摘要、摘录或触发句再挖掘。');
    if (srcType === 'pdf' && !hasExcerpt) return t('Paste a bounded PDF excerpt before mining. Full PDF parsing is not enabled in this local pass.', '请先粘贴有界 PDF 摘录。本地第一版尚未启用完整 PDF 解析。');
    if (srcType === 'frontier' && !hasTopic) return t('Describe the frontier topic or journal scope before mining. Live search is a separate opt-in stage.', '请先描述前沿主题或期刊范围。真实检索属于单独 opt-in 阶段。');
    if (!(hasTopic || hasExcerpt || hasTitle || payload.url)) return t('Enter a topic, source title, URL, or excerpt before mining.', '请先输入主题、来源标题、链接或摘录。');
    return '';
  }
  function draftFromRun(data) {
    const source = ((data && data.source_evidence) || [])[0] || {};
    const idea = ((data && data.idea_ledger) || [])[0] || {};
    return {
      source_type: source.source_type || 'manual',
      topic: idea.idea_title || data.candidate_topic || '',
      excerpt: source.evidence_quote || idea.source_quote || '',
      title: source.title || idea.source_title || '',
      journal: source.journal || idea.source_journal || '',
      year: source.year || idea.source_year || '',
      doi: source.doi || '',
      url: source.url || '',
      allow_network: false,
    };
  }
  function runRecordKey(row, fallback) {
    return String((row && (row.history_key || row.record_key)) || [
      row && row.run_id,
      row && row.created_at,
      fallback == null ? '' : fallback,
    ].filter(v => v != null && v !== '').join('::') || '');
  }
  function applyRunPayload(data, recordKey) {
    result = data;
    selectedRunId = data && data.run_id ? data.run_id : null;
    selectedRecordKey = recordKey || runRecordKey(data || {}, 'current') || selectedRunId;
    priorArt = data && data.prior_art_check ? data.prior_art_check : null;
    window.EU_IDEA_HANDOFF = data && data.handoff ? data.handoff : null;
    projectSeed = data && data.agent_project ? data.agent_project : null;
    draft = draftFromRun(data || {});
    srcType = draft.source_type || 'manual';
    planEdits = ((window.EU_IDEA_HANDOFF || {}).handoff_plan || {}).human_plan_notes || '';
    sourceResolved = null;
    err = null;
    activeStep = window.EU_IDEA_HANDOFF || projectSeed ? 'handoff' : data ? 'ledger' : 'source';
  }
  function historyRowFromRun(data) {
    const source = ((data && data.source_evidence) || [])[0] || {};
    const idea = ((data && data.idea_ledger) || [])[0] || {};
    return {
      run_id: data && data.run_id,
      created_at: data && data.created_at,
      history_key: runRecordKey(data || {}, 'current'),
      title: idea.idea_title || source.title || t('Idea run', 'Idea 记录'),
      source_title: source.title,
      source_year: source.year,
      journal: source.journal,
      go_no_go: idea.go_no_go,
      feasibility_tier: (idea.feasibility || {}).tier,
    };
  }
  function upsertHistoryRun(data) {
    const row = historyRowFromRun(data);
    if (!row.run_id) return;
    const current = history && Array.isArray(history.runs) ? history.runs : [];
    history = Object.assign({}, history || { ok: true }, {
      runs: [row].concat(current.filter(r => r && r.run_id !== row.run_id)).slice(0, 100),
    });
  }
  function stepState(step) {
    const hasResult = !!result;
    const hasHandoff = !!window.EU_IDEA_HANDOFF;
    const hasProject = !!projectSeed;
    if (step === 'source') return 'ready';
    if (step === 'ledger') return hasResult ? 'ready' : 'locked';
    if (step === 'evidence') return hasResult ? 'ready' : 'locked';
    if (step === 'handoff') return hasHandoff || hasProject || hasResult ? 'ready' : 'locked';
    return 'locked';
  }
  function stepSummary() {
    const idea = result && (result.idea_ledger || [])[0];
    const pre = result && result.pre_experiment;
    const handoff = window.EU_IDEA_HANDOFF;
    const project = projectSeed;
    const source = result && (result.source_evidence || [])[0];
    return `
      <div class="ideas-summary-strip">
        <div class="ideas-summary-item ${source ? 'ready' : 'idle'}"><span>${icon(source ? 'check' : 'file', 13)}</span><div><b>${t('Source', '来源')}</b><small>${source ? esc(source.title || source.journal || 'bound') : t('waiting for input', '等待输入')}</small></div></div>
        <div class="ideas-summary-item ${idea ? 'ready' : 'idle'}"><span>${icon(idea ? 'target' : 'clock', 13)}</span><div><b>${t('Idea ledger', 'Idea 台账')}</b><small>${idea ? esc(idea.go_no_go || 'draft') : t('not mined', '尚未挖掘')}</small></div></div>
        <div class="ideas-summary-item ${pre ? 'ready' : 'idle'}"><span>${icon(pre ? 'beaker' : 'shield', 13)}</span><div><b>${t('Pre-experiment', '预实验')}</b><small>${pre ? esc(pre.status || 'checked') : t('after mining', '挖掘后生成')}</small></div></div>
        <div class="ideas-summary-item ${handoff || project ? 'ready' : 'idle'}"><span>${icon(project ? 'agent' : handoff ? 'check' : 'arrow', 13)}</span><div><b>${t('Plan handoff', '计划交接')}</b><small>${project ? t('project seed ready', '项目种子已生成') : handoff ? t('frozen for Agent', '已冻结给 Agent') : t('locked until review', '评审后解锁')}</small></div></div>
      </div>`;
  }
  function stepNav() {
    const steps = [
      ['source', t('1. Source', '1. 来源'), t('Paste the paper clue or research hunch.', '粘贴文章线索或研究直觉。'), 'file'],
      ['ledger', t('2. Idea ledger', '2. Idea 台账'), t('Review evidence-bound candidate ideas.', '查看证据绑定候选想法。'), 'target'],
      ['evidence', t('3. Feasibility', '3. 可行性'), t('Check dictionary, export coverage, and prior art.', '检查字典、导出覆盖和已有研究。'), 'beaker'],
      ['handoff', t('4. Handoff', '4. 交接'), t('Freeze a seed for Research Projects.', '冻结给研究项目的种子。'), 'agent'],
    ];
    return `<div class="ideas-step-nav">${steps.map(row => {
      const state = stepState(row[0]);
      const cls = [activeStep === row[0] ? 'active' : '', state].filter(Boolean).join(' ');
      return `<button class="ideas-step-tab ${cls}" data-idea-step="${row[0]}" ${state === 'locked' ? 'aria-disabled="true"' : ''}>
        <span class="ideas-step-icon">${icon(state === 'ready' && row[0] !== activeStep ? 'check' : row[3], 13)}</span>
        <span><b>${row[1]}</b><small>${row[2]}</small></span>
      </button>`;
    }).join('')}</div>`;
  }
  function lockedStep(step) {
    return `
      <div class="card pad ideas-core-card">
        <div class="empty-mini" style="min-height:240px;">
          <div>${icon('lock', 22)}</div>
          <h3>${t('Create the idea ledger first', '请先生成 idea 台账')}</h3>
          <p>${t('This step is locked until the local mining pass has bound source evidence and checked the active export.', '本步骤需要先完成本地挖掘，绑定来源证据并检查当前导出。')}</p>
          <button class="btn primary" data-idea-step="source">${icon('arrow', 13)} ${t('Go to Source', '回到来源')}</button>
        </div>
      </div>`;
  }
  function sourceForm() {
    return `
      <div class="card pad ideas-compose ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('spark', 14)}</span>
          <div><h2>${t('Start with one clue', '先输入一个线索')}</h2><p>${t('Pick the source type, then provide the question and the sentence that triggered it. Everything else is optional.', '选择来源类型，然后填写研究问题和触发它的原文句子。其余信息都可以稍后补。')}</p></div>
        </div>
        <div class="ideas-mode-row">${sourceTabs()}<div class="pill ok"><span class="dot"></span>${t('Local discovery only', '仅本地发现')}</div></div>
        <div class="note info mt-12 ideas-mode-note compact">
          <div class="ico">${icon('shield', 14)}</div>
          <div class="body"><div class="t">${esc(modeTitle())}</div><div class="d">${esc(sourceModeHint())}</div></div>
        </div>
        <div class="ideas-primary-grid mt-14">
          <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="4" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
          <label class="field ideas-field"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="4" placeholder="${t('Paste only the sentence(s) or bounded excerpt that triggered the idea. We store this quote and hashes, not the full paper.', '只粘贴触发想法的句子或有界摘录。系统保存引用和哈希，不保存全文。')}">${esc(fieldValue('excerpt'))}</textarea></label>
        </div>
        <details class="ideas-advanced mt-14">
          <summary>${icon('list', 13)} ${t('Source metadata', '来源元数据')} <span>${t('optional, but useful for citation and prior-art checks', '可选；用于引用和已有研究检查')}</span></summary>
          <div class="ideas-meta-grid mt-10">
            <label class="field ideas-field"><span>Title</span><input id="ideaTitle" placeholder="Article or review title" value="${esc(fieldValue('title'))}" /></label>
            <label class="field ideas-field"><span>Journal</span><input id="ideaJournal" placeholder="e.g. Intensive Care Medicine" value="${esc(fieldValue('journal'))}" /></label>
            <label class="field ideas-field ideas-year"><span>Year</span><input id="ideaYear" placeholder="2026" value="${esc(fieldValue('year'))}" /></label>
            <label class="field ideas-field"><span>DOI / PMID</span><input id="ideaDoi" placeholder="10.xxxx or PMID" value="${esc(fieldValue('doi'))}" /></label>
            <label class="field ideas-field ideas-url-field"><span>URL</span><input id="ideaUrl" placeholder="https://..." value="${esc(fieldValue('url'))}" /></label>
          </div>
        </details>
        <details class="ideas-advanced mt-10">
          <summary>${icon('shield', 13)} ${t('Network and provider opt-in', '网络与模型 opt-in')} <span>${t('off by default', '默认关闭')}</span></summary>
          <label class="rtodo-row mt-10 ideas-network-row">
            <input type="checkbox" id="ideaNetworkOptIn" ${fieldValue('allow_network') === 'true' || fieldValue('allow_network') === true ? 'checked' : ''} />
            <span class="rtodo-t">${t('Allow one bounded network metadata/prior-art request for this source', '允许本来源进行一次有界网络元数据 / prior-art 请求')}</span>
            <span class="rtodo-ref mono">opt-in</span>
          </label>
          <div class="muted mt-8">${t('This pass does not fetch the URL, parse a PDF, or call an external LLM unless you explicitly opt in and a provider is configured.', '除非你明确 opt-in 且 provider 已配置，否则这一版不会抓取链接、解析 PDF 或调用外部 LLM。')}</div>
        </details>
        ${!result ? `<div class="ideas-zero-line mt-12">${icon('clock', 13)} <span>${t('No idea ledger yet', '还没有 idea 台账')}</span><b>${t('Next: create one local, auditable record.', '下一步：生成一条本地、可审计记录。')}</b></div>` : ''}
        <div class="row gap-8 mt-16 ideas-actions">
          <button class="btn" data-idea-resolve ${resolving ? 'aria-disabled="true"' : ''}>${resolving ? '<span class="spin"></span>' : icon('search', 14)} ${t('Resolve source', '解析来源')}</button>
          <button class="btn primary" data-idea-mine ${mining ? 'aria-disabled="true"' : ''}>${mining ? '<span class="spin"></span>' : icon('play', 14)} ${t('Create idea ledger', '生成 idea 台账')}</button>
        </div>
        ${sourceResolved ? `<div class="note ${sourceResolved.source_adapter && sourceResolved.source_adapter.status && sourceResolved.source_adapter.status.includes('blocked') ? 'warn' : 'ok'} mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${esc((sourceResolved.source_adapter || {}).status || 'source resolved')}</div><div class="d">${esc((sourceResolved.source_adapter || {}).reason || 'Bounded source metadata is ready for the idea ledger.')}</div></div></div>` : ''}
      </div>`;
  }
  function historyBlock() {
    const rows = (history && history.runs) || [];
    return `
      <div class="card pad">
        <div class="section-head">
          <span class="sec-ico">${icon('history', 14)}</span>
          <div><h2>${t('Local idea runs', '本地 idea 记录')}</h2><p>${t('Metadata-only history. Source bodies and patient rows are not stored here.', '仅元数据历史。不保存全文或患者行。')}</p></div>
        </div>
        ${rows.length ? `<div class="ledger compact">${rows.slice(0, 6).map(r => `
          <div class="ledger-row">
            <span class="ledger-ico">${icon(r.go_no_go === 'recommend' ? 'check' : 'shield', 14)}</span>
            <div><div style="font-weight:650;font-size:12.5px;">${esc(r.title || 'Idea run')}</div><div class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc(r.feasibility_tier || '—')} · ${esc(r.journal || 'source')} · ${esc(r.created_at || '')}</div></div>
          </div>`).join('')}</div>` : `<div class="empty-mini">${t('No local idea run yet.', '还没有本地 idea 运行。')}</div>`}
      </div>`;
  }
  function sourceEvidence() {
    const rows = (result && result.source_evidence) || [];
    if (!rows.length) return '';
    return `
      <div class="card pad ideas-core-card">
        <div class="eyebrow">${t('Source evidence', '来源证据')}</div>
        <div class="ideas-source-stack mt-8">
          ${rows.map(r => `
            <article class="ideas-source-card">
              <div class="ideas-source-head">
                <div class="ideas-source-title">
                  <b>${esc(r.title || t('Untitled source', '未命名来源'))}</b>
                  <span class="mono">${esc(r.doi || r.pmid || r.url || r.citation_key || '')}</span>
                </div>
                <div class="ideas-source-meta">
                  <span>${fmt(r.year)}</span>
                  <span>${fmt(r.journal)}</span>
                  <span class="pill ok">${icon('shield', 11)} ${t('hash only', '仅哈希')}</span>
                </div>
              </div>
              <div class="ideas-evidence-quote">
                <span>${t('Evidence sentence', '证据句')}</span>
                <p>${esc(r.evidence_quote || t('No quoted sentence was provided for this source.', '该来源没有提供原文证据句。'))}</p>
              </div>
            </article>`).join('')}
        </div>
      </div>`;
  }
  function ideaLedger() {
    const rows = (result && result.idea_ledger) || [];
    if (!rows.length) return '';
    return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('target', 14)}</span>
          <div><h2>${t('Idea ledger', 'Idea 台账')}</h2><p>${t('Each row must carry source evidence, dictionary feasibility, prior-art status, and next action.', '每一行都必须带来源证据、字典可行性、已有研究状态和下一步。')}</p></div>
        </div>
        <div class="idea-ledger-stack mt-10">
          ${rows.map(r => {
            const concepts = (r.mapped_concepts || []).map(c => `<span class="chip">${esc(c.concept_id)} · ${esc(c.tier)}</span>`).join(' ');
            const feas = r.feasibility || {};
            const prior = r.prior_art || {};
            return `<article class="idea-ledger-card">
              <div class="idea-ledger-main">
                <div class="idea-ledger-title">${esc(r.idea_title || t('Candidate idea', '候选想法'))}</div>
                <p>${esc(r.rationale || t('No rationale recorded yet.', '尚未记录 rationale。'))}</p>
              </div>
              <div class="idea-ledger-grid">
                <section>
                  <h4>${t('Mapped concepts', '映射特征')}</h4>
                  <div class="ideas-chip-list">${concepts || `<span class="muted">${t('No mapped concept yet', '尚无映射特征')}</span>`}</div>
                </section>
                <section>
                  <h4>${t('Feasibility', '可行性')}</h4>
                  <span class="pill ${feas.tier === 'executable' ? 'ok' : 'warn'}">${esc(feas.label || feas.tier || 'unknown')}</span>
                  <p>${esc(feas.reason || t('No feasibility rationale recorded.', '尚未记录可行性理由。'))}</p>
                </section>
                <section>
                  <h4>${t('Prior art', '已有研究')}</h4>
                  <span class="pill warn">${esc(prior.status || 'not checked')}</span>
                  <p>${esc(prior.reason || t('Prior-art search has not been run.', '尚未运行已有研究检索。'))}</p>
                </section>
                <section>
                  <h4>${t('Decision', '决策')}</h4>
                  <b>${esc(r.go_no_go || 'pending')}</b>
                  <p>${esc(r.next_action || t('Choose the next action after feasibility review.', '完成可行性审阅后选择下一步。'))}</p>
                </section>
              </div>
            </article>`;
          }).join('')}
        </div>
        <div class="ideas-inline-actions mt-14">
          <button class="btn primary" data-idea-step="evidence">${icon('beaker', 13)} ${t('Review feasibility', '查看可行性')}</button>
          <button class="btn" data-idea-step="handoff">${icon('agent', 13)} ${t('Plan handoff', '计划交接')}</button>
        </div>
      </div>`;
  }
  function priorArtPanel() {
    if (!result) return '';
    const prior = priorArt && priorArt.prior_art ? priorArt.prior_art : null;
    const rows = prior && Array.isArray(prior.results) ? prior.results : [];
    const queries = prior && Array.isArray(prior.queries_to_run) ? prior.queries_to_run : [];
    const status = prior ? (prior.status || (prior.search_performed ? 'checked' : 'blocked')) : 'not checked';
    const statusTone = prior && prior.search_performed ? 'ok' : 'warn';
    return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('search', 14)}</span>
          <div><h2>${t('Prior-art check', '已有研究检查')}</h2><p>${t('Optional metadata search, kept behind explicit opt-in.', '可选元数据检索，必须显式 opt-in。')}</p></div>
        </div>
        <div class="ideas-prior-card mt-10">
          <div class="ideas-prior-top">
            <div>
              <span class="pill ${statusTone}">${esc(status)}</span>
              <p>${esc((prior && prior.reason) || t('No prior-art request has been made yet.', '尚未发起已有研究检索。'))}</p>
            </div>
            <button class="btn ${prior && prior.search_performed ? '' : 'primary'}" data-idea-prior-art ${priorArting ? 'aria-disabled="true"' : ''}>${priorArting ? '<span class="spin"></span>' : icon('search', 13)} ${t('Check prior art', '检查已有研究')}</button>
          </div>
          <label class="rtodo-row ideas-network-row">
            <input type="checkbox" id="ideaNetworkOptIn" ${fieldValue('allow_network') === 'true' || fieldValue('allow_network') === true ? 'checked' : ''} />
            <span class="rtodo-t">${t('Allow one bounded PubMed metadata search for this idea', '允许为这个 idea 进行一次有界 PubMed 元数据检索')}</span>
            <span class="rtodo-ref mono">opt-in</span>
          </label>
          ${queries.length ? `<details class="ideas-query-details">
            <summary>${icon('search', 13)} ${t('Suggested search queries', '建议检索式')} <span>${queries.length}</span></summary>
            <div class="ideas-query-list">${queries.map(q => `<code>${esc(q)}</code>`).join('')}</div>
          </details>` : ''}
        </div>
        ${rows.length ? `<div class="ideas-prior-results mt-12">${rows.map(r => `<article><b>${esc(r.title || '')}</b><span>${esc(r.journal || '')} · ${fmt(r.year)} · PMID ${esc(r.pmid || '')}</span></article>`).join('')}</div>` : ''}
      </div>`;
  }
  function preExperiment() {
    const pre = result && result.pre_experiment;
    if (!pre) return '';
    const stats = pre.feature_statistics || [];
    const visibleStats = stats.slice(0, 4);
    const hiddenStats = stats.slice(4);
    const riskCount = stats.filter(s => pct(s.coverage_pct) < 50).length;
    const featureRow = (s) => {
      const n = s.numeric_summary || {};
      const coverage = pct(s.coverage_pct);
      const tone = coverageTone(coverage);
      const summary = n.available ? `median ${fmt(n.median)} · min ${fmt(n.min)} · max ${fmt(n.max)}` : t('categorical, non-numeric, or empty', '分类、非数值或为空');
      return `<div class="ideas-feature-row ${tone}">
        <div class="ideas-feature-name">
          <b>${esc(s.label)}</b>
          <span class="mono">${esc(s.module || '')} · ${esc(s.concept_id || '')}</span>
        </div>
        <div class="ideas-feature-cov">
          <div class="ideas-cov-head"><span>${t('Coverage', '覆盖')}</span><b>${pctLabel(s.coverage_pct)}</b></div>
          <div class="ideas-cov-bar"><i style="width:${coverage}%"></i></div>
        </div>
        <div class="ideas-feature-meta">
          <span>${t('Records', '记录')} ${fmt(s.records)}</span>
          <span>${t('Missing', '缺失')} ${pctLabel(s.missing_pct)}</span>
        </div>
        <div class="ideas-feature-summary">${esc(summary)}</div>
      </div>`;
    };
    return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('beaker', 14)}</span>
          <div><h2>${t('Pre-experiment on active export', '基于当前导出的预实验')}</h2><p>${t('Quick feasibility triage from the active export.', '基于当前导出的快速可行性预审。')} ${esc(activeSourceLine())}</p></div>
        </div>
        <div class="ideas-pre-summary mt-10">
          <div><span>Status</span><b>${esc(pre.status || '—')}</b></div>
          <div><span>Entities</span><b>${fmt(pre.cohort && pre.cohort.entities)}</b></div>
          <div><span>Modules</span><b>${fmt(pre.cohort && pre.cohort.modules)}</b></div>
          <div><span>${t('Low coverage', '低覆盖')}</span><b>${fmt(riskCount)}</b></div>
        </div>
        ${stats.length ? `<div class="ideas-feature-list mt-12">${visibleStats.map(featureRow).join('')}</div>
          ${hiddenStats.length ? `<details class="ideas-compact-details mt-10"><summary>${icon('list', 13)} ${t('Show all feature checks', '查看全部特征检查')} <span>${stats.length}</span></summary><div class="ideas-feature-list mt-10">${stats.map(featureRow).join('')}</div></details>` : ''}`
        : `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${esc(pre.reason || 'No feature statistics available')}</div></div></div>`}
        ${pre.interpretation && pre.interpretation.length ? `<div class="ideas-interpretation mt-10">${pre.interpretation.slice(0, 2).map(x => `<div>${icon('shield', 13)} <span>${esc(x)}</span></div>`).join('')}</div>` : ''}
      </div>`;
  }
  function planBuilder() {
    const plan = result && result.handoff_plan;
    if (!plan) return '';
    return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('agent', 14)}</span>
          <div><h2>${t('Plan handoff', '计划交接')}</h2><p>${t('Confirm or edit the plan before sending it to Agent Projects. This does not unlock a manuscript draft.', '交给 Agent Projects 前先确认或微调计划。这里不会解锁论文草稿。')}</p></div>
        </div>
        <div class="note ok mt-8"><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${esc(plan.research_question || '')}</div><div class="d">${t('Draft analysis plan is locked until human confirmation and evidence gates pass.', '分析计划草稿在人工确认和证据闸通过前保持锁定。')}</div></div></div>
        <div class="ledger compact mt-12">
          ${(plan.analysis_plan || []).map((x, i) => `<div class="ledger-row"><span class="ledger-ico">${String(i + 1).padStart(2, '0')}</span><div>${esc(x)}</div></div>`).join('')}
        </div>
        <label class="field ideas-plan-edits mt-12"><span>${t('Natural-language plan edits', '用自然语言微调计划')}</span><textarea id="ideaPlanEdits" rows="4" placeholder="${t('e.g. use AKI as the endpoint, restrict to first ICU stay, add missingness sensitivity...', '例如:把 AKI 作为结局,限制首次 ICU 入住,增加缺失敏感性分析...')}">${esc(planEdits)}</textarea></label>
        <div class="row gap-8 mt-12">
          <button class="btn primary" data-idea-handoff ${handoffing ? 'aria-disabled="true"' : ''}>${handoffing ? '<span class="spin"></span>' : icon('arrow', 14)} ${t('Freeze handoff for Agent', '冻结并交给 Agent')}</button>
          ${window.EU_IDEA_HANDOFF ? `<button class="btn primary" data-idea-create-project ${projectCreating ? 'aria-disabled="true"' : ''}>${projectCreating ? '<span class="spin"></span>' : icon('agent', 13)} ${t('Create Agent project', '创建研究项目')}</button>` : ''}
          <button class="btn" data-nav="agent">${icon('agent', 13)} ${t('Open Agent Projects', '打开 Agent Projects')}</button>
        </div>
        ${window.EU_IDEA_HANDOFF ? `<div class="note ok mt-12"><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${t('Handoff written', '交接已写入')}</div><div class="d mono">${esc(window.EU_IDEA_HANDOFF.run_id || '')}</div></div></div>` : ''}
        ${projectSeed ? `<div class="note ok mt-12"><div class="ico">${icon('agent', 14)}</div><div class="body"><div class="t">${t('Agent project seed created', '研究项目种子已创建')}</div><div class="d mono">${esc(projectSeed.study_id || '')}</div></div></div>` : ''}
      </div>`;
  }
  function blockedPanel() {
    const rows = (result && result.blocked_features) || [];
    if (!rows.length) return '';
    return `
      <div class="card pad ideas-core-card">
        <div class="eyebrow">BLOCKED UNTIL OPT-IN / NEXT STAGE</div>
        <div class="ledger compact mt-8">${rows.map(r => `<div class="ledger-row"><span class="ledger-ico">${icon(r.status === 'blocked' ? 'lock' : 'clock', 13)}</span><div><b>${esc(r.id)}</b><div class="muted">${esc(r.reason)}</div></div></div>`).join('')}</div>
      </div>`;
  }
  function activeStepPanel() {
    const warning = err ? `<div class="note warn mb-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${esc(err)}</div></div></div>` : '';
    if (activeStep === 'source') return warning + sourceForm();
    if (activeStep === 'ledger') {
      if (!result) return warning + lockedStep('ledger');
      return warning + sourceEvidence() + ideaLedger();
    }
    if (activeStep === 'evidence') {
      if (!result) return warning + lockedStep('evidence');
      return warning + preExperiment() + priorArtPanel() + blockedPanel() + `<div class="ideas-inline-actions mt-14"><button class="btn primary" data-idea-step="handoff">${icon('agent', 13)} ${t('Prepare Agent handoff', '准备 Agent 交接')}</button><button class="btn" data-nav="dictionary">${icon('list', 13)} ${t('Open dictionary', '查看字典')}</button><button class="btn" data-nav="extraction">${icon('extract', 13)} ${t('Use active export', '使用当前导出')}</button></div>`;
    }
    if (activeStep === 'handoff') {
      if (!result) return warning + lockedStep('handoff');
      return warning + planBuilder();
    }
    return warning + sourceForm();
  }
  function recordRows() {
    let rows = [];
    const seen = new Set();
    const push = row => {
      const id = String(row.id || row.runId || '');
      if (id && seen.has(id)) return;
      if (id) seen.add(id);
      rows.push(row);
    };
    ((history && history.runs) || []).slice(0, 8).forEach((r, i) => push({
      id: runRecordKey(r, i) || r.run_id || r.created_at || r.title || 'history',
      runId: r.run_id || '',
      title: r.title || t('Idea run', 'Idea 记录'),
      meta: loadingRun === r.run_id ? t('loading local run...', '正在加载本地 run...') : `${t('Local run', '本地 run')} · ${r.feasibility_tier || '—'} · ${r.journal || 'source'} · ${r.created_at || ''}`,
      status: r.go_no_go === 'recommend' ? 'ready' : 'idle',
    }));
    if (result) {
      const idea = (result.idea_ledger || [])[0] || {};
      const currentKey = selectedRecordKey || runRecordKey(result, 'current') || result.run_id || 'current';
      const current = {
        id: currentKey,
        runId: result.run_id || '',
        title: idea.idea_title || result.candidate_topic || t('Current idea run', '当前 idea run'),
        meta: `${t('Local run', '本地 run')} · ${idea.go_no_go || 'pending'} · ${result.source_type || srcType}`,
        status: idea.go_no_go === 'recommend' ? 'ready' : 'draft',
      };
      if (!rows.some(r => String(r.id) === String(current.id))) rows = [current].concat(rows);
    }
    return rows;
  }
  function ideaList() {
    const rows = recordRows();
    const dotCls = { ready: 'ready', draft: 'draft', idle: 'idle' };
    const activeId = selectedRecordKey || selectedRunId || (result && result.run_id) || (rows[0] && rows[0].id);
    return `
      <div class="ag-list ideas-list">
        <div class="ag-list-head">
          <div><span class="ttl">${t('Local idea runs', '本地 idea run')} · ${rows.length}</span><div class="ag-list-cap">${t('stored on this machine · not Agent analysis runs', '保存在本机 · 不是 Agent 分析运行')}</div></div>
          <button class="ag-newbtn" data-idea-new>${icon('plus', 13)} ${t('New', '新建')}</button>
        </div>
        <div class="ag-studies">
          ${rows.length ? rows.map((r, i) => `
            <button class="studycard ${String(r.id) === String(activeId) ? 'on' : ''}" data-idea-record="${esc(r.runId || r.id)}" data-idea-record-key="${esc(r.id)}" ${loadingRun === (r.runId || r.id) ? 'aria-disabled="true"' : ''}>
              <div class="sc-top">
                <span class="sc-dot ${dotCls[r.status] || 'idle'}"></span>
                <span class="sc-name">${esc(r.title)}</span>
                <span class="sc-mode idea">${t('Idea', '想法')}</span>
              </div>
              <div class="sc-meta"><span class="sc-folder">${icon('folder', 11)} ${esc(r.meta)}</span></div>
            </button>`).join('') : `
            <div class="empty-mini ideas-empty-list">
              <div>${icon('target', 18)}</div>
              <h3>${t('No ideas yet', '还没有想法')}</h3>
              <p>${t('Start with a paper, PDF excerpt, frontier topic, or manual hunch.', '从文章、PDF 摘录、前沿主题或手动直觉开始。')}</p>
            </div>`}
        </div>
      </div>`;
  }
  function ideaDetailHead() {
    const first = result && (result.idea_ledger || [])[0];
    const title = first && first.idea_title ? first.idea_title : t('Idea Mining workspace', 'Idea 挖掘工作台');
    const decision = first && first.go_no_go ? first.go_no_go : t('not mined yet', '尚未挖掘');
    return `
      <div class="ag-dhead ideas-dhead">
        <div class="ag-dtop">
          <div style="min-width:0;">
            <div class="ag-title">${esc(title)}</div>
            <div class="ag-src">
              <span class="lk">${icon('target', 12)} ${esc(modeTitle())}</span>
              <span class="mid"></span>
              <span class="lk">${icon('db', 12)} ${esc(activeSourceLine())}</span>
              <span class="mid"></span>
              <span class="pill ${first && first.go_no_go === 'recommend' ? 'ok' : 'demo'}"><span class="dot"></span>${esc(decision)}</span>
            </div>
          </div>
          <div class="row gap-8">
            <button class="btn sm" data-nav="agent">${icon('agent', 13)} ${t('Open Research Projects', '打开研究项目')}</button>
            <button class="btn sm" data-idea-new>${icon('plus', 13)} ${t('New idea', '新想法')}</button>
          </div>
        </div>
        ${stepSummary()}
      </div>`;
  }
  function ideaShell() {
    return `
      <div class="ag-wrap idea-workbench">
        ${ideaList()}
        <div class="ag-detail">
          ${ideaDetailHead()}
          <div class="ag-body ideas-body">
            <div class="ideas-work-grid">
              ${stepNav()}
              <div class="ideas-step-panel">${activeStepPanel()}</div>
            </div>
          </div>
        </div>
      </div>`;
  }
  function wire(root) {
    const host = root.querySelector('#ideasHost') || root;
    host.addEventListener('click', e => {
      const btn = e.target.closest('[data-idea-record]');
      if (!btn || !host.contains(btn)) return;
      e.preventDefault();
      const runId = btn.dataset.ideaRecord || '';
      const recordKey = btn.dataset.ideaRecordKey || runId;
      if (!runId || loadingRun) return;
      loadingRun = runId;
      selectedRunId = runId;
      selectedRecordKey = recordKey;
      err = null;
      repaint();
      loadIdeaRunApi({ run_id: runId }).then(data => {
        applyRunPayload(data, recordKey);
      }).catch(error => {
        err = error.message || String(error);
      }).finally(() => {
        loadingRun = null;
        repaint();
      });
    });
    root.querySelectorAll('[data-idea-src]').forEach(btn => btn.addEventListener('click', () => {
      collectPayload(document);
      srcType = btn.dataset.ideaSrc || 'manual';
      draft.source_type = srcType;
      repaint();
    }));
    root.querySelectorAll('[data-idea-step]').forEach(btn => btn.addEventListener('click', () => {
      const step = btn.dataset.ideaStep || 'source';
      if (stepState(step) === 'locked') {
        activeStep = 'source';
      } else {
        activeStep = step;
      }
      repaint();
    }));
    root.querySelectorAll('[data-idea-new]').forEach(btn => btn.addEventListener('click', () => {
      result = null; err = null; planEdits = ''; sourceResolved = null; priorArt = null; projectSeed = null; selectedRunId = null; selectedRecordKey = null; draft = {}; activeStep = 'source'; window.EU_IDEA_HANDOFF = null; repaint();
    }));
    const resolveBtn = root.querySelector('[data-idea-resolve]');
    if (resolveBtn) resolveBtn.addEventListener('click', () => {
      if (resolving || !(window.EU_API && window.EU_API.resolveIdeaSource)) return;
      const payload = collectPayload(document);
      resolving = true; err = null;
      repaint();
      window.EU_API.resolveIdeaSource(payload).then(data => {
        sourceResolved = data;
        const s = data.suggested_payload || {};
        draft = Object.assign({}, draft, Object.fromEntries(Object.entries(s).filter(([, v]) => v != null && v !== '')));
      }).catch(e => { err = e.message || String(e); }).finally(() => { resolving = false; repaint(); });
    });
    const mineBtn = root.querySelector('[data-idea-mine]');
    if (mineBtn) mineBtn.addEventListener('click', () => {
      if (mining || !(window.EU_API && window.EU_API.mineIdeas)) return;
      const payload = collectPayload(document);
      const validationError = validatePayload(payload);
      if (validationError) { err = validationError; repaint(); return; }
      mining = true; err = null; result = null; priorArt = null; projectSeed = null; window.EU_IDEA_HANDOFF = null;
      repaint();
      window.EU_API.mineIdeas(payload).then(data => {
        result = data; selectedRunId = data.run_id || null; selectedRecordKey = runRecordKey(data, 'current') || selectedRunId; err = null; planEdits = ''; activeStep = 'ledger'; window.EU_IDEA_LAST_RUN = data; upsertHistoryRun(data);
      }).catch(e => {
        err = e.message || String(e);
      }).finally(() => { mining = false; repaint(); });
    });
    const planBox = root.querySelector('#ideaPlanEdits');
    if (planBox) planBox.addEventListener('input', () => { planEdits = planBox.value; });
    const priorBtn = root.querySelector('[data-idea-prior-art]');
    if (priorBtn) priorBtn.addEventListener('click', () => {
      if (priorArting || !result || !(window.EU_API && window.EU_API.checkIdeaPriorArt)) return;
      const payload = document.querySelector('#ideaTopic') || document.querySelector('#ideaNetworkOptIn') ? collectPayload(document) : draft;
      priorArting = true; err = null;
      repaint();
      window.EU_API.checkIdeaPriorArt({
        run_id: result.run_id,
        idea_id: result.selected_idea_id,
        allow_network: !!payload.allow_network,
      }).then(data => { priorArt = data; activeStep = 'evidence'; }).catch(e => { err = e.message || String(e); }).finally(() => { priorArting = false; repaint(); });
    });
    const handoffBtn = root.querySelector('[data-idea-handoff]');
    if (handoffBtn) handoffBtn.addEventListener('click', () => {
      if (handoffing || !result || !(window.EU_API && window.EU_API.handoffIdea)) return;
      handoffing = true; err = null; planEdits = inputVal(document, '#ideaPlanEdits');
      repaint();
      window.EU_API.handoffIdea({
        run_id: result.run_id,
        idea_id: result.selected_idea_id,
        plan_edits: planEdits,
      }).then(data => {
        window.EU_IDEA_HANDOFF = data;
        projectSeed = null;
        activeStep = 'handoff';
        try { localStorage.setItem('easyicu_last_idea_handoff', JSON.stringify({ run_id: data.run_id, idea_id: data.idea_id, title: data.candidate_topic })); } catch (e) {}
      }).catch(e => { err = e.message || String(e); }).finally(() => { handoffing = false; repaint(); });
    });
    const projectBtn = root.querySelector('[data-idea-create-project]');
    if (projectBtn) projectBtn.addEventListener('click', () => {
      if (projectCreating || !window.EU_IDEA_HANDOFF || !(window.EU_API && window.EU_API.createIdeaAgentProject)) return;
      projectCreating = true; err = null;
      repaint();
      window.EU_API.createIdeaAgentProject({
        run_id: window.EU_IDEA_HANDOFF.run_id,
        idea_id: window.EU_IDEA_HANDOFF.idea_id,
        plan_edits: planEdits,
      }).then(data => {
        projectSeed = data.project || null;
        window.EU_IDEA_AGENT_PROJECT = projectSeed;
        activeStep = 'handoff';
        try { localStorage.setItem('easyicu_last_idea_agent_project', JSON.stringify(projectSeed || {})); } catch (e) {}
      }).catch(e => { err = e.message || String(e); }).finally(() => { projectCreating = false; repaint(); });
    });
  }
  function requestHistory() {
    if (history || !(window.EU_API && window.EU_API.loadIdeaHistory)) return;
    window.EU_API.loadIdeaHistory({ limit: 10 }).then(data => { history = data; repaint(); }).catch(() => {});
  }
  S.ideas = {
    section: 'ideas',
    nav: 'ideas',
    wide: true,
    crumbs: ['Home', 'Idea Mining'],
    status: '<span class="pill ok"><span class="dot"></span> Local-first</span>',
    actionHtml: '<button class="btn sm" data-nav="agent">' + icon('agent', 13) + ' Research Projects</button>',
    rail() {
      const last = result && (result.idea_ledger || [])[0];
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Discovery', '发现')}</span><span class="pill ok" style="height:20px;"><span class="dot"></span>${t('Local', '本地')}</span></div>
        <div class="col gap-6" style="font-size:12px;">
          <div class="setup-row"><span class="k">${t('Mode', '模式')}</span><span class="vv">${esc(srcType)}</span></div>
          <div class="setup-row"><span class="k">${t('Export', '导出')}</span><span class="vv" title="${esc(activeSourceLine())}" style="max-width:118px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(activeSourceShortLine())}</span></div>
          <div class="setup-row"><span class="k">${t('Step', '步骤')}</span><span class="vv">${esc(activeStepLabel())}</span></div>
          <div class="setup-row"><span class="k">${t('Decision', '决策')}</span><span class="vv">${esc((last && last.go_no_go) || '—')}</span></div>
        </div>
        <div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Flow', '流程')}</div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6">${icon('file', 13)} ${t('Source clue', '来源线索')}</div>
          <div class="row gap-6">${icon('target', 13)} ${t('Idea ledger', 'Idea 台账')}</div>
          <div class="row gap-6">${icon('beaker', 13)} ${t('Feasibility', '可行性')}</div>
          <div class="row gap-6">${icon('agent', 13)} ${t('Agent handoff', 'Agent 交接')}</div>
        </div>
      </div>`;
    },
    render() {
      return `
        <div class="page-head" style="margin-bottom:16px;">
          <div class="row" style="justify-content:space-between;align-items:flex-start;gap:16px;">
            <div>
              <div class="eyebrow">DISCOVERY · IDEA MINING · PRE-EXPERIMENT</div>
              <h1 style="margin-top:6px;">${t('Idea Mining', 'Idea 挖掘')}</h1>
              <p class="lead">${t('A workspace for turning papers, review themes, or raw hunches into an auditable idea ledger and an Agent-ready seed.', '把文章、review 主题或研究直觉转成可审计 idea 台账和研究项目可接手的种子。')}</p>
              <div style="font-size:11.5px;color:var(--ink-4);margin-top:9px;">${t('Separated from Research Projects: mining decides what is worth running; Agent Projects runs confirmed analyses.', '已和研究项目拆分：Idea 挖掘判断什么值得做；研究项目只运行确认后的分析。')}</div>
            </div>
          </div>
        </div>
        <div id="ideasHost">${ideaShell()}</div>`;
    },
    afterRender(root) {
      wire(root);
      requestHistory();
    },
  };
})();
