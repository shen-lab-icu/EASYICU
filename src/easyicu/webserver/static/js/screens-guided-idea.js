/* ============================================================
   screens-guided-idea.js — Guided Idea Mining sub-flow owner.
   Owns the idea state (source clue, resolver result, ledger,
   plan draft, prior-art check, handoff, project seed), its
   provider configuration, and the literature-folder browser.

   Split out of screens-guided.js, which had grown to 6000 lines
   holding a conversation engine, four sub-flows and a DOM event
   binder in one closure. This sub-flow was the least entangled
   seam: measured against the whole file it read only 15 of the
   parent functions and 7 of its ~40 mutable closure variables,
   where the sibling "project folder" block of the same size
   touched 39.

   The parent shell passes its callbacks in once through init();
   this module never reaches back into that closure. The three
   pieces of idea state live here and are read through the
   accessors below, so there is one owner for them rather than a
   variable two files both assign.
   ============================================================ */
(function () {
  'use strict';

  const { esc, escAttr: attr } = window.EU_HTML;

  /* Bound by the guided shell at load. The defaults keep an unbound module
     inert rather than throwing halfway through a render. */
  let host = {
    thread: () => [],
    chips: () => [],
    /* A write, not a read: starting the idea flow clears the shell's
       suggestion chips. Passing the array would not work — the shell replaces
       it rather than emptying it. */
    clearChips: () => {},
    activeExportSource: () => null,
    bi: (en, zh) => ({ en, zh }),
    compactHash: (value) => String(value == null ? '' : value),
    compactPath: (value) => String(value == null ? '' : value),
    fmtInt: (value, fallback) => (value == null ? fallback : String(value)),
    fmtNum: (value, fallback) => (value == null ? fallback : String(value)),
    fmtPct: (value) => String(value == null ? '' : value),
    guidedMetricCard: () => '',
    markThrough: () => {},
    pushUser: () => {},
    renderAside: () => {},
    renderChips: () => {},
    renderThread: () => {},
    scheduleGuidedSlotSave: () => {},
    setVal: () => {},
  };

  let guidedIdea = null;
  let guidedIdeaProvider = null;
  let guidedLiteratureBrowser = null;

  function resetGuidedIdeaState() {
    guidedIdea = {
      sourceType: 'manual',
      topic: '',
      excerpt: '',
      title: '',
      journal: '',
      year: '',
      doi: '',
      pmid: '',
      url: '',
      sourceFileName: '',
      sourceFileSha256: '',
      sourceFileBytes: 0,
      sourceFilePages: 0,
      literatureFolder: '',
      literaturePdfCount: 0,
      allowNetwork: false,
      planEdits: '',
      resolving: false,
      pdfIngesting: false,
      literatureScanning: false,
      discovering: false,
      mining: false,
      planning: false,
      priorArting: false,
      handoffing: false,
      projectCreating: false,
      resolved: null,
      result: null,
      planDraft: null,
      prior: null,
      handoff: null,
      project: null,
      pdf: null,
      literatureScan: null,
      discovery: null,
      error: null,
      apiIntroComplete: false,
      sourceEditorOpen: true,
      dataContextConfirmed: false,
    };
    guidedIdeaProvider = createGuidedIdeaProviderState();
    guidedLiteratureBrowser = { open: false, loading: false, error: null, data: null, path: '' };
  }
  function clearGuidedIdeaOutputs(clearResolved) {
    if (!guidedIdea) return;
    if (clearResolved) guidedIdea.resolved = null;
    guidedIdea.result = null;
    guidedIdea.planDraft = null;
    guidedIdea.prior = null;
    guidedIdea.handoff = null;
    guidedIdea.project = null;
    guidedIdea.dataContextConfirmed = false;
  }
  function startGuidedIdeaFlow(label) {
    if (label) host.pushUser(label);
    resetGuidedIdeaState();
    host.setVal({ question: 'idea mining', analysis: 'not started', draft: 'locked' });
    host.markThrough('question', 'active');
    host.thread().push({ bot: true, html: host.bi(
      `First I’ll help you confirm API readiness. Local-only idea mining can continue without an API, but AI synthesis and Agent handoff stay blocked until you explicitly configure a provider.`,
      `第一步先确认 API 就绪状态。本地 idea 挖掘可以不用 API 继续；但 AI 综合和 Agent 交接在你显式配置 provider 前会保持阻断。`,
    ) });
    host.thread().push({ guidedIdeaApiSetup: true });
    host.clearChips();
    host.renderThread();
    host.renderChips();
    host.scheduleGuidedSlotSave('start_idea');
  }
  function guidedIdeaPayload() {
    if (!guidedIdea) resetGuidedIdeaState();
    return {
      source_type: guidedIdea.sourceType || 'manual',
      topic: String(guidedIdea.topic || '').trim(),
      excerpt: String(guidedIdea.excerpt || '').trim(),
      title: String(guidedIdea.title || '').trim(),
      journal: String(guidedIdea.journal || '').trim(),
      year: String(guidedIdea.year || '').trim(),
      doi: String(guidedIdea.doi || '').trim(),
      pmid: String(guidedIdea.pmid || '').trim(),
      url: String(guidedIdea.url || '').trim(),
      source_file_name: String(guidedIdea.sourceFileName || '').trim(),
      source_file_sha256: String(guidedIdea.sourceFileSha256 || '').trim(),
      literature_folder: String(guidedIdea.literatureFolder || '').trim(),
      literature_pdf_count: Number(guidedIdea.literaturePdfCount || 0),
      allow_network: !!guidedIdea.allowNetwork,
    };
  }
  function guidedIdeaHasInput() {
    const p = guidedIdeaPayload();
    return !!(p.topic || p.excerpt || p.title || p.url || p.doi || p.pmid || p.source_file_sha256 || p.literature_folder);
  }
  function applyGuidedIdeaSuggestion(suggested) {
    if (!guidedIdea || !suggested) return;
    const scalarMap = {
      topic: 'topic',
      excerpt: 'excerpt',
      title: 'title',
      journal: 'journal',
      year: 'year',
      doi: 'doi',
      pmid: 'pmid',
      url: 'url',
      source_file_name: 'sourceFileName',
      source_file_sha256: 'sourceFileSha256',
      literature_folder: 'literatureFolder',
    };
    Object.entries(scalarMap).forEach(([from, to]) => {
      if (!guidedIdea[to] && suggested[from]) guidedIdea[to] = String(suggested[from]);
    });
    if (!guidedIdea.literaturePdfCount && suggested.literature_pdf_count != null) {
      guidedIdea.literaturePdfCount = Number(suggested.literature_pdf_count || 0);
    }
  }
  function guidedIdeaSelected() {
    const result = guidedIdea && guidedIdea.result;
    const ideas = result && Array.isArray(result.idea_ledger) ? result.idea_ledger : [];
    const id = result && result.selected_idea_id;
    return ideas.find(row => row.idea_id === id) || ideas[0] || null;
  }
  function guidedIdeaStatusText() {
    if (!guidedIdea) return '';
    if (guidedIdea.resolving) return t('Resolving bounded source metadata...', '正在解析有界来源元数据...');
    if (guidedIdea.discovering) return t('Searching opt-in PubMed metadata and mapping candidate ideas...', '正在按 opt-in 检索 PubMed 元数据并映射候选 idea...');
    if (guidedIdea.mining) return t('Mining local idea ledger and active-export feasibility...', '正在生成本地 idea ledger 并检查 active export 可行性...');
    if (guidedIdea.planning) return t('Drafting or revising the pre-Agent study plan...', '正在生成或修订 Agent 前研究计划...');
    if (guidedIdea.priorArting) return t('Checking prior art under explicit opt-in rules...', '正在按显式 opt-in 规则检查 prior art...');
    if (guidedIdea.handoffing) return t('Writing local handoff plan...', '正在写入本地 handoff plan...');
    if (guidedIdea.projectCreating) return t('Creating metadata-only project seed...', '正在创建 metadata-only 项目种子...');
    if (guidedIdea.error) return esc(guidedIdea.error);
    if (guidedIdea.result && !guidedIdea.dataContextConfirmed) return t('Candidate idea found. Confirm a local export/cohort/module context before feasibility assessment or Agent handoff.', '已找到候选 idea。可行性评估或 Agent 交接前，需要先确认本地导出、队列与模块上下文。');
    if (guidedIdea.project) return t('Metadata-only Agent project seed created. It is not an analysis run or manuscript claim.', 'metadata-only Agent 项目种子已创建；这还不是分析运行，也不是稿件结论。');
    if (guidedIdea.handoff) return t('Handoff draft is frozen. Create a project seed only when you are ready to continue the confirmed setup in Guided Copilot.', '交接草稿已冻结。只有准备在研究引导中继续确认配置时，才创建项目种子。');
    if (guidedIdea.result && guidedIdea.dataContextConfirmed) return t('Data context confirmed. Review feasibility and edit the plan before any Agent handoff.', '数据上下文已确认。请先审阅可行性评估并编辑计划，再做 Agent 交接。');
    if (guidedIdea.resolved) {
      const adapter = guidedIdea.resolved.source_adapter || {};
      const status = String(adapter.status || '');
      if (status.includes('failed') || status.includes('empty') || status.includes('blocked')) {
        return `${t('Source resolver returned', '来源解析返回')} ${status}${adapter.reason ? `: ${adapter.reason}` : ''}`;
      }
      return t('Source metadata resolved. Run local mining next.', '来源元数据已解析。下一步运行本地挖掘。');
    }
    return t('Add a source clue or topic, then run local mining.', '先添加来源线索或主题，然后运行本地挖掘。');
  }
  function guidedIdeaProviderContext() {
    return {
      getState: () => guidedIdeaProvider,
      setState: next => { guidedIdeaProvider = next; },
      getIdea: () => guidedIdea,
      api: () => window.EU_API || {},
      t,
      esc,
      attr,
      icon,
      renderThread: host.renderThread,
    };
  }
  function createGuidedIdeaProviderState(overrides) {
    return window.EU_GUIDED_IDEA_PROVIDER.createState(overrides);
  }
  function requestGuidedIdeaProviderStatus(force) {
    window.EU_GUIDED_IDEA_PROVIDER.requestStatus(guidedIdeaProviderContext(), force);
  }
  function enableGuidedIdeaProvider() {
    window.EU_GUIDED_IDEA_PROVIDER.enableProvider(guidedIdeaProviderContext());
  }
  function saveGuidedIdeaProviderConfig(root) {
    window.EU_GUIDED_IDEA_PROVIDER.saveConfig(guidedIdeaProviderContext(), root);
  }
  function renderGuidedIdeaCapabilityPanel() {
    return window.EU_GUIDED_IDEA_PROVIDER.renderCapabilityPanel(guidedIdeaProviderContext());
  }
  function renderGuidedIdeaApiSetupCard() {
    if (!guidedIdea) resetGuidedIdeaState();
    requestGuidedIdeaProviderStatus(false);
    return window.EU_GUIDED_IDEA_PROVIDER.renderSetupPrompt(guidedIdeaProviderContext());
  }
  function renderGuidedIdeaProviderMiniStatus() {
    requestGuidedIdeaProviderStatus(false);
    return window.EU_GUIDED_IDEA_PROVIDER.renderMiniStatus(guidedIdeaProviderContext());
  }
  function showGuidedIdeaSourceForm() {
    if (!guidedIdea) resetGuidedIdeaState();
    guidedIdea.apiIntroComplete = true;
    const idx = host.thread().findIndex(item => item.guidedIdeaApiSetup);
    if (idx >= 0) host.thread().splice(idx, 1, { guidedIdea: true });
    else if (!host.thread().some(item => item.guidedIdea)) host.thread().push({ guidedIdea: true });
    host.renderThread();
    host.scheduleGuidedSlotSave('continue_idea_after_api_setup');
  }
  function showGuidedIdeaApiSetup() {
    if (!guidedIdea) resetGuidedIdeaState();
    guidedIdea.apiIntroComplete = false;
    const idx = host.thread().findIndex(item => item.guidedIdea);
    if (idx >= 0) host.thread().splice(idx, 1, { guidedIdeaApiSetup: true });
    else if (!host.thread().some(item => item.guidedIdeaApiSetup)) host.thread().push({ guidedIdeaApiSetup: true });
    host.renderThread();
    host.scheduleGuidedSlotSave('return_idea_api_setup');
  }
  function renderGuidedIdeaPdfPicker() {
    const pdf = guidedIdea && guidedIdea.pdf;
    const status = guidedIdea && guidedIdea.pdfIngesting
      ? t('Reading selected PDF...', '正在读取选中的 PDF...')
      : pdf
        ? `${pdf.filename || guidedIdea.sourceFileName || 'PDF'} · ${host.fmtInt(pdf.page_count, '?')} ${t('pages', '页')} · ${host.compactHash(pdf.sha256 || guidedIdea.sourceFileSha256)}`
        : t('Choose a local PDF. Only a bounded excerpt and SHA-256 hash are kept; full text is not stored.', '选择本地 PDF。只保留有界摘录和 SHA-256 哈希，不保存全文。');
    return `
      <div class="gdx-status ${pdf ? 'ok' : ''}">
        <input type="file" accept="application/pdf,.pdf" data-gi-pdf-file hidden />
        <span>${icon(pdf ? 'check' : 'file', 12)}</span>
        <div><strong>${esc(status)}</strong><small>${t('The browser reads the file locally and sends bounded metadata to the local EasyICU server.', '浏览器在本机读取文件，并只把有界元数据交给本机 EasyICU 服务。')}</small></div>
        <button type="button" class="btn sm" data-gi-pdf-pick ${guidedIdea.pdfIngesting ? 'disabled' : ''}>${icon('folder', 12)} ${pdf ? t('Choose another PDF', '换一个 PDF') : t('Choose PDF', '选择 PDF')}</button>
      </div>
      ${pdf && pdf.excerpt ? `<div class="gdi-muted">${t('Extracted excerpt', '已提取摘录')} · ${host.fmtInt(pdf.excerpt_char_count, '0')} chars</div>` : ''}`;
  }
  function renderGuidedLiteratureBrowser() {
    const browser = guidedLiteratureBrowser || {};
    if (!browser.open) return '';
    const data = browser.data || {};
    const entries = Array.isArray(data.entries) ? data.entries : [];
    const shortcuts = Array.isArray(data.shortcuts) ? data.shortcuts : [];
    const currentPath = data.path || browser.path || '';
    const parent = data.parent || '';
    const failed = browser.error || (data && data.ok === false ? data.error : '');
    return `
      <div class="gds-browser" data-guided-literature-browser>
        <div class="gds-browser-head">
          <div>
            <strong>${t('Literature folder picker', '文献库文件夹选择器')}</strong>
            <span>${t('Browse local folders through EasyICU. Select a folder that contains downloaded PDFs.', '通过 EasyICU 浏览本地文件夹。请选择包含已下载 PDF 的文件夹。')}</span>
          </div>
          <button class="btn sm ghost" type="button" data-lit-browser-close>${icon('close', 12)}</button>
        </div>
        <div class="gds-browser-path"><span>${t('Current', '当前')}</span><code>${esc(host.compactPath(currentPath) || t('Home folder', '主目录'))}</code></div>
        ${shortcuts.length ? `<div class="gds-browser-shortcuts">${shortcuts.map((item, i) => `
          <button class="btn sm" type="button" data-lit-browser-shortcut="${i}">${esc(item.name || 'Folder')}</button>`).join('')}</div>` : ''}
        ${failed ? `<div class="gds-browser-message warn">${icon('info', 12)} <span>${esc(String(failed))}</span></div>` : ''}
        <div class="gds-browser-list">
          ${browser.loading ? `<div class="gds-browser-empty">${icon('refresh', 13)} ${t('Loading folders...', '正在加载文件夹...')}</div>` : ''}
          ${!browser.loading && !entries.length ? `<div class="gds-browser-empty">${t('No child folders here. You can still choose the current folder.', '这里没有下级文件夹。也可以直接选择当前文件夹。')}</div>` : ''}
          ${!browser.loading && entries.map((entry, i) => `
            <button class="gds-browser-row" type="button" data-lit-browser-entry="${i}">
              <span class="gds-ico">${icon('folder', 13)}</span>
              <span><strong>${esc(entry.name || 'Folder')}</strong><code>${esc(host.compactPath(entry.path || ''))}</code></span>
              ${entry.hint ? `<em>${esc(entry.hint)}</em>` : ''}
            </button>`).join('')}
        </div>
        <div class="gds-browser-actions">
          <button class="btn sm" type="button" data-lit-browser-up ${parent ? '' : 'disabled'}>${icon('back', 12)} ${t('Up', '上一级')}</button>
          <span class="grow"></span>
          <button class="btn primary sm" type="button" data-lit-browser-use ${currentPath ? '' : 'disabled'}>${icon('check', 12)} ${t('Use this folder', '选择此文件夹')}</button>
        </div>
      </div>`;
  }
  function renderGuidedIdeaLiteraturePicker() {
    const scan = guidedIdea && guidedIdea.literatureScan;
    const docs = scan && Array.isArray(scan.documents) ? scan.documents.slice(0, 4) : [];
    return `
      <div class="gdi-field wide">
        <span>${t('Local literature folder', '本地文献库文件夹')}</span>
        <div class="path-field">
          <span class="pf-path">${esc(guidedIdea.literatureFolder || t('Choose a folder containing local PDFs', '选择一个包含本地 PDF 的文件夹'))}</span>
          <button type="button" class="btn sm" data-gi-lit-browse>${icon('folder', 12)} ${t('Browse...', '浏览...')}</button>
        </div>
        <input data-gi-field="literatureFolder" value="${attr(guidedIdea.literatureFolder || '')}" placeholder="${attr(t('Optional: paste a folder path if browser selection is unavailable', '可选：如果浏览器选择不可用，可以粘贴文件夹路径'))}" />
      </div>
      <div class="gdx-actions">
        <button type="button" class="btn primary" data-gi-lit-scan ${guidedIdea.literatureScanning ? 'disabled' : ''}>${icon('search', 13)} ${guidedIdea.literatureScanning ? t('Scanning PDFs...', '正在扫描 PDF...') : t('Scan literature folder', '扫描文献库文件夹')}</button>
      </div>
      ${renderGuidedLiteratureBrowser()}
      ${scan ? `<div class="gdx-status ok">
        <span>${icon('check', 12)}</span>
        <div><strong>${host.fmtInt((scan.folder || {}).pdf_count, 0)} ${t('PDFs found', '个 PDF 已发现')}</strong><small>${esc(host.compactPath((scan.folder || {}).path || guidedIdea.literatureFolder || ''))}</small></div>
      </div>` : ''}
      ${docs.length ? `<div class="gdi-feature-list">${docs.map(doc => `<div class="gdi-feature-row"><div><strong>${esc(doc.title || doc.filename || 'PDF')}</strong><small>${esc(host.compactPath(doc.path || doc.filename || ''))}</small></div><span>${host.compactHash(doc.sha256 || '')}</span></div>`).join('')}</div>` : ''}`;
  }
  function renderGuidedIdeaDiscovery() {
    const discovery = guidedIdea && guidedIdea.discovery;
    if (!discovery && guidedIdea && guidedIdea.sourceType !== 'frontier') return '';
    const candidates = discovery && Array.isArray(discovery.idea_candidates) ? discovery.idea_candidates.slice(0, 6) : [];
    const queries = discovery && Array.isArray(discovery.queries_to_run) ? discovery.queries_to_run.slice(0, 4) : [];
    return `
      <div class="gdi-prior">
        <div class="gdi-ledger-title">
          <div>
            <span class="gdx-label">${t('Literature discovery', '文献发现')}</span>
            <strong>${esc((discovery && discovery.status) || t('not searched yet', '尚未检索'))}</strong>
          </div>
          <button type="button" class="btn sm ${guidedIdea && guidedIdea.allowNetwork ? 'primary' : ''}" data-gi-discover ${guidedIdea && guidedIdea.discovering ? 'disabled' : ''}>${guidedIdea && guidedIdea.discovering ? '<span class="spin"></span>' : icon('search', 12)} ${t('Discover papers', '检索文章')}</button>
        </div>
        <p>${esc((discovery && discovery.reason) || t('Use this for frontier/review topics. It searches PubMed metadata only after network opt-in, then maps candidate ideas to the EasyICU dictionary and active export.', '用于前沿/review 主题。只有勾选网络 opt-in 后才检索 PubMed 元数据，然后把候选 idea 映射到 EasyICU 字典和当前导出。'))}</p>
        ${queries.length ? `<div class="gdi-query-list">${queries.map(q => `<code>${esc(q)}</code>`).join('')}</div>` : ''}
        ${candidates.length ? `<div class="gdi-feature-list">
          ${candidates.map((row, i) => {
            const idea = row.idea || {};
            const src = row.source || {};
            const feas = idea.feasibility || {};
            return `<div class="gdi-feature-row">
              <div>
                <strong>${esc(idea.idea_title || src.title || 'Candidate idea')}</strong>
                <small>${esc([src.year, src.journal, src.pmid ? 'PMID ' + src.pmid : ''].filter(Boolean).join(' · '))}</small>
              </div>
              <span class="pill ${feas.tier === 'executable' ? 'ok' : 'warn'}">${esc(feas.tier || idea.go_no_go || 'review')}</span>
              <button type="button" class="btn sm" data-gi-discovery-use="${i}">${t('Use', '使用')}</button>
            </div>`;
          }).join('')}
        </div>` : ''}
      </div>`;
  }
  function renderGuidedIdeaSourceFields() {
    if (!guidedIdea) resetGuidedIdeaState();
    const tab = guidedIdea.sourceType || 'manual';
    const tabs = [
      ['manual', t('Manual idea', '手动想法')],
      ['url', t('Article URL', '文章链接')],
      ['pdf', t('PDF file', 'PDF 文件')],
      ['literature_folder', t('Literature folder', '文献库文件夹')],
      ['frontier', t('Frontier topic', '前沿主题')],
    ];
    const optIn = () => `
      <label class="gdi-check">
        <input type="checkbox" data-gi-network ${guidedIdea.allowNetwork ? 'checked' : ''} />
        <span>${t('Allow one bounded network metadata/prior-art request for this source', '允许针对该来源进行一次有界网络元数据/prior-art 请求')}</span>
        <em>opt-in</em>
      </label>`;
    const sourceIntro = (title, body) => `
      <div class="gdi-source-mode-note">
        <strong>${title}</strong>
        <span>${body}</span>
      </div>`;
    let body = '';
    if (tab === 'manual') {
      body = `
        ${sourceIntro(t('Manual idea mode', '手动想法模式'), t('Use this when the user already has a clinical hunch. Keep it lightweight: one question plus the sentence that motivated it.', '用于用户已经有临床直觉的情况。这里保持轻量：一个问题，加一句触发它的来源或理由。'))}
        <label class="gdi-field wide">
          <span>${t('Candidate research question', '候选研究问题')}</span>
          <textarea rows="4" data-gi-field="topic" placeholder="${attr(t('e.g. Does early vasopressor strategy change mortality in septic ICU patients?', '例如：早期升压药策略是否影响脓毒症 ICU 患者死亡率？'))}">${esc(guidedIdea.topic || '')}</textarea>
        </label>
        <label class="gdi-field wide">
          <span>${t('Why this is worth testing', '为什么值得检验')}</span>
          <textarea rows="3" data-gi-field="excerpt" placeholder="${attr(t('Optional: paste the sentence, clinical observation, or rationale that triggered this idea.', '可选：粘贴触发这个想法的句子、临床观察或理由。'))}">${esc(guidedIdea.excerpt || '')}</textarea>
        </label>`;
    } else if (tab === 'url') {
      body = `
        ${sourceIntro(t('Article URL mode', '文章链接模式'), t('Paste the URL first. With source opt-in, EasyICU resolves bounded metadata; without opt-in, you can still add title/DOI manually.', '先粘贴文章链接。勾选来源 opt-in 后，EasyICU 会解析有界元数据；不勾选也可以手动补标题和 DOI。'))}
        <label class="gdi-field wide">
          <span>${t('Article URL', '文章链接')}</span>
          <input data-gi-field="url" value="${attr(guidedIdea.url || '')}" placeholder="https://www.nejm.org/doi/full/..." />
        </label>
        <div class="gdi-meta-grid">
          <label class="gdi-field"><span>DOI / PMID</span><input data-gi-field="doi" value="${attr(guidedIdea.doi || '')}" placeholder="10.xxxx or PMID" /></label>
          <label class="gdi-field"><span>Title</span><input data-gi-field="title" value="${attr(guidedIdea.title || '')}" placeholder="${attr(t('Resolved or manually entered title', '解析或手动输入的标题'))}" /></label>
          <label class="gdi-field"><span>Journal</span><input data-gi-field="journal" value="${attr(guidedIdea.journal || '')}" placeholder="e.g. New England Journal of Medicine" /></label>
          <label class="gdi-field"><span>Year</span><input data-gi-field="year" value="${attr(guidedIdea.year || '')}" placeholder="2026" /></label>
        </div>
        <label class="gdi-field wide">
          <span>${t('Article insight to translate', '要转化的文章启发')}</span>
          <textarea rows="3" data-gi-field="excerpt" placeholder="${attr(t('Optional: paste the abstract sentence or trial/review conclusion that should become an ICU-database question.', '可选：粘贴应转化为 ICU 数据库问题的摘要句、试验结论或 review 结论。'))}">${esc(guidedIdea.excerpt || '')}</textarea>
        </label>
        ${optIn()}`;
    } else if (tab === 'pdf') {
      body = `
        ${sourceIntro(t('PDF file mode', 'PDF 文件模式'), t('Choose a local PDF. The browser sends only bounded metadata/excerpt plus SHA-256 to the local server; the full paper is not kept.', '选择本地 PDF。浏览器只把有界元数据/摘录和 SHA-256 交给本机服务；不会保留全文。'))}
        ${renderGuidedIdeaPdfPicker()}
        <label class="gdi-field wide">
          <span>${t('Question or reading note', '问题或阅读笔记')}</span>
          <textarea rows="3" data-gi-field="topic" placeholder="${attr(t('Optional: describe what you want the PDF to inspire.', '可选：说明你希望这篇 PDF 启发什么方向。'))}">${esc(guidedIdea.topic || '')}</textarea>
        </label>
        <div class="gdi-meta-grid">
          <label class="gdi-field"><span>Title</span><input data-gi-field="title" value="${attr(guidedIdea.title || '')}" placeholder="${attr(t('Auto-filled from PDF when available', '可由 PDF 自动填充'))}" /></label>
          <label class="gdi-field"><span>DOI / PMID</span><input data-gi-field="doi" value="${attr(guidedIdea.doi || '')}" placeholder="10.xxxx or PMID" /></label>
        </div>
        <label class="gdi-field wide">
          <span>${t('Bounded excerpt override', '有界摘录补充')}</span>
          <textarea rows="3" data-gi-field="excerpt" placeholder="${attr(t('Optional: paste a short passage if the PDF parser did not extract the right motivating sentence.', '可选：如果 PDF 解析没有抽到合适的触发句，可以粘贴一小段。'))}">${esc(guidedIdea.excerpt || '')}</textarea>
        </label>`;
    } else if (tab === 'literature_folder') {
      body = `
        ${sourceIntro(t('Literature folder mode', '文献库文件夹模式'), t('Point to a local folder of PDFs, scan metadata locally, then choose or mine candidate ideas from that library.', '指向本地 PDF 文献库文件夹，在本地扫描元数据，然后从这批文献里选择或挖掘候选 idea。'))}
        ${renderGuidedIdeaLiteraturePicker()}
        <label class="gdi-field wide">
          <span>${t('Review scope', '综述范围')}</span>
          <textarea rows="3" data-gi-field="topic" placeholder="${attr(t('e.g. septic shock resuscitation, ARDS ventilation, AKI staging in ICU cohorts', '例如：脓毒性休克复苏、ARDS 通气、ICU 队列中的 AKI 分期。'))}">${esc(guidedIdea.topic || '')}</textarea>
        </label>`;
    } else {
      body = `
        ${sourceIntro(t('Frontier topic mode', '前沿主题模式'), t('Use this when there is no single source yet. It can prepare queries locally, and only searches PubMed metadata after explicit network opt-in.', '用于还没有单篇来源的情况。它会先在本地准备检索式，只有显式网络 opt-in 后才检索 PubMed 元数据。'))}
        <label class="gdi-field wide">
          <span>${t('Frontier topic / journal scope', '前沿主题 / 期刊范围')}</span>
          <textarea rows="4" data-gi-field="topic" placeholder="${attr(t('e.g. unresolved ICU research questions from recent sepsis or ARDS reviews', '例如：近期脓毒症或 ARDS 综述提出的未解决 ICU 研究问题。'))}">${esc(guidedIdea.topic || '')}</textarea>
        </label>
        <div class="gdi-meta-grid">
          <label class="gdi-field"><span>${t('Journal scope', '期刊范围')}</span><input data-gi-field="journal" value="${attr(guidedIdea.journal || '')}" placeholder="NEJM, ICM, JAMA, Lancet..." /></label>
          <label class="gdi-field"><span>${t('Year window', '年份窗口')}</span><input data-gi-field="year" value="${attr(guidedIdea.year || '')}" placeholder="2024-2026" /></label>
        </div>
        ${optIn()}
        ${renderGuidedIdeaDiscovery()}`;
    }
    return `
      <div class="gdi-tabs" role="group" aria-label="Idea source type">
        ${tabs.map(([key, label]) => `<button type="button" class="${tab === key ? 'on' : ''}" data-gi-source="${key}">${label}</button>`).join('')}
      </div>
      <div class="gdi-form source-${attr(tab)}">${body}</div>`;
  }
  function renderGuidedIdeaEvidence(result) {
    const src = ((result && result.source_evidence) || [])[0] || {};
    if (!src.source_id && !src.title) return '';
    return `
      <div class="gdi-source-card">
        <div class="gdx-label">${t('Source evidence', '来源证据')}</div>
        <strong>${esc(src.title || 'Untitled source')}</strong>
        <small>${[src.year, src.journal, src.doi || src.pmid].filter(Boolean).map(esc).join(' · ') || esc(src.source_type || 'manual')}</small>
        ${src.evidence_quote ? `<blockquote>${esc(src.evidence_quote)}</blockquote>` : ''}
        <div class="gdi-muted">${t('Only metadata, a bounded quote, and hashes are persisted.', '仅持久化元数据、有界摘录和哈希。')}</div>
      </div>`;
  }
  function renderGuidedIdeaSourceSummary(result) {
    const src = ((result && result.source_evidence) || [])[0] || {};
    return `
      <div class="gdx-status ok">
        <span>${icon('check', 12)}</span>
        <div>
          <strong>${esc(src.title || guidedIdea.title || guidedIdea.topic || t('Source locked for this candidate', '当前候选已绑定来源'))}</strong>
          <small>${esc([src.year || guidedIdea.year, src.journal || guidedIdea.journal, src.doi || guidedIdea.doi || src.pmid || guidedIdea.pmid].filter(Boolean).join(' · ') || t('metadata-only source snapshot', 'metadata-only 来源快照'))}</small>
        </div>
        <button type="button" class="btn sm" data-gi-edit-source>${t('Edit source', '修改来源')}</button>
      </div>`;
  }
  function renderGuidedIdeaFlowGuide() {
    const result = guidedIdea && guidedIdea.result;
    const confirmed = !!(guidedIdea && guidedIdea.dataContextConfirmed);
    const planned = !!(guidedIdea && (guidedIdea.planDraft || (guidedIdea.result && guidedIdea.result.idea_plan)));
    const handoffReady = !!(guidedIdea && guidedIdea.handoff);
    const projectReady = !!(guidedIdea && guidedIdea.project);
    const stages = [
      [t('1 Source clue', '1 来源线索'), result ? 'done' : 'active', t('article / PDF / topic only', '仅文章 / PDF / 主题')],
      [t('2 Candidate idea', '2 候选 idea'), result ? 'done' : 'locked', t('question + dictionary mapping', '问题 + 字典映射')],
      [t('3 Data context', '3 数据上下文'), confirmed ? 'done' : result ? 'active' : 'locked', t('export, cohort, modules', '导出、队列、模块')],
      [t('4 Plan / replan', '4 计划 / replan'), projectReady || handoffReady ? 'done' : planned ? 'done' : confirmed ? 'active' : 'locked', t('draft first, then Agent handoff', '先计划，再交接 Agent')],
    ];
    return `<div class="gdi-flow">${stages.map(([label, state, note]) => `
      <div class="gdi-flow-step ${state}">
        <span>${state === 'done' ? icon('check', 11, 3) : state === 'locked' ? icon('lock', 10) : icon('spark', 11)}</span>
        <div><strong>${label}</strong><small>${note}</small></div>
      </div>`).join('')}</div>`;
  }
  function renderGuidedIdeaLedger(idea) {
    if (!idea) return '';
    const concepts = (idea.mapped_concepts || []).slice(0, 10);
    const feasibility = idea.feasibility || {};
    const prior = idea.prior_art || {};
    return `
      <div class="gdi-ledger">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">Idea ledger</span><strong>${esc(idea.idea_title || 'Candidate idea')}</strong></div>
          <span class="pill ${idea.go_no_go === 'recommend' ? 'ok' : 'warn'}">${esc(idea.go_no_go || 'hold')}</span>
        </div>
        <p>${esc(idea.rationale || '')}</p>
        <div class="gdi-ledger-grid">
          <div>
            <span>${t('Mapped concepts', '映射概念')}</span>
            <div class="gdi-tags">${concepts.map(row => `<code>${esc(row.concept_id || row.label)} · ${esc(row.tier || '')}</code>`).join('') || `<em>${t('No dictionary mapping yet', '暂无字典映射')}</em>`}</div>
          </div>
          <div><span>${t('Feasibility', '可行性')}</span><strong>${esc(feasibility.label || feasibility.tier || 'unknown')}</strong><small>${esc(feasibility.reason || '')}</small></div>
          <div><span>${t('Prior art', '既有研究')}</span><strong>${esc(prior.status || 'not checked')}</strong><small>${esc(prior.opportunity_frame || prior.reason || '')}</small></div>
          <div><span>${t('Next action', '下一步')}</span><strong>${esc(idea.next_action || idea.go_no_go_reason || 'review')}</strong></div>
        </div>
      </div>`;
  }
  function guidedIdeaPreExperimentReady(result) {
    const pre = result && result.pre_experiment;
    if (!pre) return false;
    const status = String(pre.status || '').toLowerCase();
    return !!status && !/(blocked|missing|not_configured|no_active|failed|unavailable)/.test(status);
  }
  function guidedIdeaActiveExportText(result) {
    const pre = result && result.pre_experiment;
    const cohort = pre && pre.cohort ? pre.cohort : {};
    const src = host.activeExportSource && host.activeExportSource();
    const label = (src && (src.label || src.database)) || (pre && (pre.export_label || pre.source_label)) || t('active local export', '当前本地导出');
    const bits = [
      label,
      cohort.entities != null ? `${host.fmtInt(cohort.entities)} ${t('entities', '实体')}` : '',
      cohort.modules != null ? `${host.fmtInt(cohort.modules)} ${t('modules', '模块')}` : '',
      cohort.total_rows != null ? `${host.fmtInt(cohort.total_rows)} ${t('rows', '行')}` : '',
    ].filter(Boolean);
    return bits.join(' · ');
  }
  function renderGuidedIdeaDataContext(result, idea) {
    if (!result) return '';
    const ready = guidedIdeaPreExperimentReady(result);
    const confirmed = !!guidedIdea.dataContextConfirmed;
    const mapped = idea && Array.isArray(idea.mapped_concepts) ? idea.mapped_concepts.length : 0;
    const pre = result.pre_experiment || {};
    const needsAction = !confirmed;
    return `
      <div class="gdi-pre ${confirmed ? 'ok' : ''} ${needsAction ? 'needs-action' : ''}">
        ${needsAction ? `<div class="gdi-next-cue">${icon('arrow', 12)} ${ready ? t('Next step — confirm the data context to unlock feasibility & Agent handoff', '下一步 —— 确认数据上下文，解锁可行性与 Agent 交接') : t('Next step — prepare or choose a local export, then confirm the data context', '下一步 —— 先准备或选择本地导出，再确认数据上下文')}</div>` : ''}
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">${t('Step 3 · Data context', '第 3 步 · 数据上下文')}</span><strong>${confirmed ? t('Confirmed active export for feasibility review', '已确认使用当前导出做可行性审阅') : t('Not confirmed yet', '尚未确认')}</strong></div>
          <span class="pill ${confirmed ? 'ok' : ready ? 'warn' : 'bad'}">${confirmed ? t('confirmed', '已确认') : ready ? t('needs user confirmation', '需要用户确认') : t('needs data', '需要数据')}</span>
        </div>
        <p>${esc(ready
          ? t('I detected an active local export and used it only for a feasibility scan. This does not mean the study cohort, feature modules, or final extraction are finished; confirm it only if this is the data context you want to continue with.', '我检测到了当前本地导出，并且只把它用于可行性扫描。这不代表本次研究的队列、特征模块或最终抽取已经完成；只有当你确认要沿用这个数据上下文时才继续。')
          : t('No usable active export is confirmed for this idea. Prepare or register a local export first, then come back to feasibility review.', '当前 idea 还没有可用且已确认的 active export。请先准备或注册本地导出，再回来做可行性审阅。'))}</p>
        <div class="gdi-ledger-grid">
          <div><span>${t('Detected export', '检测到的导出')}</span><strong>${esc(ready ? guidedIdeaActiveExportText(result) : t('none confirmed', '尚未确认'))}</strong></div>
          <div><span>${t('Mapped concepts', '映射概念')}</span><strong>${host.fmtInt(mapped, '0')}</strong><small>${t('candidate dictionary links only', '仅候选字典映射')}</small></div>
          <div><span>${t('Feasibility status', '可行性状态')}</span><strong>${esc(pre.status || 'not available')}</strong><small>${esc(pre.reason || t('aggregate-only feasibility; no patient rows shown', '仅聚合可行性；不展示患者行'))}</small></div>
          <div><span>${t('Agent handoff', 'Agent 交接')}</span><strong>${confirmed ? t('unlocked for plan review', '已解锁计划审阅') : t('locked', '已锁定')}</strong><small>${t('requires explicit data-context confirmation', '需要显式确认数据上下文')}</small></div>
        </div>
        <div class="gdx-actions">
          <button type="button" class="btn primary" data-gi-confirm-data ${ready || confirmed ? '' : 'disabled'}>${icon('check', 13)} ${confirmed ? t('Data context confirmed', '数据上下文已确认') : t('Use this active export for feasibility', '使用当前导出做可行性审阅')}</button>
          <button type="button" class="btn" data-guided-goal="data_extraction">${icon('folder', 13)} ${t('Prepare or choose data first', '先准备或选择数据')}</button>
        </div>
      </div>`;
  }
  function renderGuidedIdeaPreExperiment(result) {
    const pre = result && result.pre_experiment;
    if (!pre) return '';
    const cohort = pre.cohort || {};
    const stats = (pre.feature_statistics || []).slice(0, 8);
    return `
      <div class="gdi-pre">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">${t('Feasibility on active export', 'active export 可行性评估')}</span><strong>${esc(pre.status || 'blocked')}</strong></div>
          <span class="pill">${esc(pre.payload_scope || 'aggregate')}</span>
        </div>
        ${pre.reason ? `<p>${esc(pre.reason)}</p>` : ''}
        <div class="gdi-stats">
          ${host.guidedMetricCard(t('Entities', '实体数'), host.fmtInt(cohort.entities, 'n/a'))}
          ${host.guidedMetricCard(t('Modules', '模块'), host.fmtInt(cohort.modules, 'n/a'))}
          ${host.guidedMetricCard(t('Feature checks', '特征检查'), host.fmtInt(stats.length, '0'))}
          ${host.guidedMetricCard(t('Rows', '行数'), host.fmtInt(cohort.total_rows, 'n/a'))}
        </div>
        ${stats.length ? `<div class="gdi-feature-list">
          ${stats.map(row => {
            const isEvent = row && row.metric_kind === 'event_rate';
            const metricValue = isEvent ? Number(row.event_rate_pct || 0) : Number(row.coverage_pct || 0);
            const metricLabel = isEvent ? t('event rate', '事件率') : t('coverage', '覆盖率');
            return `<div class="gdi-feature-row">
              <div><strong>${esc(row.label || row.concept_id)}</strong><small>${esc(row.concept_id || '')} · ${esc(row.module || '')}</small></div>
              <div class="gdi-feature-bar"><span style="width:${Math.max(0, Math.min(100, metricValue))}%"></span></div>
              <span>${metricLabel}: ${host.fmtPct(isEvent ? row.event_rate_pct : row.coverage_pct)}</span>
              <small>${esc(guidedFeatureSummary(row))}</small>
            </div>`;
          }).join('')}
        </div>` : ''}
        ${(pre.interpretation || []).length ? `<div class="gdr-note">${pre.interpretation.map(row => esc(row)).join('<br>')}</div>` : ''}
      </div>`;
  }
  function guidedFeatureSummary(row) {
    if (row && row.metric_kind === 'event_rate') {
      return `${t('events', '事件')} ${host.fmtInt(row.event_entities ?? row.records, '0')} · ${t('non-events', '非事件')} ${host.fmtInt(row.non_event_entities, '0')} · ${t('negative cases are not missing', '阴性患者不是缺失')}`;
    }
    const summary = row && row.numeric_summary;
    if (summary && typeof summary === 'object') {
      if (summary.available) {
        return `n=${host.fmtInt(summary.n, '0')} · median ${host.fmtNum(summary.median, 'n/a')} · ${host.fmtNum(summary.min, 'n/a')} to ${host.fmtNum(summary.max, 'n/a')}`;
      }
      return t('non-numeric or empty', '非数值或为空');
    }
    return String(summary || (row && row.status) || '');
  }
  function renderGuidedIdeaPrior() {
    if (!guidedIdea || (!guidedIdea.prior && !guidedIdea.result)) return '';
    const prior = (guidedIdea.prior && guidedIdea.prior.prior_art) || (guidedIdea.result && guidedIdea.result.prior_art) || {};
    const queries = prior.queries_to_run || [];
    const results = prior.results || [];
    return `
      <div class="gdi-prior">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">${t('Literature inspiration', '已有文献与启发')}</span><strong>${esc(prior.status || 'not checked')}</strong></div>
          <button type="button" class="btn sm" data-gi-prior ${guidedIdea.priorArting ? 'disabled' : ''}>${icon('search', 12)} ${t('Check literature', '检查已有文献')}</button>
        </div>
        <p>${esc(prior.reason || t('Optional bounded network metadata search. It does not use an LLM provider, and it stays blocked until you explicitly opt in for this source.', '可选有界网络元数据搜索。它不使用 LLM provider，且在当前来源显式 opt-in 前保持阻断。'))}</p>
        ${queries.length ? `<div class="gdi-query-list">${queries.slice(0, 4).map(q => `<code>${esc(q)}</code>`).join('')}</div>` : ''}
        ${results.length ? `<div class="gdi-feature-list">${results.slice(0, 5).map(row => `<div class="gdi-feature-row"><div><strong>${esc(row.title || 'result')}</strong><small>${esc([row.year, row.journal, row.pmid].filter(Boolean).join(' · '))}</small></div><span>${esc(row.database || '')}</span></div>`).join('')}</div>` : ''}
      </div>`;
  }
  function renderGuidedIdeaHandoff() {
    if (!window.EU_GUIDED_IDEA_PLAN || !window.EU_GUIDED_IDEA_PLAN.render) return '';
    return window.EU_GUIDED_IDEA_PLAN.render({
      getIdea: () => guidedIdea,
      selectedIdea: guidedIdeaSelected,
      t,
      esc,
      attr,
      icon,
    });
  }
  function renderGuidedIdeaCard() {
    if (!guidedIdea) resetGuidedIdeaState();
    requestGuidedIdeaProviderStatus(false);
    const idea = guidedIdeaSelected();
    const result = guidedIdea.result;
    const miningBlocked = guidedIdea.mining || guidedIdea.resolving;
    const showSourceEditor = !result || guidedIdea.sourceEditorOpen;
    return `
      <div class="gd-idea-card">
        <div class="gdx-head">
          <span class="gdx-ico">${icon('spark', 15)}</span>
          <div>
            <strong>${t('Mine a study idea inside Copilot', '在 Copilot 内挖掘研究想法')}</strong>
            <span>${t('This only turns a source clue into a candidate research question. Data source, cohort, feature modules, and Agent handoff stay separate guided steps.', '这里仅把来源线索转成候选研究问题。数据源、队列、特征模块和 Agent 交接仍是后续独立引导步骤。')}</span>
          </div>
        </div>
        ${renderGuidedIdeaFlowGuide()}
        ${renderGuidedIdeaProviderMiniStatus()}
        ${showSourceEditor ? renderGuidedIdeaSourceFields() : renderGuidedIdeaSourceSummary(result)}
        <div class="gdx-status ${guidedIdea.error ? 'bad' : result ? 'ok' : ''}">
          <span>${icon(guidedIdea.error ? 'x' : result ? 'check' : 'shield', 12)}</span>
          <div><strong>${guidedIdeaStatusText()}</strong><small>${t('No patient rows, full papers, external calls, or provider clients unless you explicitly opt in and provider readiness passes.', '不会返回患者行、全文、外部调用或 provider client，除非你显式 opt-in 且 provider readiness 通过。')}</small></div>
        </div>
        ${showSourceEditor ? `<div class="gdx-actions">
          <button type="button" class="btn" data-gi-resolve ${guidedIdea.resolving ? 'disabled' : ''}>${icon('search', 13)} ${t('Resolve source', '解析来源')}</button>
          <button type="button" class="btn primary" data-gi-mine ${miningBlocked ? 'disabled' : ''}>${icon('play', 13)} ${t('Mine locally', '本地挖掘 idea')}</button>
          <button type="button" class="btn" data-guided-goal="data_extraction">${t('Prepare data first', '先准备数据')}</button>
        </div>` : ''}
        ${showSourceEditor ? '' : renderGuidedIdeaEvidence(result)}
        ${renderGuidedIdeaLedger(idea)}
        ${renderGuidedIdeaDataContext(result, idea)}
        ${guidedIdea.dataContextConfirmed ? renderGuidedIdeaPreExperiment(result) : ''}
        ${guidedIdea.dataContextConfirmed ? renderGuidedIdeaPrior() : ''}
        ${renderGuidedIdeaHandoff()}
        ${guidedIdea.dataContextConfirmed && guidedIdea.project ? `<div class="gdx-status ok"><span>${icon('check', 12)}</span><div><strong>${t('Agent project seed created', 'Agent project seed 已创建')}</strong><small>${esc(host.compactPath((guidedIdea.project.project || {}).project_dir || (guidedIdea.project.project || {}).study_id || ''))}</small></div></div>` : ''}
      </div>`;
  }
  function loadGuidedLiteratureBrowser(path) {
    if (!guidedLiteratureBrowser) guidedLiteratureBrowser = { open: false, loading: false, error: null, data: null, path: '' };
    guidedLiteratureBrowser.open = true;
    guidedLiteratureBrowser.loading = true;
    guidedLiteratureBrowser.error = null;
    guidedLiteratureBrowser.path = String(path || guidedLiteratureBrowser.path || guidedIdea.literatureFolder || '');
    host.renderThread();
    if (!window.EU_API || !window.EU_API.listDir) {
      guidedLiteratureBrowser.loading = false;
      guidedLiteratureBrowser.error = t('Local folder picker API is unavailable.', '本地文件夹选择 API 不可用。');
      host.renderThread();
      return;
    }
    window.EU_API.listDir(guidedLiteratureBrowser.path)
      .then(result => {
        guidedLiteratureBrowser.loading = false;
        guidedLiteratureBrowser.data = result || {};
        guidedLiteratureBrowser.path = (result && result.path) || guidedLiteratureBrowser.path || '';
        guidedLiteratureBrowser.error = result && result.ok === false ? (result.error || 'folder_error') : null;
        host.renderThread();
      })
      .catch(err => {
        guidedLiteratureBrowser.loading = false;
        guidedLiteratureBrowser.error = String(err && err.message || err || 'folder_error');
        host.renderThread();
      });
  }
  function ingestGuidedIdeaPdfFile(file) {
    if (!guidedIdea || !file) return;
    const name = String(file.name || '');
    if (!/\.pdf$/i.test(name) && file.type && file.type !== 'application/pdf') {
      guidedIdea.error = t('Choose a PDF file.', '请选择 PDF 文件。');
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.ingestIdeaPdf) {
      guidedIdea.error = t('PDF ingestion API is unavailable.', 'PDF 解析 API 不可用。');
      host.renderThread();
      return;
    }
    guidedIdea.sourceType = 'pdf';
    guidedIdea.pdfIngesting = true;
    guidedIdea.error = null;
    host.renderThread();
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      const contentBase64 = text.includes(',') ? text.split(',').pop() : text;
      window.EU_API.ingestIdeaPdf({ filename: name, content_base64: contentBase64 })
        .then(result => {
          guidedIdea.pdfIngesting = false;
          guidedIdea.pdf = result && result.pdf ? result.pdf : null;
          if (guidedIdea.pdf) {
            guidedIdea.sourceFileName = guidedIdea.pdf.filename || name;
            guidedIdea.sourceFileSha256 = guidedIdea.pdf.sha256 || '';
            guidedIdea.sourceFileBytes = Number(guidedIdea.pdf.bytes || file.size || 0);
            guidedIdea.sourceFilePages = Number(guidedIdea.pdf.page_count || 0);
          }
          applyGuidedIdeaSuggestion(result && result.suggested_payload);
          guidedIdea.resolved = result;
          host.renderThread();
          host.scheduleGuidedSlotSave('ingest_idea_pdf');
        })
        .catch(err => {
          guidedIdea.pdfIngesting = false;
          guidedIdea.error = err.message || String(err);
          host.renderThread();
          host.scheduleGuidedSlotSave('ingest_idea_pdf_error');
        });
    };
    reader.onerror = () => {
      guidedIdea.pdfIngesting = false;
      guidedIdea.error = t('Could not read the selected PDF file.', '无法读取选中的 PDF 文件。');
      host.renderThread();
    };
    reader.readAsDataURL(file);
  }
  function scanGuidedLiteratureFolder() {
    if (!guidedIdea || guidedIdea.literatureScanning) return;
    const path = String(guidedIdea.literatureFolder || '').trim();
    if (!path) {
      guidedIdea.error = t('Choose or paste a local literature folder first.', '请先选择或粘贴本地文献库文件夹。');
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.scanIdeaLiteratureFolder) {
      guidedIdea.error = t('Literature-folder scan API is unavailable.', '文献库文件夹扫描 API 不可用。');
      host.renderThread();
      return;
    }
    guidedIdea.sourceType = 'literature_folder';
    guidedIdea.literatureScanning = true;
    guidedIdea.error = null;
    host.renderThread();
    window.EU_API.scanIdeaLiteratureFolder({ path })
      .then(result => {
        guidedIdea.literatureScanning = false;
        guidedIdea.literatureScan = result;
        const folder = (result && result.folder) || {};
        guidedIdea.literatureFolder = folder.path || path;
        guidedIdea.literaturePdfCount = Number(folder.pdf_count || 0);
        applyGuidedIdeaSuggestion(result && result.suggested_payload);
        guidedIdea.resolved = result;
        host.renderThread();
        host.scheduleGuidedSlotSave('scan_idea_literature_folder');
      })
      .catch(err => {
        guidedIdea.literatureScanning = false;
        guidedIdea.error = err.message || String(err);
        host.renderThread();
        host.scheduleGuidedSlotSave('scan_idea_literature_folder_error');
      });
  }
  function runGuidedIdeaResolve() {
    if (!guidedIdea || guidedIdea.resolving) return;
    if (!guidedIdeaHasInput()) {
      guidedIdea.error = t('Add a topic, source quote, title, DOI, PMID, URL, local PDF, or literature folder first.', '请先添加主题、来源句子、标题、DOI、PMID、URL、本地 PDF 或文献库文件夹。');
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.resolveIdeaSource) {
      guidedIdea.error = 'Idea source backend is unavailable.';
      host.renderThread();
      return;
    }
    guidedIdea.resolving = true;
    guidedIdea.error = null;
    clearGuidedIdeaOutputs(false);
    host.renderThread();
    window.EU_API.resolveIdeaSource(guidedIdeaPayload()).then(result => {
      guidedIdea.resolving = false;
      guidedIdea.resolved = result;
      applyGuidedIdeaSuggestion(result && result.suggested_payload);
      host.renderThread();
      host.scheduleGuidedSlotSave('resolve_idea_source');
    }).catch(err => {
      guidedIdea.resolving = false;
      guidedIdea.error = err.message || String(err);
      host.renderThread();
      host.scheduleGuidedSlotSave('resolve_idea_source_error');
    });
  }
  function runGuidedIdeaDiscover() {
    if (!guidedIdea || guidedIdea.discovering) return;
    const topic = String(guidedIdea.topic || guidedIdea.title || '').trim();
    if (!topic) {
      guidedIdea.error = t('Describe a frontier topic or review scope before literature discovery.', '请先描述前沿主题或 review 范围，再做文献发现。');
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.discoverIdeas) {
      guidedIdea.error = t('Literature discovery backend is unavailable.', '文献发现后端不可用。');
      host.renderThread();
      return;
    }
    guidedIdea.sourceType = 'frontier';
    guidedIdea.discovering = true;
    guidedIdea.error = null;
    host.renderThread();
    window.EU_API.discoverIdeas(Object.assign({}, guidedIdeaPayload(), {
      topic,
      journal: guidedIdea.journal || '',
      limit: 8,
      allow_network: !!guidedIdea.allowNetwork,
    })).then(result => {
      guidedIdea.discovering = false;
      guidedIdea.discovery = result;
      if (result && result.suggested_payload && result.status !== 'blocked_network_opt_in_required') {
        applyGuidedIdeaSuggestion(result.suggested_payload);
      }
      host.renderThread();
      host.scheduleGuidedSlotSave('discover_idea_literature');
    }).catch(err => {
      guidedIdea.discovering = false;
      guidedIdea.error = err.message || String(err);
      host.renderThread();
      host.scheduleGuidedSlotSave('discover_idea_literature_error');
    });
  }
  function useGuidedIdeaDiscoveryCandidate(index) {
    if (!guidedIdea || !guidedIdea.discovery) return;
    const rows = Array.isArray(guidedIdea.discovery.idea_candidates) ? guidedIdea.discovery.idea_candidates : [];
    const row = rows[Number(index || 0)];
    if (!row) return;
    applyGuidedIdeaSuggestion(row.suggested_payload || {});
    guidedIdea.sourceType = 'frontier';
    guidedIdea.resolved = {
      ok: true,
      mode: 'frontier_discovery_candidate',
      resolved_source: row.source || null,
      suggested_payload: row.suggested_payload || {},
      source_adapter: {
        status: 'pubmed_candidate_selected',
        network_calls: guidedIdea.discovery.network_calls || 0,
        external_llm_calls: 0,
      },
    };
    guidedIdea.error = null;
    host.renderThread();
    host.scheduleGuidedSlotSave('use_discovered_idea_candidate');
  }
  function runGuidedIdeaMine() {
    if (!guidedIdea || guidedIdea.mining) return;
    if (!guidedIdeaHasInput()) {
      guidedIdea.error = t('Add a topic, source quote, title, DOI, PMID, URL, local PDF, or literature folder first.', '请先添加主题、来源句子、标题、DOI、PMID、URL、本地 PDF 或文献库文件夹。');
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.mineIdeas) {
      guidedIdea.error = 'Idea mining backend is unavailable.';
      host.renderThread();
      return;
    }
    guidedIdea.mining = true;
    guidedIdea.error = null;
    host.renderThread();
    window.EU_API.mineIdeas(guidedIdeaPayload()).then(result => {
      guidedIdea.mining = false;
      guidedIdea.result = result;
      guidedIdea.planDraft = null;
      guidedIdea.handoff = null;
      guidedIdea.project = null;
      guidedIdea.dataContextConfirmed = false;
      guidedIdea.sourceEditorOpen = false;
      const idea = guidedIdeaSelected();
      host.setVal({
        question: idea ? (idea.idea_title || 'idea ledger') : 'idea ledger',
        data: t('confirm export/context', '确认导出/上下文'),
        cohort: t('not confirmed', '尚未确认'),
        concepts: t('candidate mappings only', '仅候选映射'),
        analysis: t('not started', '未开始'),
      });
      host.markThrough('data', 'active');
      host.renderThread();
      host.renderAside();
      host.scheduleGuidedSlotSave('mine_idea');
    }).catch(err => {
      guidedIdea.mining = false;
      guidedIdea.error = err.message || String(err);
      host.renderThread();
      host.scheduleGuidedSlotSave('mine_idea_error');
    });
  }
  function confirmGuidedIdeaDataContext() {
    if (!guidedIdea || !guidedIdea.result) return;
    if (!guidedIdeaPreExperimentReady(guidedIdea.result)) {
      guidedIdea.error = t('Prepare or register a local EasyICU export before confirming the data context.', '请先准备或注册本地 EasyICU 导出，再确认数据上下文。');
      host.renderThread();
      return;
    }
    const idea = guidedIdeaSelected();
    guidedIdea.dataContextConfirmed = true;
    guidedIdea.error = null;
    host.setVal({
      data: guidedIdeaActiveExportText(guidedIdea.result),
      cohort: t('from active export · review denominator next', '来自当前导出 · 下一步审阅分母'),
      concepts: idea && Array.isArray(idea.mapped_concepts)
        ? `${host.fmtInt(idea.mapped_concepts.length)} ${t('candidate mappings', '个候选映射')}`
        : t('review mapped concepts', '审阅映射概念'),
      analysis: t('feasibility only', '仅可行性'),
    });
    host.markThrough('cohort', 'active');
    host.renderThread();
    host.renderAside();
    host.scheduleGuidedSlotSave('confirm_idea_data_context');
  }
  function runGuidedIdeaPlan(mode) {
    if (!guidedIdea || guidedIdea.planning) return;
    const idea = guidedIdeaSelected();
    if (!idea || !guidedIdea.result) {
      guidedIdea.error = t('Run local idea mining before creating a study plan.', '请先完成本地 idea 挖掘，再生成研究计划。');
      host.renderThread();
      return;
    }
    if (!guidedIdea.dataContextConfirmed) {
      guidedIdea.error = t('Confirm the local export/cohort/module context before drafting a study plan.', '生成研究计划前，请先确认本地导出、队列和模块上下文。');
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.planIdea) {
      guidedIdea.error = t('Idea planning backend is unavailable.', 'Idea 计划后端不可用。');
      host.renderThread();
      return;
    }
    guidedIdea.planning = true;
    guidedIdea.error = null;
    host.renderThread();
    window.EU_API.planIdea({
      run_id: guidedIdea.result.run_id,
      idea_id: idea.idea_id,
      mode: mode || (guidedIdea.planDraft ? 'replan' : 'plan'),
      plan_edits: guidedIdea.planEdits || '',
    }).then(result => {
      guidedIdea.planning = false;
      guidedIdea.planDraft = result;
      guidedIdea.handoff = null;
      guidedIdea.project = null;
      host.setVal({ analysis: t('plan draft ready', '计划草案已生成'), draft: t('locked', '已锁定') });
      host.renderThread();
      host.renderAside();
      host.scheduleGuidedSlotSave(mode === 'replan' ? 'replan_idea' : 'plan_idea');
    }).catch(err => {
      guidedIdea.planning = false;
      guidedIdea.error = err.message || String(err);
      host.renderThread();
      host.scheduleGuidedSlotSave('plan_idea_error');
    });
  }
  function runGuidedIdeaPriorArt() {
    if (!guidedIdea || guidedIdea.priorArting) return;
    const idea = guidedIdeaSelected();
    if (!idea || !guidedIdea.result) {
      guidedIdea.error = 'Run local idea mining before prior-art check.';
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.checkIdeaPriorArt) {
      guidedIdea.error = 'Prior-art backend is unavailable.';
      host.renderThread();
      return;
    }
    guidedIdea.priorArting = true;
    guidedIdea.error = null;
    host.renderThread();
    window.EU_API.checkIdeaPriorArt({
      run_id: guidedIdea.result.run_id,
      idea_id: idea.idea_id,
      allow_network: !!guidedIdea.allowNetwork,
    }).then(result => {
      guidedIdea.priorArting = false;
      guidedIdea.prior = result;
      host.renderThread();
      host.scheduleGuidedSlotSave('check_idea_prior_art');
    }).catch(err => {
      guidedIdea.priorArting = false;
      guidedIdea.error = err.message || String(err);
      host.renderThread();
      host.scheduleGuidedSlotSave('check_idea_prior_art_error');
    });
  }
  function runGuidedIdeaHandoff() {
    if (!guidedIdea || guidedIdea.handoffing) return;
    const idea = guidedIdeaSelected();
    if (!idea || !guidedIdea.result) {
      guidedIdea.error = 'Run local idea mining before creating a handoff.';
      host.renderThread();
      return;
    }
    if (!guidedIdea.dataContextConfirmed) {
      guidedIdea.error = t('Confirm the local export/cohort/module context before freezing an Agent handoff.', '冻结 Agent 交接前，请先确认本地导出、队列和模块上下文。');
      host.renderThread();
      return;
    }
    if (!guidedIdea.planDraft && !(guidedIdea.result && guidedIdea.result.idea_plan)) {
      guidedIdea.error = t('Generate and review the study plan before freezing an Agent handoff.', '请先生成并审阅研究计划，再冻结交接给 Agent。');
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.handoffIdea) {
      guidedIdea.error = 'Idea handoff backend is unavailable.';
      host.renderThread();
      return;
    }
    guidedIdea.handoffing = true;
    guidedIdea.error = null;
    host.renderThread();
    window.EU_API.handoffIdea({
      run_id: guidedIdea.result.run_id,
      idea_id: idea.idea_id,
      plan_edits: guidedIdea.planEdits || '',
    }).then(result => {
      guidedIdea.handoffing = false;
      guidedIdea.handoff = result;
      host.setVal({ analysis: 'handoff ready', draft: 'locked' });
      host.renderThread();
      host.renderAside();
      host.scheduleGuidedSlotSave('handoff_idea');
    }).catch(err => {
      guidedIdea.handoffing = false;
      guidedIdea.error = err.message || String(err);
      host.renderThread();
      host.scheduleGuidedSlotSave('handoff_idea_error');
    });
  }
  function runGuidedIdeaCreateProject() {
    if (!guidedIdea || guidedIdea.projectCreating) return;
    const idea = guidedIdeaSelected();
    if (!idea || !guidedIdea.result) {
      guidedIdea.error = 'Run local idea mining before creating an Agent project.';
      host.renderThread();
      return;
    }
    if (!guidedIdea.dataContextConfirmed) {
      guidedIdea.error = t('Confirm the local export/cohort/module context before creating an Agent project seed.', '创建 Agent 项目种子前，请先确认本地导出、队列和模块上下文。');
      host.renderThread();
      return;
    }
    if (!window.EU_API || !window.EU_API.createIdeaAgentProject) {
      guidedIdea.error = 'Agent project seed backend is unavailable.';
      host.renderThread();
      return;
    }
    guidedIdea.projectCreating = true;
    guidedIdea.error = null;
    host.renderThread();
    window.EU_API.createIdeaAgentProject({
      run_id: guidedIdea.result.run_id,
      idea_id: idea.idea_id,
      plan_edits: guidedIdea.planEdits || '',
    }).then(result => {
      guidedIdea.projectCreating = false;
      guidedIdea.project = result;
      host.setVal({ analysis: 'Agent seed ready', draft: 'locked' });
      host.renderThread();
      host.renderAside();
      host.scheduleGuidedSlotSave('create_idea_agent_project');
    }).catch(err => {
      guidedIdea.projectCreating = false;
      guidedIdea.error = err.message || String(err);
      host.renderThread();
      host.scheduleGuidedSlotSave('create_idea_agent_project_error');
    });
  }


  function isGuidedIdeaIntent(text) {
    const s = String(text || '').toLowerCase();
    return /idea|study idea|research idea|paper|article|pdf|literature|frontier|review topic|研究想法|研究问题|挖掘|论文|文章|文献|综述|前沿|选题/.test(s);
  }

  /* --------------------------------------------------------------------
     Session slots. The shell built the idea section of its snapshot by
     reaching into this state 45 times and restored it field by field; it
     asks this module for the section instead. Field names and the exact
     fallbacks are unchanged — the wire format is what a saved session
     already holds.
     -------------------------------------------------------------------- */

  function slotSnapshot() {
    if (!guidedIdea) return null;
    const idea = guidedIdeaSelected();
    return {
      source_type: guidedIdea.sourceType || 'manual',
      topic: guidedIdea.topic || '',
      excerpt: guidedIdea.excerpt || '',
      title: guidedIdea.title || '',
      journal: guidedIdea.journal || '',
      year: guidedIdea.year || '',
      doi: guidedIdea.doi || '',
      pmid: guidedIdea.pmid || '',
      url: guidedIdea.url || '',
      allow_network: !!guidedIdea.allowNetwork,
      plan_edits: guidedIdea.planEdits || '',
      data_context_confirmed: !!guidedIdea.dataContextConfirmed,
      run_id: guidedIdea.result && guidedIdea.result.run_id,
      selected_idea_id: idea && idea.idea_id,
      plan_created_at: guidedIdea.planDraft && guidedIdea.planDraft.created_at,
      handoff_id: guidedIdea.handoff && (guidedIdea.handoff.handoff_id || guidedIdea.handoff.run_id),
      agent_project_dir: guidedIdea.project && guidedIdea.project.project && guidedIdea.project.project.project_dir,
    };
  }

  function restoreSlot(slot) {
    if (!slot || typeof slot !== 'object') return;
    resetGuidedIdeaState();
    guidedIdea.sourceType = slot.source_type || guidedIdea.sourceType;
    guidedIdea.topic = slot.topic || '';
    guidedIdea.excerpt = slot.excerpt || '';
    guidedIdea.title = slot.title || '';
    guidedIdea.journal = slot.journal || '';
    guidedIdea.year = slot.year || '';
    guidedIdea.doi = slot.doi || '';
    guidedIdea.pmid = slot.pmid || '';
    guidedIdea.url = slot.url || '';
    guidedIdea.allowNetwork = !!slot.allow_network;
    guidedIdea.planEdits = slot.plan_edits || '';
    guidedIdea.dataContextConfirmed = !!slot.data_context_confirmed;
    guidedIdea.sourceEditorOpen = !slot.run_id;
    if (slot.run_id) {
      guidedIdea.result = { ok: true, run_id: slot.run_id, idea_ledger: [] };
      restoreArtifacts(slot.run_id);
    }
  }

  function restoreArtifacts(runId) {
    const expectedRun = String(runId || '').trim();
    if (!expectedRun || !window.EU_API || !window.EU_API.loadIdeaRun) return;
    window.EU_API.loadIdeaRun({ run_id: expectedRun }).then(data => {
      if (!guidedIdea || !data || data.run_id !== expectedRun) return;
      guidedIdea.result = data;
      guidedIdea.planDraft = data.idea_plan || null;
      guidedIdea.handoff = data.handoff || null;
      guidedIdea.prior = data.prior_art_check ? { prior_art: data.prior_art_check } : null;
      guidedIdea.project = data.agent_project ? { ok: true, project: data.agent_project } : null;
      const notes = data.handoff && data.handoff.handoff_plan && data.handoff.handoff_plan.human_plan_notes;
      if (notes) guidedIdea.planEdits = notes;
      host.renderThread();
      host.renderAside();
    }).catch(err => {
      console.warn('[EasyICU] Guided idea artifact restore failed:', err);
    });
  }

  /* --------------------------------------------------------------------
     State transitions the shell used to perform by assigning the closure
     variables directly. They are methods now so this module stays the only
     writer of its own state.
     -------------------------------------------------------------------- */

  function clearIdeaState() {
    guidedIdea = null;
    guidedLiteratureBrowser = null;
  }

  function selectProvider(name) {
    guidedIdeaProvider = createGuidedIdeaProviderState({
      provider: name || 'openai',
      configOpen: !!(guidedIdeaProvider && guidedIdeaProvider.configOpen),
    });
  }

  function toggleProviderConfig() {
    if (!guidedIdeaProvider) guidedIdeaProvider = { provider: 'openai' };
    guidedIdeaProvider = Object.assign({}, guidedIdeaProvider, {
      configOpen: !guidedIdeaProvider.configOpen,
      saveError: null,
    });
  }

  window.EU_GUIDED_IDEA = {
    init(bindings) {
      host = Object.assign({}, host, bindings || {});
    },
    state: () => guidedIdea,
    provider: () => guidedIdeaProvider,
    literatureBrowser: () => guidedLiteratureBrowser,
    clearIdeaState,
    slotSnapshot,
    restoreSlot,
    restoreArtifacts,
    selectProvider,
    toggleProviderConfig,
    clearGuidedIdeaOutputs,
    confirmGuidedIdeaDataContext,
    createGuidedIdeaProviderState,
    enableGuidedIdeaProvider,
    guidedIdeaSelected,
    ingestGuidedIdeaPdfFile,
    loadGuidedLiteratureBrowser,
    renderGuidedIdeaApiSetupCard,
    renderGuidedIdeaCard,
    requestGuidedIdeaProviderStatus,
    resetGuidedIdeaState,
    runGuidedIdeaCreateProject,
    runGuidedIdeaDiscover,
    runGuidedIdeaHandoff,
    runGuidedIdeaMine,
    runGuidedIdeaPlan,
    runGuidedIdeaPriorArt,
    runGuidedIdeaResolve,
    saveGuidedIdeaProviderConfig,
    scanGuidedLiteratureFolder,
    showGuidedIdeaApiSetup,
    showGuidedIdeaSourceForm,
    startGuidedIdeaFlow,
    useGuidedIdeaDiscoveryCandidate,
    isGuidedIdeaIntent,
  };
})();
