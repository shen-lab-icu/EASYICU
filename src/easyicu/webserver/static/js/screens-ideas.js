/* Screen: Idea Mining — first-class discovery workflow.
   Local-first Stage67: user-supplied metadata/excerpt -> idea ledger ->
   dictionary/export feasibility assessment -> Agent handoff plan. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  let srcType = 'manual';
  let mining = false;
  let resolving = false;
  let discovering = false;
  let priorArting = false;
  let planning = false;
  let handoffing = false;
  let projectCreating = false;
  let loadingRun = null;
  let err = null;
  let result = null;
  let sourceResolved = null;
  let discovery = null;
  let priorArt = null;
  let planDraft = null;
  let projectSeed = null;
  let sampleChecking = false;
  let sampleFeasibility = null;
  let selectedRunId = null;
  let selectedRecordKey = null;
  let history = null;
  let planEdits = '';
  let draft = {};
  let activeStep = 'source';
  let pdfIngesting = false;
  let pdfInfo = null;
  let literatureScanning = false;
  let literatureScan = null;
  let zoteroWidget = null;

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
    if (activeStep === 'handoff') return t('Plan / replan', '计划 / replan');
    return t('Source', '来源');
  }
  function pubmedConnectorEnabled() {
    const policy = window.EU_CAPABILITIES || {};
    const policySettings = policy.settings || {};
    if (Object.prototype.hasOwnProperty.call(policySettings, 'connector_pubmed_enabled')) {
      return policySettings.connector_pubmed_enabled !== false;
    }
    const settings = window.EU_SETTINGS || {};
    return settings.connector_pubmed_enabled !== false;
  }
  function sourceNetworkOptIn() {
    return fieldValue('allow_network') === 'true' || fieldValue('allow_network') === true;
  }
  function networkAllowed() {
    return pubmedConnectorEnabled() && sourceNetworkOptIn();
  }
  function normalizePlanStep(row, i) {
    if (row && typeof row === 'object') {
      return {
        phase: row.phase || t('Step', '步骤'),
        title: row.title || row.action || t('Plan step', '计划步骤'),
        action: row.action || '',
        output: row.output || '',
        guardrail: row.guardrail || '',
      };
    }
    const text = String(row || '').trim();
    const lower = text.toLowerCase();
    const has = (needle) => lower.includes(needle);
    const mk = (phase, title, action, output, guardrail) => ({ phase, title, action, output, guardrail });
    if (has('lock the clinical question')) {
      return mk(
        t('Question', '问题'),
        t('Freeze the clinical question and estimand', '锁定临床问题和估计目标'),
        t('Confirm population, exposure or index time, comparator, outcome, and analysis window before reading any effect estimate.', '先确认人群、暴露或时间零点、比较组、结局和分析窗口，再看任何效应估计。'),
        t('One locked PICOT-style question.', '一条锁定的 PICOT 风格问题。'),
        t('Idea Mining proposes the question; it has not finished cohort selection or analysis.', 'Idea 挖掘只提出问题，还没有完成队列筛选或分析。')
      );
    }
    if (has('confirm the active easyicu export') || has('cohort denominator')) {
      return mk(
        t('Data context', '数据上下文'),
        t('Confirm export, cohort, and modules', '确认导出、队列和模块'),
        t('Select the real local export, denominator, required modules, and concept dictionary mappings with the user.', '和用户确认真实本地导出、分母、所需模块和概念映射。'),
        t('Confirmed export/cohort/module contract for Agent Projects.', '交给研究项目的导出/队列/模块契约。'),
        t('MOCK or demo exports are UI rehearsal only.', 'MOCK 或演示导出只能用于界面演练。')
      );
    }
    if (has('outcome-blind feasibility') || has('missingness structure')) {
      return mk(
        t('Feasibility', '可行性'),
        t('Run outcome-blind feasibility assessment', '运行不看结局效应的可行性评估'),
        t('Check concept availability, joint completeness, time-index support, missingness, and event rate before modeling.', '建模前检查概念可用性、联合完整度、时间索引、缺失结构和事件率。'),
        t('Feasibility table with denominators and blockers.', '包含分母和阻断项的可行性表。'),
        t('Do not present feasibility as a clinical finding.', '不能把可行性检查当成临床结论。')
      );
    }
    if (has('treatment-strategy comparison') || has('timing anchors')) {
      return mk(
        t('Design', '设计'),
        t('Translate the article into an ICU treatment-strategy question', '把文章转译成 ICU 治疗策略问题'),
        t('Define vasopressor/fluid timing anchors, exposure summaries, comparator groups, and eligible shock or sepsis windows.', '定义升压药/补液的时间锚点、暴露摘要、比较组，以及休克或脓毒症窗口。'),
        t('Treatment-strategy contrast ready for descriptive review.', '可进入描述性审阅的治疗策略对照。'),
        t('Flag confounding by indication and immortal-time risk before modeling.', '建模前标记适应症混杂和 immortal-time 风险。')
      );
    }
    if (has('balance and sensitivity') || has('sensitivity checks')) {
      return mk(
        t('Robustness', '稳健性'),
        t('Predefine balance and sensitivity checks', '预先定义平衡性和敏感性检查'),
        t('Compare baseline severity, missingness, exposure timing, and alternative dose or window definitions.', '比较基线严重程度、缺失、暴露时序，以及替代剂量或窗口定义。'),
        t('Sensitivity checklist for replan.', '用于 replan 的敏感性清单。'),
        t('Keep claims exploratory unless assumptions are audited.', '除非假设被审计，否则结论保持探索性。')
      );
    }
    if (has('prior-art') || has('literature')) {
      return mk(
        t('Prior art', '既有文献'),
        t('Use existing literature as an inspiration map', '把已有文献当成启发地图'),
        t('Check whether prior studies answered the same question or suggest better comparators, subgroups, timing, or outcomes.', '检查既有研究是否已经回答同一问题，或提示更合适的比较组、亚组、时序和结局。'),
        t('Already answered, partially answered, or new exploratory angle.', '判定为已回答、部分回答，或新的探索角度。'),
        t('Prior work shapes novelty; it does not automatically kill the idea.', '既有研究塑造创新点，不会自动否定 idea。')
      );
    }
    if (has('agent projects') || has('handoff')) {
      return mk(
        t('Agent handoff', 'Agent 交接'),
        t('Create a project seed only after confirmation', '确认后再创建研究项目种子'),
        t('Send the locked question, feasibility table, literature interpretation, and analysis steps to Agent Projects.', '把锁定问题、可行性表、文献解释和分析步骤交给研究项目。'),
        t('Metadata-only project seed.', '仅元数据的项目种子。'),
        t('Manuscript claims remain blocked until evidence checks and human sign-off pass.', '证据核验和人工签署前，论文结论保持锁定。')
      );
    }
    return mk(
      t('Step', '步骤'),
      text || `${t('Plan step', '计划步骤')} ${i + 1}`,
      '',
      '',
      t('Review this legacy planning note before Agent handoff.', '交给 Agent 前需要审阅这条历史计划说明。')
    );
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
  function takeGuidedHandoff() {
    const handoff = window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.take
      ? window.EU_GUIDED_HANDOFF.take('ideas') : null;
    if (!handoff || !handoff.prefill) return;
    const hint = String(handoff.prefill.question_hint || '').trim();
    if (hint && !fieldValue('topic')) draft = Object.assign({}, draft || {}, { topic: hint });
  }
  function guidedPrefillNote() {
    return window.EU_GUIDED_HANDOFF && window.EU_GUIDED_HANDOFF.noteHtml
      ? window.EU_GUIDED_HANDOFF.noteHtml('ideas') : '';
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
      pmid: inputValOr(root, '#ideaPmid', 'pmid'),
      url: inputValOr(root, '#ideaUrl', 'url'),
      abstract: inputValOr(root, '#ideaAbstract', 'abstract'),
      citation_key: inputValOr(root, '#ideaCitationKey', 'citation_key'),
      zotero_key: inputValOr(root, '#ideaZoteroKey', 'zotero_key'),
      source_origin: inputValOr(root, '#ideaSourceOrigin', 'source_origin'),
      source_origin_label: inputValOr(root, '#ideaSourceOriginLabel', 'source_origin_label'),
      source_file_name: (pdfInfo && pdfInfo.filename) || fieldValue('source_file_name'),
      source_file_sha256: (pdfInfo && pdfInfo.sha256) || fieldValue('source_file_sha256'),
      literature_folder: inputValOr(root, '#ideaLiteratureFolder', 'literature_folder'),
      literature_pdf_count: literatureScan && literatureScan.folder ? Number(literatureScan.folder.pdf_count || 0) : Number(fieldValue('literature_pdf_count') || 0),
      allow_network: pubmedConnectorEnabled() && (opt ? !!opt.checked : !!(draft && draft.allow_network)),
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
  function discoverIdeasApi(body) {
    if (window.EU_API && window.EU_API.discoverIdeas) return window.EU_API.discoverIdeas(body);
    return postLocalJSON('/api/ideas/discover', body || {});
  }
  function sourceTabs() {
    const rows = [
      ['manual', 'Manual idea', '手动想法', 'Start from a clinical hunch.'],
      ['url', 'Article URL', '文章链接', 'Resolve bounded article metadata from a DOI or URL.'],
      ['pdf', 'PDF file', 'PDF 文件', 'Choose a local PDF; only bounded metadata is retained.'],
      ['literature_folder', 'Literature folder', '文献库文件夹', 'Scan a local folder of PDFs.'],
      ['zotero', 'Zotero library', 'Zotero 文献库', 'Search the local Zotero Desktop library.'],
      ['frontier', 'Frontier topic', '前沿主题', 'Prepare or opt in to PubMed metadata discovery.'],
    ];
    return `
      <div class="modeswitch ideas-source-switch" data-ideas-tabs>
        ${rows.map(r => `<button class="${srcType === r[0] ? 'on' : ''}" data-idea-src="${r[0]}" title="${esc(r[3])}">${t(r[1], r[2])}</button>`).join('')}
      </div>`;
  }
  function modeTitle() {
    if (srcType === 'url') return t('Article URL seed', '文章链接种子');
    if (srcType === 'pdf') return t('PDF file seed', 'PDF 文件种子');
    if (srcType === 'literature_folder') return t('Literature folder seed', '文献库文件夹种子');
    if (srcType === 'zotero') return t('Zotero library seed', 'Zotero 文献库种子');
    if (srcType === 'frontier') return t('Frontier topic seed', '前沿主题种子');
    return t('Manual idea seed', '手动想法种子');
  }
  function modePrimaryLabel() {
    if (srcType === 'frontier') return t('Frontier topic / journal scope', '前沿主题 / 期刊范围');
    if (srcType === 'url') return t('Question this article suggests', '这篇文章启发的问题');
    if (srcType === 'pdf') return t('Question from this excerpt', '这段摘录启发的问题');
    if (srcType === 'literature_folder') return t('Review scope for this library', '这批文献的综述范围');
    if (srcType === 'zotero') return t('Question this Zotero paper suggests', '这篇 Zotero 文献启发的问题');
    return t('Idea / research question', '想法 / 研究问题');
  }
  function modeExcerptLabel() {
    if (srcType === 'frontier') return t('Reasoning notes or review theme', '推理备注或 review 主题');
    if (srcType === 'pdf') return t('Bounded PDF excerpt', '有界 PDF 摘录');
    if (srcType === 'url') return t('Abstract / quoted trigger sentence', '摘要 / 触发原文句子');
    if (srcType === 'literature_folder') return t('Library notes', '文献库备注');
    if (srcType === 'zotero') return t('Abstract / trigger sentence', '摘要 / 触发句');
    return t('Source quote or rationale sentence', '来源引用或触发句');
  }
  function modePlaceholder() {
    if (srcType === 'frontier') return 'e.g. ICU long-term outcomes after septic shock, editorials/reviews in Intensive Care Medicine';
    if (srcType === 'url') return 'e.g. What ICU-database study could test this trial or review insight?';
    if (srcType === 'pdf') return 'e.g. The excerpt suggests a measurable ICU exposure, outcome, or subgroup.';
    if (srcType === 'literature_folder') return 'e.g. septic shock resuscitation reviews, ARDS ventilation editorials';
    if (srcType === 'zotero') return 'e.g. What ICU-database question should this paper become?';
    return 'e.g. Vasopressor-first resuscitation and mortality among adult septic shock ICU patients';
  }
  function sourceModeHint() {
    if (srcType === 'url') {
      return t('Article URL mode resolves bounded metadata from a DOI or URL after source opt-in. Add the key quote if you want the ledger to preserve the article-specific rationale.', '文章链接模式会在来源 opt-in 后从 DOI 或 URL 解析有界元数据。若希望台账保留文章特定理由，请补一条关键引文。');
    }
    if (srcType === 'pdf') {
      return t('PDF mode reads a selected local PDF through the browser and sends only bounded metadata, excerpt, and SHA-256 to the local server.', 'PDF 模式通过浏览器读取本地 PDF，只把有界元数据、摘录和 SHA-256 交给本机服务。');
    }
    if (srcType === 'literature_folder') {
      return t('Literature folder mode scans local PDFs for metadata and candidate source clues. No full paper text or patient rows are stored in this screen.', '文献库文件夹模式会扫描本地 PDF 元数据和候选来源线索。本页不保存全文或患者行。');
    }
    if (srcType === 'zotero') {
      return t('Zotero mode can auto-connect to Zotero Desktop, or you can paste DOI, BibTeX, RIS, or title/abstract directly. EasyICU stores metadata, citation key, hash, and a bounded abstract excerpt only.', 'Zotero 模式可自动连接 Zotero Desktop，也可直接粘贴 DOI、BibTeX、RIS 或标题摘要。EasyICU 只保存元数据、citation key、哈希和有界摘要摘录。');
    }
    if (srcType === 'frontier') {
      return t('Frontier mode can prepare queries without network access. If you opt in, it searches bounded PubMed metadata/abstracts, maps candidate ideas to the EasyICU dictionary, then lets you choose one for the local ledger.', '前沿模式在未联网时只准备检索式；如果你 opt-in，会检索有界 PubMed 元数据/摘要，把候选 idea 映射到 EasyICU 字典，并让你选择其中一个生成本地台账。');
    }
    return t('Manual mode creates a local, evidence-bound idea ledger from the text you provide.', '手动模式会根据你提供的文本生成本地、证据绑定的 idea 台账。');
  }
  function sourceModeGuide() {
    const rows = {
      manual: [
        [t('Type one clinical question or hunch', '写一句临床问题或直觉'), t('No paper is required; a rationale sentence is optional.', '不需要文章；触发理由可选。')],
        [t('Create a local idea ledger', '生成本地 idea 台账'), t('This checks dictionary fit and active-export feasibility before any Agent run.', '先检查字典匹配和当前导出可行性，不会启动 Agent。')],
      ],
      url: [
        [t('Paste the article link', '粘贴文章链接'), t('Then resolve metadata if network opt-in is allowed, or fill the title manually.', '允许网络 opt-in 时可解析元数据；也可以手动补标题。')],
        [t('Translate the article into an ICU question', '把文章转成 ICU 问题'), t('The article is inspiration, not proof that the same analysis is already done.', '文章是启发，不等于同一分析已经完成。')],
      ],
      pdf: [
        [t('Choose a local PDF', '选择本地 PDF'), t('The browser reads it locally; this screen retains bounded metadata/excerpt and SHA-256.', '浏览器本地读取；本页只保留有界元数据/摘录和 SHA-256。')],
        [t('Add the ICU question it suggests', '补充它启发的 ICU 问题'), t('If parsing misses the key sentence, paste a short bounded excerpt.', '如果解析没抓到关键句，粘贴一小段有界摘录。')],
      ],
      literature_folder: [
        [t('Point to a local paper folder', '选择本地文献文件夹'), t('Scan PDF metadata and representative snippets without storing full text.', '扫描 PDF 元数据和代表摘录，不保存全文。')],
        [t('Define the review scope', '定义综述范围'), t('Use the folder to surface candidate ICU-database questions, then choose one.', '用这批文献提出候选 ICU 数据库问题，再选择其中一个。')],
      ],
      zotero: [
        [t('Auto-connect or paste a source', '自动连接或直接粘贴文献'), t('Search Zotero Desktop when available, or paste DOI/BibTeX/RIS/title metadata with no setup.', '可在 Zotero Desktop 可用时检索，也可无需配置直接粘贴 DOI/BibTeX/RIS/标题元数据。')],
        [t('Use one paper as the source clue', '把一篇文献作为来源线索'), t('The title, DOI, abstract excerpt, and citation key feed the local idea ledger.', '标题、DOI、摘要摘录和 citation key 会进入本地 idea 台账。')],
      ],
      frontier: [
        [t('Describe the frontier topic', '描述前沿主题'), t('Without opt-in, EasyICU only prepares bounded queries and a local rationale.', '未 opt-in 时只准备有界检索式和本地理由。')],
        [t('Opt in only when ready', '需要时再 opt-in'), t('Bounded PubMed metadata search can then map candidate papers to EasyICU concepts.', '之后可用有界 PubMed 元数据检索，把候选文献映射到 EasyICU 概念。')],
      ],
    }[srcType] || [];
    return `
      <div class="ideas-source-gate ${esc(srcType)}">
        <div class="ideas-source-gate-title">${esc(modeTitle())}</div>
        <div class="ideas-source-gate-steps">
          ${rows.map((row, i) => `<div><span>${String(i + 1).padStart(2, '0')}</span><b>${row[0]}</b><em>${row[1]}</em></div>`).join('')}
        </div>
      </div>`;
  }
  function optionalMetadataBlock(body, title) {
    return `
      <details class="ideas-secondary-fields">
        <summary>${icon('list', 13)} ${title || t('Optional metadata', '可选元数据')} <span>${t('can fill later', '可稍后补')}</span></summary>
        <div class="ideas-secondary-body">${body}</div>
      </details>`;
  }
  function validatePayload(payload) {
    const hasTopic = !!payload.topic;
    const hasExcerpt = !!payload.excerpt;
    const hasTitle = !!payload.title;
    if (srcType === 'url' && !payload.url) return t('Paste the article URL, then add a title, abstract, excerpt, or rationale sentence.', '请先粘贴文章链接，并补充标题、摘要、摘录或触发句。');
    if (srcType === 'url' && !(hasTopic || hasExcerpt || hasTitle || payload.doi)) return t('Resolve the URL or add a title, DOI, abstract, excerpt, or rationale sentence before mining.', '请先解析链接，或补充标题、DOI、摘要、摘录或触发句再挖掘。');
    if (srcType === 'pdf' && !(hasExcerpt || payload.source_file_sha256 || hasTitle)) return t('Choose a local PDF or paste a bounded PDF excerpt before mining.', '请先选择本地 PDF 或粘贴有界 PDF 摘录。');
    if (srcType === 'literature_folder' && !payload.literature_folder) return t('Choose or paste a local literature folder before mining.', '请先选择或粘贴本地文献库文件夹。');
    if (srcType === 'zotero' && !(hasTopic || hasExcerpt || hasTitle || payload.zotero_key)) return t('Search Zotero and select a paper, or add a title, question, or bounded abstract excerpt before mining.', '请先检索 Zotero 并选择一篇文献，或补充标题、问题、或有界摘要摘录再挖掘。');
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
      pmid: source.pmid || '',
      url: source.url || '',
      citation_key: source.citation_key || '',
      zotero_key: source.zotero_key || '',
      source_origin: source.source_origin || '',
      source_origin_label: source.source_origin_label || '',
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
    planDraft = data && data.idea_plan ? data.idea_plan : null;
    sampleFeasibility = data && data.bounded_sample_feasibility ? data.bounded_sample_feasibility : null;
    window.EU_IDEA_HANDOFF = data && data.handoff ? data.handoff : null;
    projectSeed = data && data.agent_project ? data.agent_project : null;
    draft = draftFromRun(data || {});
    srcType = draft.source_type || 'manual';
    planEdits = ((window.EU_IDEA_HANDOFF || {}).handoff_plan || {}).human_plan_notes || '';
    sourceResolved = null;
    err = null;
    activeStep = window.EU_IDEA_HANDOFF || projectSeed || planDraft ? 'handoff' : data ? 'ledger' : 'source';
  }
  function runIdeaDiscovery() {
    if (discovering) return;
    const payload = collectPayload(document);
    if (!payload.topic && !payload.title) {
      err = t('Describe a frontier topic or journal scope before literature discovery.', '请先描述前沿主题或期刊范围，再做文献发现。');
      repaint();
      return;
    }
    discovering = true;
    err = null;
    repaint();
    discoverIdeasApi(Object.assign({}, payload, { limit: 8 })).then(data => {
      discovery = data;
      sourceResolved = null;
      const suggested = data && data.status !== 'blocked_network_opt_in_required' ? data.suggested_payload : null;
      if (suggested) {
        draft = Object.assign({}, draft, Object.fromEntries(Object.entries(suggested).filter(([, v]) => v != null && v !== '')));
      }
    }).catch(e => {
      err = e.message || String(e);
    }).finally(() => {
      discovering = false;
      repaint();
    });
  }
  function applySuggestedPayload(suggested) {
    if (!suggested) return;
    draft = Object.assign({}, draft, Object.fromEntries(Object.entries(suggested).filter(([, v]) => v != null && v !== '')));
  }
  function ideaZoteroWidget() {
    if (zoteroWidget) return zoteroWidget;
    if (!(window.EU_IDEA_ZOTERO && window.EU_IDEA_ZOTERO.create)) return null;
    zoteroWidget = window.EU_IDEA_ZOTERO.create({
      t,
      icon,
      fieldValue,
      collectPayload,
      applySuggestedPayload,
      repaint,
      setError(value) { err = value; },
      setSourceResolved(value) { sourceResolved = value; },
      setSourceType(value) { srcType = value; draft.source_type = value; },
      ensureTopicFromTitle() { if (!draft.topic && draft.title) draft.topic = draft.title; },
    });
    return zoteroWidget;
  }
  function ingestPdfFile(file) {
    if (!file || pdfIngesting) return;
    const name = String(file.name || '');
    if (!/\.pdf$/i.test(name) && file.type && file.type !== 'application/pdf') {
      err = t('Choose a PDF file.', '请选择 PDF 文件。');
      repaint();
      return;
    }
    if (!(window.EU_API && window.EU_API.ingestIdeaPdf)) {
      err = t('PDF ingestion API is unavailable.', 'PDF 解析 API 不可用。');
      repaint();
      return;
    }
    srcType = 'pdf';
    pdfIngesting = true;
    err = null;
    repaint();
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      const contentBase64 = text.includes(',') ? text.split(',').pop() : text;
      window.EU_API.ingestIdeaPdf({ filename: name, content_base64: contentBase64 })
        .then(data => {
          pdfInfo = data && data.pdf ? data.pdf : null;
          if (pdfInfo) {
            draft.source_file_name = pdfInfo.filename || name;
            draft.source_file_sha256 = pdfInfo.sha256 || '';
          }
          applySuggestedPayload(data && data.suggested_payload);
          sourceResolved = data;
        })
        .catch(e => { err = e.message || String(e); })
        .finally(() => { pdfIngesting = false; repaint(); });
    };
    reader.onerror = () => {
      pdfIngesting = false;
      err = t('Could not read the selected PDF file.', '无法读取选中的 PDF 文件。');
      repaint();
    };
    reader.readAsDataURL(file);
  }
  function scanLiteratureFolder() {
    if (literatureScanning) return;
    const payload = collectPayload(document);
    const path = String(payload.literature_folder || '').trim();
    if (!path) {
      err = t('Choose or paste a local literature folder first.', '请先选择或粘贴本地文献库文件夹。');
      repaint();
      return;
    }
    if (!(window.EU_API && window.EU_API.scanIdeaLiteratureFolder)) {
      err = t('Literature-folder scan API is unavailable.', '文献库文件夹扫描 API 不可用。');
      repaint();
      return;
    }
    srcType = 'literature_folder';
    literatureScanning = true;
    err = null;
    repaint();
    window.EU_API.scanIdeaLiteratureFolder({ path })
      .then(data => {
        literatureScan = data;
        const folder = data && data.folder ? data.folder : {};
        if (folder.path) draft.literature_folder = folder.path;
        draft.literature_pdf_count = Number(folder.pdf_count || 0);
        applySuggestedPayload(data && data.suggested_payload);
        sourceResolved = data;
      })
      .catch(e => { err = e.message || String(e); })
      .finally(() => { literatureScanning = false; repaint(); });
  }
  function useDiscoveryCandidate(index) {
    const rows = discovery && Array.isArray(discovery.idea_candidates) ? discovery.idea_candidates : [];
    const row = rows[Number(index || 0)];
    if (!row) return;
    draft = Object.assign({}, draft, Object.fromEntries(Object.entries(row.suggested_payload || {}).filter(([, v]) => v != null && v !== '')));
    sourceResolved = {
      ok: true,
      mode: 'frontier_discovery_candidate',
      resolved_source: row.source || null,
      suggested_payload: row.suggested_payload || {},
      source_adapter: {
        status: 'pubmed_candidate_selected',
        network_calls: discovery.network_calls || 0,
        external_llm_calls: 0,
      },
    };
    err = null;
    repaint();
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
    const steps = [
      ['source', t('Source', '来源'), source ? esc(source.title || source.journal || 'bound') : t('waiting for input', '等待输入'), source ? 'check' : 'file'],
      ['ledger', t('Idea ledger', 'Idea 台账'), idea ? esc(idea.go_no_go || 'draft') : t('not mined', '尚未挖掘'), idea ? 'target' : 'clock'],
      ['evidence', t('Feasibility', '可行性评估'), pre ? esc(pre.status || 'checked') : t('after mining', '挖掘后生成'), pre ? 'shield' : 'shield'],
      ['handoff', t('Plan / replan', '计划 / replan'), project ? t('project seed ready', '项目种子已生成') : handoff ? t('frozen for Agent', '已冻结给 Agent') : planDraft ? t('plan draft ready', '计划草案已生成') : t('plan required', '需要计划'), project ? 'agent' : handoff || planDraft ? 'check' : 'arrow'],
    ];
    return `<div class="ideas-summary-strip">${steps.map(row => {
      const state = stepState(row[0]);
      const cls = ['ideas-summary-item', state === 'ready' ? 'ready' : 'idle', activeStep === row[0] ? 'active' : ''].filter(Boolean).join(' ');
      return `<button type="button" class="${cls}" data-idea-step="${row[0]}" ${state === 'locked' ? 'aria-disabled="true"' : ''}>
        <span>${icon(state === 'ready' && row[0] !== activeStep ? 'check' : row[3], 13)}</span>
        <div><b>${row[1]}</b><small>${row[2]}</small></div>
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
  function discoveryPanel() {
    if (srcType !== 'frontier' && !discovery) return '';
    const candidates = discovery && Array.isArray(discovery.idea_candidates) ? discovery.idea_candidates.slice(0, 6) : [];
    const queries = discovery && Array.isArray(discovery.queries_to_run) ? discovery.queries_to_run.slice(0, 4) : [];
    const connectorOn = pubmedConnectorEnabled();
    const allowed = networkAllowed();
    return `
      <div class="ideas-discovery mt-12">
        <div class="ideas-prior-top">
          <div>
            <h3>${t('Frontier literature discovery', '前沿文献发现')}</h3>
            <p>${esc((discovery && discovery.reason) || t('Searches PubMed metadata/abstracts only after explicit network opt-in; then maps each article into an EasyICU idea candidate.', '只有明确网络 opt-in 后才检索 PubMed 元数据/摘要；随后把每篇文章映射成 EasyICU idea 候选。'))}</p>
          </div>
          <button class="btn ${allowed ? 'primary' : ''}" data-idea-discover ${discovering || !connectorOn ? 'aria-disabled="true"' : ''}>${discovering ? '<span class="spin"></span>' : icon('search', 13)} ${t('Discover papers', '检索文章')}</button>
        </div>
        ${!connectorOn ? `<div class="ideas-prior-gate blocked mt-10"><div>${icon('shield', 13)}</div><div><b>${t('PubMed connector is off', 'PubMed 连接器已关闭')}</b><span>${t('Turn on the PubMed connector in Settings before running frontier discovery.', '运行前沿发现前，请先在 Settings 打开 PubMed 连接器。')}</span></div><button class="btn sm" type="button" data-idea-open-settings>${t('Open Settings', '打开 Settings')}</button></div>` : ''}
        ${queries.length ? `<div class="ideas-query-list">${queries.map(q => `<code>${esc(q)}</code>`).join('')}</div>` : ''}
        ${candidates.length ? `<div class="ideas-feature-list mt-10">${candidates.map((row, i) => {
          const idea = row.idea || {};
          const source = row.source || {};
          const feas = idea.feasibility || {};
          return `<div class="ideas-feature-row">
            <div class="ideas-feature-name"><b>${esc(idea.idea_title || source.title || 'Candidate idea')}</b><span class="mono">${esc([source.year, source.journal, source.pmid ? 'PMID ' + source.pmid : ''].filter(Boolean).join(' · '))}</span></div>
            <span class="pill ${feas.tier === 'executable' ? 'ok' : 'warn'}">${esc(feas.tier || idea.go_no_go || 'review')}</span>
            <button class="btn sm" data-idea-use-discovery="${i}">${t('Use', '使用')}</button>
          </div>`;
        }).join('')}</div>` : ''}
      </div>`;
  }
  function optInBlock() {
    const connectorOn = pubmedConnectorEnabled();
    const checked = connectorOn && sourceNetworkOptIn();
    return `
      <details class="ideas-advanced mt-10">
        <summary>${icon('shield', 13)} ${t('Network and provider opt-in', '网络与模型 opt-in')} <span>${connectorOn ? t('source opt-in required', '需要来源 opt-in') : t('PubMed connector off', 'PubMed 连接器关闭')}</span></summary>
        <label class="rtodo-row mt-10 ideas-network-row">
          <input type="checkbox" id="ideaNetworkOptIn" ${checked ? 'checked' : ''} ${connectorOn ? '' : 'disabled'} />
          <span class="rtodo-t">${t('Allow one bounded network metadata/prior-art request for this source', '允许本来源进行一次有界网络元数据 / prior-art 请求')}</span>
          <span class="rtodo-ref mono">opt-in</span>
        </label>
        <div class="muted mt-8">${connectorOn
          ? t('URL/DOI metadata and PubMed prior-art checks stay blocked until this source-level opt-in is checked. Provider calls still require provider readiness.', 'URL/DOI 元数据和 PubMed prior-art 检查在勾选本来源 opt-in 前保持阻断。Provider 调用仍需要 provider readiness。')
          : t('The PubMed connector is disabled in Settings, so this source cannot make a network metadata request.', 'Settings 中 PubMed 连接器已关闭，因此本来源不能发起网络元数据请求。')}</div>
        ${!connectorOn ? `<button class="btn sm mt-10" type="button" data-idea-open-settings>${t('Open Settings', '打开 Settings')}</button>` : ''}
      </details>`;
  }
  function pdfPickerBlock() {
    const label = pdfIngesting
      ? t('Reading selected PDF...', '正在读取选中的 PDF...')
      : pdfInfo
        ? `${pdfInfo.filename || 'PDF'} · ${fmt(pdfInfo.page_count)} ${t('pages', '页')} · ${String(pdfInfo.sha256 || '').slice(0, 8)}...`
        : t('Choose a local PDF. Only bounded metadata/excerpt plus SHA-256 are retained.', '选择本地 PDF。只保留有界元数据/摘录和 SHA-256。');
    return `
      <div class="ideas-source-picker">
        <input type="file" accept="application/pdf,.pdf" id="ideaPdfFile" hidden />
        <div>
          <b>${esc(label)}</b>
          <span>${t('The full PDF is read locally by the browser and is not stored by this workbench.', 'PDF 全文由浏览器在本机读取，本工作台不保存全文。')}</span>
        </div>
        <button class="btn sm" type="button" data-idea-pdf-pick>${icon('folder', 13)} ${pdfInfo ? t('Choose another PDF', '换一个 PDF') : t('Choose PDF', '选择 PDF')}</button>
      </div>`;
  }
  function literatureFolderBlock() {
    const folder = literatureScan && literatureScan.folder;
    const docs = literatureScan && Array.isArray(literatureScan.documents) ? literatureScan.documents.slice(0, 4) : [];
    return `
      <div class="ideas-folder-source">
        <label class="field ideas-field"><span>${t('Local literature folder', '本地文献库文件夹')}</span><input id="ideaLiteratureFolder" placeholder="/Users/.../papers" value="${esc(fieldValue('literature_folder'))}" /></label>
        <button class="btn ${literatureScanning ? '' : 'primary'}" type="button" data-idea-lit-scan>${literatureScanning ? '<span class="spin"></span>' : icon('search', 13)} ${literatureScanning ? t('Scanning PDFs...', '正在扫描 PDF...') : t('Scan folder', '扫描文件夹')}</button>
      </div>
      ${folder ? `<div class="note ok mt-10"><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${fmt(folder.pdf_count)} ${t('PDFs found', '个 PDF 已发现')}</div><div class="d">${esc(folder.path || fieldValue('literature_folder'))}</div></div></div>` : ''}
      ${docs.length ? `<div class="ideas-feature-list mt-10">${docs.map(doc => `<div class="ideas-feature-row"><div class="ideas-feature-name"><b>${esc(doc.title || doc.filename || 'PDF')}</b><span class="mono">${esc(doc.filename || doc.path || '')}</span></div><span class="mono">${esc(String(doc.sha256 || '').slice(0, 8))}</span></div>`).join('')}</div>` : ''}`;
  }
  function zoteroSearchBlock() {
    const widget = ideaZoteroWidget();
    if (widget) return widget.render();
    return `<div class="note warn mt-10"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${t('Zotero source widget unavailable', 'Zotero 来源组件不可用')}</div><div class="d">${t('Enter title, DOI, and abstract manually below.', '请在下方手动填写标题、DOI 和摘要。')}</div></div></div>`;
  }
  function sourceSpecificForm() {
    if (srcType === 'manual') return `
      <div class="ideas-source-form manual mt-14">
        ${sourceModeGuide()}
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="5" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        ${optionalMetadataBlock(`<label class="field ideas-field"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="3" placeholder="${t('Optional: paste the sentence, clinical observation, or rationale that triggered the idea.', '可选：粘贴触发这个想法的句子、临床观察或理由。')}">${esc(fieldValue('excerpt'))}</textarea></label>`, t('Optional rationale', '可选触发理由'))}
      </div>`;
    if (srcType === 'url') return `
      <div class="ideas-source-form url mt-14">
        ${sourceModeGuide()}
        <label class="field ideas-field"><span>${t('Article URL', '文章链接')}</span><input id="ideaUrl" placeholder="https://www.nejm.org/doi/full/..." value="${esc(fieldValue('url'))}" /></label>
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="3" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        <label class="field ideas-field"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="3" placeholder="${t('Optional: paste the article sentence that should become an ICU-database question.', '可选：粘贴应转化为 ICU 数据库问题的文章句子。')}">${esc(fieldValue('excerpt'))}</textarea></label>
        ${optionalMetadataBlock(`<div class="ideas-meta-grid">
          <label class="field ideas-field"><span>DOI / PMID</span><input id="ideaDoi" placeholder="10.xxxx or PMID" value="${esc(fieldValue('doi'))}" /></label>
          <label class="field ideas-field"><span>Title</span><input id="ideaTitle" placeholder="${t('Resolved or manually entered title', '解析或手动输入的标题')}" value="${esc(fieldValue('title'))}" /></label>
          <label class="field ideas-field"><span>Journal</span><input id="ideaJournal" placeholder="e.g. NEJM" value="${esc(fieldValue('journal'))}" /></label>
          <label class="field ideas-field ideas-year"><span>Year</span><input id="ideaYear" placeholder="2026" value="${esc(fieldValue('year'))}" /></label>
        </div>`, t('Optional article metadata', '可选文章元数据'))}
        ${optInBlock()}
      </div>`;
    if (srcType === 'pdf') return `
      <div class="ideas-source-form pdf mt-14">
        ${sourceModeGuide()}
        ${pdfPickerBlock()}
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="3" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        <label class="field ideas-field"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="4" placeholder="${t('Optional: paste a bounded passage if the PDF parser did not extract the motivating sentence.', '可选：如果 PDF 解析没有抽到触发句，可以粘贴一段有界摘录。')}">${esc(fieldValue('excerpt'))}</textarea></label>
        ${optionalMetadataBlock(`<div class="ideas-meta-grid two">
          <label class="field ideas-field"><span>Title</span><input id="ideaTitle" placeholder="${t('Auto-filled from PDF when available', '可由 PDF 自动填充')}" value="${esc(fieldValue('title'))}" /></label>
          <label class="field ideas-field"><span>DOI / PMID</span><input id="ideaDoi" placeholder="10.xxxx or PMID" value="${esc(fieldValue('doi'))}" /></label>
        </div>`, t('Optional PDF metadata', '可选 PDF 元数据'))}
      </div>`;
    if (srcType === 'literature_folder') return `
      <div class="ideas-source-form literature_folder mt-14">
        ${sourceModeGuide()}
        ${literatureFolderBlock()}
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="3" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        ${optionalMetadataBlock(`<label class="field ideas-field"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="3" placeholder="${t('Optional notes about what this literature folder should help explore.', '可选：说明这批文献主要想辅助探索什么。')}">${esc(fieldValue('excerpt'))}</textarea></label>`, t('Optional library notes', '可选文献库备注'))}
      </div>`;
    if (srcType === 'zotero') return `
      <div class="ideas-source-form zotero mt-14">
        ${sourceModeGuide()}
        ${zoteroSearchBlock()}
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="3" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        <label class="field ideas-field"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="4" placeholder="${t('Optional: keep the abstract excerpt short and relevant to the ICU question.', '可选：保留一段与 ICU 问题相关的短摘要摘录。')}">${esc(fieldValue('excerpt'))}</textarea></label>
        ${optionalMetadataBlock(`<div class="ideas-meta-grid">
          <label class="field ideas-field"><span>Title</span><input id="ideaTitle" placeholder="${t('Selected Zotero title', '选中的 Zotero 标题')}" value="${esc(fieldValue('title'))}" /></label>
          <label class="field ideas-field"><span>Journal</span><input id="ideaJournal" placeholder="${t('Journal or venue', '期刊或来源')}" value="${esc(fieldValue('journal'))}" /></label>
          <label class="field ideas-field ideas-year"><span>Year</span><input id="ideaYear" placeholder="2026" value="${esc(fieldValue('year'))}" /></label>
          <label class="field ideas-field"><span>DOI / PMID</span><input id="ideaDoi" placeholder="10.xxxx" value="${esc(fieldValue('doi'))}" /></label>
          <label class="field ideas-field"><span>Zotero key</span><input id="ideaZoteroKey" placeholder="ABC123" value="${esc(fieldValue('zotero_key'))}" /></label>
          <label class="field ideas-field"><span>Citation key</span><input id="ideaCitationKey" placeholder="smith2026..." value="${esc(fieldValue('citation_key'))}" /></label>
          <label class="field ideas-field ideas-wide"><span>URL</span><input id="ideaUrl" placeholder="https://..." value="${esc(fieldValue('url'))}" /></label>
        </div>
        <input id="ideaSourceOrigin" type="hidden" value="${esc(fieldValue('source_origin'))}" />
        <input id="ideaSourceOriginLabel" type="hidden" value="${esc(fieldValue('source_origin_label'))}" />
        <textarea id="ideaAbstract" hidden>${esc(fieldValue('abstract'))}</textarea>`, t('Zotero metadata', 'Zotero 元数据'))}
      </div>`;
    return `
      <div class="ideas-source-form frontier mt-14">
        ${sourceModeGuide()}
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="4" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        ${optionalMetadataBlock(`<div class="ideas-meta-grid two">
          <label class="field ideas-field"><span>${t('Journal scope', '期刊范围')}</span><input id="ideaJournal" placeholder="NEJM, JAMA, ICM..." value="${esc(fieldValue('journal'))}" /></label>
          <label class="field ideas-field ideas-year"><span>${t('Year window', '年份窗口')}</span><input id="ideaYear" placeholder="2024-2026" value="${esc(fieldValue('year'))}" /></label>
        </div>
        <label class="field ideas-field mt-10"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="3" placeholder="${t('Optional: describe why this topic matters for ICU discovery.', '可选：说明这个主题为什么适合 ICU 数据库探索。')}">${esc(fieldValue('excerpt'))}</textarea></label>`, t('Optional search scope', '可选检索范围'))}
        ${optInBlock()}
        ${discoveryPanel()}
      </div>`;
  }
  function sourceResolvedNote() {
    if (!sourceResolved) return '';
    const adapter = sourceResolved.source_adapter || {};
    const rawStatus = String(adapter.status || '');
    const blocked = sourceResolved.blocked === true || rawStatus.includes('blocked');
    const title = adapter.display_status || adapter.label || (!rawStatus.includes('_') && rawStatus ? rawStatus : t('Source ready', '来源已就绪'));
    const body = adapter.display_reason || adapter.reason || t('Bounded source metadata is ready for the idea ledger.', '有界来源元数据已就绪，可进入 idea 台账。');
    return `<div class="note ${blocked ? 'warn' : 'ok'} mt-12"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${esc(title)}</div><div class="d">${esc(body)}</div></div></div>`;
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
        ${sourceSpecificForm()}
        ${!result ? `<div class="ideas-zero-line mt-12">${icon('clock', 13)} <span>${t('No idea ledger yet', '还没有 idea 台账')}</span><b>${t('Next: create one local, auditable record.', '下一步：生成一条本地、可审计记录。')}</b></div>` : ''}
        <div class="row gap-8 mt-16 ideas-actions">
          <button class="btn" data-idea-resolve ${resolving ? 'aria-disabled="true"' : ''}>${resolving ? '<span class="spin"></span>' : icon('search', 14)} ${t('Resolve source', '解析来源')}</button>
          <button class="btn primary" data-idea-mine ${mining ? 'aria-disabled="true"' : ''}>${mining ? '<span class="spin"></span>' : icon('play', 14)} ${t('Create idea ledger', '生成 idea 台账')}</button>
        </div>
        ${sourceResolvedNote()}
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
                  <h4>${t('Literature inspiration', '已有研究与启发')}</h4>
                  <span class="pill warn">${esc(prior.status || 'not checked')}</span>
                  <p>${esc(prior.opportunity_frame || prior.reason || t('Prior-art search has not been run; use the source as inspiration, not as a completed novelty claim.', '尚未运行已有研究检索；先把来源当作启发，而不是已经完成的新颖性判断。'))}</p>
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
          <button class="btn" data-idea-step="handoff">${icon('agent', 13)} ${t('Plan / replan', '计划 / replan')}</button>
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
    const connectorOn = pubmedConnectorEnabled();
    const allowed = networkAllowed();
    const gateTitle = !connectorOn
      ? t('PubMed connector is off', 'PubMed 连接器已关闭')
      : allowed
        ? t('Ready for one bounded search', '可运行一次有界检索')
        : t('Source opt-in required', '需要本来源 opt-in');
    const gateBody = !connectorOn
      ? t('Turn on the PubMed connector in Settings first. The backend will keep network calls blocked while it is off.', '请先在 Settings 打开 PubMed 连接器；关闭期间后端会继续阻断网络请求。')
      : allowed
        ? t('This will query PubMed metadata for this idea only. It will not send patient rows or source full text.', '只会为这个 idea 查询 PubMed 元数据，不会发送患者行或来源全文。')
        : t('Turn on the checkbox below before checking prior art. The default state keeps network calls blocked.', '先勾选下方选项，再检查已有文献。默认状态会阻断网络调用。');
    return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('search', 14)}</span>
          <div><h2>${t('Prior-art and inspiration map', '已有文献与启发地图')}</h2><p>${t('Check what has already been published, then use it to refine the new ICU-database question.', '先看别人怎么做过，再据此细化新的 ICU 数据库探索问题。')}</p></div>
        </div>
        <div class="ideas-prior-card mt-10">
          <div class="ideas-prior-top">
            <div>
              <span class="pill ${statusTone}">${esc(status)}</span>
              <p>${esc((prior && (prior.opportunity_frame || prior.reason)) || t('No prior-art request has been made yet. The source article can still inspire a new subgroup, timing window, comparator, or outcome question.', '尚未发起已有研究检索。来源文章仍可启发新的亚组、时间窗、比较方式或结局问题。'))}</p>
            </div>
            <button class="btn ${prior && prior.search_performed ? '' : 'primary'}" data-idea-prior-art ${priorArting || !connectorOn ? 'aria-disabled="true"' : ''}>${priorArting ? '<span class="spin"></span>' : icon('search', 13)} ${t('Check literature', '检查已有文献')}</button>
          </div>
          <div class="ideas-prior-gate ${allowed ? 'ready' : 'blocked'}">
            <div>${icon(allowed ? 'check' : 'shield', 13)}</div>
            <div><b>${gateTitle}</b><span>${gateBody}</span></div>
            ${!connectorOn ? `<button class="btn sm" type="button" data-idea-open-settings>${t('Open Settings', '打开 Settings')}</button>` : ''}
          </div>
          <label class="rtodo-row ideas-network-row">
            <input type="checkbox" id="ideaNetworkOptIn" ${allowed ? 'checked' : ''} ${connectorOn ? '' : 'disabled'} />
            <span class="rtodo-t">${t('Allow one bounded PubMed metadata search for this idea', '允许为这个 idea 进行一次有界 PubMed 元数据检索')}</span>
            <span class="rtodo-ref mono">opt-in</span>
          </label>
          ${queries.length ? `<details class="ideas-query-details">
            <summary>${icon('search', 13)} ${t('Suggested search queries', '建议检索式')} <span>${queries.length}</span></summary>
            <div class="ideas-query-list">${queries.map(q => `<code>${esc(q)}</code>`).join('')}</div>
          </details>` : ''}
        </div>
        ${(prior && prior.next_use) ? `<div class="ideas-interpretation mt-10"><div>${icon('spark', 13)} <span>${esc(prior.next_use)}</span></div></div>` : ''}
        ${rows.length ? `<div class="ideas-prior-results mt-12">${rows.map(r => `<article><b>${esc(r.title || '')}</b><span>${esc(r.journal || '')} · ${fmt(r.year)} · PMID ${esc(r.pmid || '')}</span></article>`).join('')}</div>` : ''}
      </div>`;
  }
  function handoffReceipt() {
    const handoff = window.EU_IDEA_HANDOFF;
    if (!handoff && !projectSeed) return '';
    const plan = (handoff && handoff.handoff_plan) || {};
    const steps = Array.isArray(plan.analysis_plan) ? plan.analysis_plan.length : 0;
    const studyId = projectSeed && projectSeed.study_id ? projectSeed.study_id : '';
    const projectDir = projectSeed && projectSeed.project_dir ? projectSeed.project_dir : '';
    const question = (projectSeed && (projectSeed.question || projectSeed.title)) || plan.research_question || (handoff && handoff.candidate_topic) || t('Idea-derived study', '由 idea 生成的研究');
    const ready = !!projectSeed;
    const title = ready ? t('Agent project ready', '研究项目已创建') : t('Handoff frozen', '交接已冻结');
    const body = ready
      ? t('Open Agent Projects to continue from this seed. It includes the locked plan, feasibility context, and evidence boundaries.', '可以到研究项目继续推进。该种子包含已锁定计划、可行性上下文和证据边界。')
      : t('The plan is frozen as a metadata-only handoff. Create a project seed when you are ready to run the study workflow.', '计划已冻结为仅元数据交接。准备运行研究流程时，再创建项目种子。');
    return `
      <div class="ideas-handoff-receipt ${ready ? 'ready' : 'frozen'} mt-12">
        <div class="ideas-handoff-main">
          <div class="ideas-handoff-icon">${icon(ready ? 'agent' : 'check', 16)}</div>
          <div>
            <div class="ideas-handoff-kicker">${title}</div>
            <h3>${esc(question)}</h3>
            <p>${body}</p>
          </div>
        </div>
        <div class="ideas-handoff-grid">
          <div><span>${t('Study ID', '研究 ID')}</span><b class="mono">${esc(studyId || (handoff && handoff.run_id) || '—')}</b></div>
          <div><span>${t('Plan steps', '计划步骤')}</span><b>${fmt(steps)}</b></div>
          <div><span>${t('Reportable', '可作为结果报告')}</span><b>${t('No', '否')}</b></div>
          <div><span>${t('Draft access', '论文草稿')}</span><b>${t('Locked', '未解锁')}</b></div>
        </div>
        ${projectDir ? `<div class="ideas-handoff-path"><span>${t('Project folder', '项目文件夹')}</span><code>${esc(projectDir)}</code></div>` : ''}
        <div class="ideas-handoff-actions">
          ${ready ? `<button class="btn primary" data-nav="agent">${icon('agent', 13)} ${t('Open Agent Projects', '打开研究项目')}</button>` : `<button class="btn primary" data-idea-create-project ${projectCreating ? 'aria-disabled="true"' : ''}>${projectCreating ? '<span class="spin"></span>' : icon('agent', 13)} ${t('Create Agent project', '创建研究项目')}</button>`}
          <button class="btn" data-nav="agent">${icon('agent', 13)} ${t('View project list', '查看项目列表')}</button>
        </div>
      </div>`;
  }
  function preExperiment() {
    const pre = result && result.pre_experiment;
    if (!pre) return '';
    const stats = pre.feature_statistics || [];
    const sampleStats = sampleFeasibility && Array.isArray(sampleFeasibility.feature_statistics) ? sampleFeasibility.feature_statistics : [];
    const visibleStats = stats.slice(0, 4);
    const hiddenStats = stats.slice(4);
    const isEventMetric = (s) => s && s.metric_kind === 'event_rate';
    const isSchemaMetric = (s) => s && (s.metric_kind === 'schema_presence' || s.status === 'metadata_only' || s.coverage_basis === 'manifest_file_inventory');
    // Backend ships coverage_pct=null + denominator_resolved=false when the
    // cohort denominator could not be resolved — that is indeterminate, NOT
    // 0% coverage, so it must not paint a red risk row or inflate riskCount.
    const isIndeterminate = (s) => s && !isEventMetric(s) && !isSchemaMetric(s)
      && (s.denominator_resolved === false || s.coverage_pct == null);
    const riskCount = stats.filter(s => !isEventMetric(s) && !isSchemaMetric(s) && !isIndeterminate(s) && (s.low_coverage || pct(s.coverage_pct) < 50)).length;
    const featureRow = (s) => {
      const n = s.numeric_summary || {};
      const eventMetric = isEventMetric(s);
      const schemaMetric = isSchemaMetric(s);
      const indeterminate = isIndeterminate(s);
      const metricPct = schemaMetric ? 100 : (eventMetric ? pct(s.event_rate_pct) : (indeterminate ? 0 : pct(s.coverage_pct)));
      const tone = schemaMetric ? 'warn' : (indeterminate ? 'warn' : (eventMetric ? 'event' : coverageTone(metricPct)));
      const summary = schemaMetric
        ? t('Schema present; run a bounded sample check before interpreting coverage.', '结构存在；解释覆盖率前先运行有界样本检查。')
        : indeterminate
        ? t('Denominator unresolved — coverage is indeterminate, not 0%. Run a bounded sample check to measure it.', '分母未确定 —— 覆盖率不确定，而非 0%。请运行有界样本检查来测量。')
        : eventMetric
        ? t('binary/event indicator; non-events are not missing', '二分类/事件指标；阴性患者不是缺失')
        : (n.available ? `median ${fmt(n.median)} · min ${fmt(n.min)} · max ${fmt(n.max)}` : t('categorical, non-numeric, or empty', '分类、非数值或为空'));
      const meta = schemaMetric
        ? `<span>${t('Declared records', '声明记录')} ${fmt(s.records_declared)}</span><span>${esc(s.coverage_basis || 'schema')}</span>`
        : eventMetric
        ? `<span>${t('Events', '事件')} ${fmt(s.event_entities ?? s.records)}</span><span>${t('Non-events', '非事件')} ${fmt(s.non_event_entities)}</span>`
        : `<span>${t('Records', '记录')} ${fmt(s.records)}</span><span>${t('Missing', '缺失')} ${pctLabel(s.missing_pct)}</span>`;
      const headLabel = schemaMetric ? t('Schema', '结构') : (eventMetric ? t('Event rate', '事件率') : t('Coverage', '覆盖'));
      const headValue = schemaMetric ? t('present', '存在') : (indeterminate ? t('indeterminate', '不确定') : pctLabel(eventMetric ? s.event_rate_pct : s.coverage_pct));
      return `<div class="ideas-feature-row ${tone}">
        <div class="ideas-feature-name">
          <b>${esc(s.label)}</b>
          <span class="mono">${esc(s.module || '')} · ${esc(s.concept_id || '')}</span>
        </div>
        <div class="ideas-feature-cov">
          <div class="ideas-cov-head"><span>${headLabel}</span><b>${headValue}</b></div>
          <div class="ideas-cov-bar"><i style="width:${metricPct}%"></i></div>
        </div>
        <div class="ideas-feature-meta">
          ${meta}
        </div>
        <div class="ideas-feature-summary">${esc(summary)}</div>
      </div>`;
    };
    const noteText = (x) => {
      const raw = String(x || '');
      if (raw.includes('manifest/schema only')) return t(raw, '有特征目前只通过 manifest / schema 验证；解释覆盖率前请运行有界样本检查或研究项目。');
      if (raw.includes('outcome-blind bounded sample check')) return t(raw, '这是不看结局效应的有界样本检查，可用于判断是否值得继续，但不是论文 source data。');
      if (raw.includes('Low sample coverage remains')) return t(raw, '样本覆盖仍偏低；进入 Agent 执行前需要确认分母和缺失结构。');
      if (raw.includes('Required concepts were sample-checked')) return t(raw, '必要概念已完成样本检查，且没有返回原始记录或直接标识符。');
      return raw;
    };
    return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('shield', 14)}</span>
          <div><h2>${t('Feasibility assessment on active export', '基于当前导出的可行性评估')}</h2><p>${t('Outcome-blind feasibility check from the active export.', '基于当前导出的 outcome-blind 可行性检查。')} ${esc(activeSourceLine())}</p></div>
        </div>
        <div class="ideas-pre-summary mt-10">
          <div><span>${t('Status', '状态')}</span><b>${esc(pre.status || '—')}</b></div>
          <div><span>${t('Entities', '实体')}</span><b>${fmt(pre.cohort && pre.cohort.entities)}</b></div>
          <div><span>${t('Modules', '模块')}</span><b>${fmt(pre.cohort && pre.cohort.modules)}</b></div>
          <div><span>${t('Low coverage', '低覆盖')}</span><b>${fmt(riskCount)}</b></div>
        </div>
        ${stats.length ? `<div class="ideas-feature-list mt-12">${visibleStats.map(featureRow).join('')}</div>
          ${hiddenStats.length ? `<details class="ideas-compact-details mt-10"><summary>${icon('list', 13)} ${t('Show all feature checks', '查看全部特征检查')} <span>${stats.length}</span></summary><div class="ideas-feature-list mt-10">${stats.map(featureRow).join('')}</div></details>` : ''}`
        : `<div class="note warn mt-12"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="t">${esc(pre.reason || 'No feature statistics available')}</div></div></div>`}
        ${pre.interpretation && pre.interpretation.length ? `<div class="ideas-interpretation mt-10">${pre.interpretation.slice(0, 2).map(x => `<div>${icon('shield', 13)} <span>${esc(noteText(x))}</span></div>`).join('')}</div>` : ''}
        <div class="ideas-inline-actions mt-12">
          <button class="btn" data-idea-sample-feasibility ${sampleChecking ? 'aria-disabled="true"' : ''}>${sampleChecking ? '<span class="spin"></span>' : icon('search', 13)} ${t('Run bounded sample check', '运行有界样本检查')}</button>
          ${sampleFeasibility ? `<span class="mono">${esc(sampleFeasibility.status || 'checked')} · ${fmt((sampleFeasibility.sample || {}).max_records_per_feature)} ${t('records / feature', '条/特征')}</span>` : ''}
        </div>
        ${sampleFeasibility ? `<details class="ideas-compact-details mt-10" open><summary>${icon('shield', 13)} ${t('Bounded sample result', '有界样本结果')} <span>${sampleStats.length}</span></summary>${sampleFeasibility.interpretation && sampleFeasibility.interpretation.length ? `<div class="ideas-interpretation mt-10">${sampleFeasibility.interpretation.slice(0, 2).map(x => `<div>${icon('shield', 13)} <span>${esc(noteText(x))}</span></div>`).join('')}</div>` : ''}<div class="ideas-feature-list mt-10">${sampleStats.slice(0, 4).map(featureRow).join('')}</div></details>` : ''}
      </div>`;
  }
  function planBuilder() {
    const plan = (planDraft && planDraft.plan) || (window.EU_IDEA_HANDOFF && window.EU_IDEA_HANDOFF.handoff_plan);
    if (!result) return '';
    if (!plan) {
      return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('agent', 14)}</span>
          <div><h2>${t('Plan / replan before Agent', 'Agent 前计划 / replan')}</h2><p>${t('Generate a concrete study plan from the idea ledger, feasibility assessment, and literature-inspiration map before freezing an Agent handoff.', '先根据 idea 台账、可行性评估和已有文献启发地图生成具体研究计划，然后再冻结交接给 Agent。')}</p></div>
        </div>
        <div class="note warn mt-8"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${t('Plan required before handoff', '交接前需要计划')}</div><div class="d">${t('This step collects the clinical question, feasibility risks, cohort/module confirmations, analysis family, reference method motifs, and reporting boundary. It is not an Agent run.', '这一步收集临床问题、可行性风险、队列/模块确认、分析类型、参考方法套路和报告边界。它不是 Agent run。')}</div></div></div>
        <div class="row gap-8 mt-12">
          <button class="btn primary" data-idea-plan ${planning ? 'aria-disabled="true"' : ''}>${planning ? '<span class="spin"></span>' : icon('agent', 14)} ${t('Generate study plan', '生成研究计划')}</button>
        </div>
      </div>`;
    }
    const patterns = Array.isArray(plan.reference_analysis_patterns) ? plan.reference_analysis_patterns : [];
    const constraints = Array.isArray(plan.clinical_icu_constraints) ? plan.clinical_icu_constraints : [];
    const confirmations = Array.isArray(plan.required_user_confirmations) ? plan.required_user_confirmations : [];
    const miniList = (rows) => rows.length ? `<div class="ideas-feature-list mt-10">${rows.map(row => {
      const label = typeof row === 'object' ? (row.pattern || row.title || '') : String(row || '');
      const body = typeof row === 'object' ? [row.use_for, row.guardrail].filter(Boolean).join(' · ') : '';
      return `<div class="ideas-feature-row"><div class="ideas-feature-name"><b>${esc(label)}</b>${body ? `<span class="mono">${esc(body)}</span>` : ''}</div></div>`;
    }).join('')}</div>` : '';
    const planSteps = Array.isArray(plan.analysis_plan) ? plan.analysis_plan : [];
    const planStep = (row, i) => {
      const obj = normalizePlanStep(row, i);
      return `<article class="ideas-plan-step">
        <div class="ideas-plan-num">${String(i + 1).padStart(2, '0')}</div>
        <div class="ideas-plan-copy">
          <div class="ideas-plan-phase">${esc(obj.phase || t('Step', '步骤'))}</div>
          <h3>${esc(obj.title || obj.action || t('Plan step', '计划步骤'))}</h3>
          ${obj.action ? `<p>${esc(obj.action)}</p>` : ''}
          <div class="ideas-plan-meta">
            ${obj.output ? `<span><b>${t('Output', '产物')}</b>${esc(obj.output)}</span>` : ''}
            ${obj.guardrail ? `<span><b>${t('Guardrail', '约束')}</b>${esc(obj.guardrail)}</span>` : ''}
          </div>
        </div>
      </article>`;
    };
    return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('agent', 14)}</span>
          <div><h2>${t('Plan / replan before Agent', 'Agent 前计划 / replan')}</h2><p>${t('Confirm or revise the plan before sending it to Agent Projects. This does not unlock a manuscript draft.', '交给 Agent Projects 前先确认或修订计划。这里不会解锁论文草稿。')}</p></div>
        </div>
        <div class="note ok mt-8"><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${esc(plan.research_question || '')}</div><div class="d">${esc((plan.agent_boundary && plan.agent_boundary.reason) || t('Draft analysis plan is locked until human confirmation and evidence checks pass.', '分析计划草稿在人工确认和证据核验通过前保持锁定。'))}</div></div></div>
        <div class="ideas-plan-steps mt-12">${planSteps.map(planStep).join('')}</div>
        ${patterns.length ? `<details class="ideas-compact-details mt-10" open><summary>${icon('book', 13)} ${t('Reference method patterns', '参考方法套路')} <span>${patterns.length}</span></summary>${miniList(patterns)}</details>` : ''}
        ${constraints.length ? `<details class="ideas-compact-details mt-10"><summary>${icon('shield', 13)} ${t('ICU constraints', 'ICU 场景约束')} <span>${constraints.length}</span></summary>${miniList(constraints)}</details>` : ''}
        ${confirmations.length ? `<div class="ideas-interpretation mt-10"><div>${icon('shield', 13)} <span>${t('Still needs confirmation', '仍需确认')}: ${confirmations.map(esc).join(' · ')}</span></div></div>` : ''}
        <label class="field ideas-plan-edits mt-12"><span>${t('Natural-language plan edits', '用自然语言微调计划')}</span><textarea id="ideaPlanEdits" rows="4" placeholder="${t('e.g. use AKI as the endpoint, restrict to first ICU stay, add missingness sensitivity...', '例如:把 AKI 作为结局,限制首次 ICU 入住,增加缺失敏感性分析...')}">${esc(planEdits)}</textarea></label>
        <div class="row gap-8 mt-12">
          <button class="btn" data-idea-replan ${planning ? 'aria-disabled="true"' : ''}>${planning ? '<span class="spin"></span>' : icon('refresh', 14)} ${t('Replan from notes', '根据说明重规划')}</button>
          <button class="btn primary" data-idea-handoff ${handoffing ? 'aria-disabled="true"' : ''}>${handoffing ? '<span class="spin"></span>' : icon('arrow', 14)} ${t('Freeze handoff for Agent', '冻结并交给 Agent')}</button>
        </div>
        ${handoffReceipt()}
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
      return warning + preExperiment() + priorArtPanel() + blockedPanel() + `<div class="ideas-inline-actions mt-14"><button class="btn primary" data-idea-step="handoff">${icon('agent', 13)} ${t('Prepare plan / replan', '准备计划 / replan')}</button><button class="btn" data-nav="dictionary">${icon('list', 13)} ${t('Open dictionary', '查看字典')}</button><button class="btn" data-nav="extraction">${icon('extract', 13)} ${t('Use active export', '使用当前导出')}</button></div>`;
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
  function ideaListContext(rows) {
    if (!rows.length) return '';
    return `
      <div class="ideas-list-context" role="note">
        <div class="ideas-list-context-k">${t('History', '历史记录')}</div>
        <div class="ideas-list-context-d">${t('These are metadata-only idea ledgers from this machine. They are reference material, not active Agent analyses.', '这里是本机保存的仅元数据 idea 台账。它们是参考材料，不是正在运行的 Agent 分析。')}</div>
      </div>`;
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
        ${ideaListContext(rows)}
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
            <button class="btn sm" data-nav="agent">${icon('agent', 13)} ${t('Open Agent Projects', '打开研究项目')}</button>
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
      draft.allow_network = false;
      repaint();
    }));
    root.querySelectorAll('#ideaNetworkOptIn').forEach(input => input.addEventListener('change', () => {
      collectPayload(document);
      repaint();
    }));
    root.querySelectorAll('[data-idea-open-settings]').forEach(btn => btn.addEventListener('click', () => {
      location.hash = '#settings';
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
      result = null; err = null; planEdits = ''; sourceResolved = null; discovery = null; priorArt = null; planDraft = null; projectSeed = null; sampleFeasibility = null; selectedRunId = null; selectedRecordKey = null; if (zoteroWidget) zoteroWidget.reset(); draft = {}; activeStep = 'source'; window.EU_IDEA_HANDOFF = null; repaint();
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
    const discoverBtn = root.querySelector('[data-idea-discover]');
    if (discoverBtn) discoverBtn.addEventListener('click', () => {
      if (discoverBtn.getAttribute('aria-disabled') === 'true') return;
      runIdeaDiscovery();
    });
    const pdfPick = root.querySelector('[data-idea-pdf-pick]');
    if (pdfPick) pdfPick.addEventListener('click', () => {
      const fileInput = root.querySelector('#ideaPdfFile');
      if (fileInput) fileInput.click();
    });
    const pdfFile = root.querySelector('#ideaPdfFile');
    if (pdfFile) pdfFile.addEventListener('change', () => {
      const file = pdfFile.files && pdfFile.files[0];
      ingestPdfFile(file);
      pdfFile.value = '';
    });
    const litScan = root.querySelector('[data-idea-lit-scan]');
    if (litScan) litScan.addEventListener('click', () => {
      scanLiteratureFolder();
    });
    const widget = ideaZoteroWidget();
    if (widget) widget.wire(root);
    root.querySelectorAll('[data-idea-use-discovery]').forEach(btn => btn.addEventListener('click', () => {
      useDiscoveryCandidate(btn.dataset.ideaUseDiscovery || 0);
    }));
    const mineBtn = root.querySelector('[data-idea-mine]');
    if (mineBtn) mineBtn.addEventListener('click', () => {
      if (mining || !(window.EU_API && window.EU_API.mineIdeas)) return;
      const payload = collectPayload(document);
      const validationError = validatePayload(payload);
      if (validationError) { err = validationError; repaint(); return; }
      mining = true; err = null; result = null; priorArt = null; projectSeed = null; sampleFeasibility = null; window.EU_IDEA_HANDOFF = null;
      repaint();
      window.EU_API.mineIdeas(payload).then(data => {
        result = data; selectedRunId = data.run_id || null; selectedRecordKey = runRecordKey(data, 'current') || selectedRunId; err = null; planEdits = ''; planDraft = null; activeStep = 'ledger'; window.EU_IDEA_LAST_RUN = data; upsertHistoryRun(data);
      }).catch(e => {
        err = e.message || String(e);
      }).finally(() => { mining = false; repaint(); });
    });
    const sampleBtn = root.querySelector('[data-idea-sample-feasibility]');
    if (sampleBtn) sampleBtn.addEventListener('click', () => {
      if (sampleChecking || !result || !(window.EU_API && window.EU_API.checkIdeaSampleFeasibility)) return;
      sampleChecking = true; err = null; repaint();
      window.EU_API.checkIdeaSampleFeasibility({
        run_id: result.run_id,
        idea_id: result.selected_idea_id,
      }).then(data => { sampleFeasibility = data; activeStep = 'evidence'; }).catch(e => { err = e.message || String(e); }).finally(() => { sampleChecking = false; repaint(); });
    });
    const planBox = root.querySelector('#ideaPlanEdits');
    if (planBox) planBox.addEventListener('input', () => {
      planEdits = planBox.value;
      window.EU_IDEA_HANDOFF = null;
      projectSeed = null;
    });
    const priorBtn = root.querySelector('[data-idea-prior-art]');
    if (priorBtn) priorBtn.addEventListener('click', () => {
      if (priorBtn.getAttribute('aria-disabled') === 'true') return;
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
    const planBtn = root.querySelector('[data-idea-plan]');
    if (planBtn) planBtn.addEventListener('click', () => {
      if (planning || !result || !(window.EU_API && window.EU_API.planIdea)) return;
      planning = true; err = null; planEdits = inputVal(document, '#ideaPlanEdits') || planEdits;
      repaint();
      window.EU_API.planIdea({
        run_id: result.run_id,
        idea_id: result.selected_idea_id,
        mode: 'plan',
        plan_edits: planEdits,
      }).then(data => {
        planDraft = data;
        window.EU_IDEA_HANDOFF = null;
        projectSeed = null;
        activeStep = 'handoff';
      }).catch(e => { err = e.message || String(e); }).finally(() => { planning = false; repaint(); });
    });
    const replanBtn = root.querySelector('[data-idea-replan]');
    if (replanBtn) replanBtn.addEventListener('click', () => {
      if (planning || !result || !(window.EU_API && window.EU_API.planIdea)) return;
      planning = true; err = null; planEdits = inputVal(document, '#ideaPlanEdits') || planEdits;
      repaint();
      window.EU_API.planIdea({
        run_id: result.run_id,
        idea_id: result.selected_idea_id,
        mode: 'replan',
        plan_edits: planEdits,
      }).then(data => {
        planDraft = data;
        window.EU_IDEA_HANDOFF = null;
        projectSeed = null;
        activeStep = 'handoff';
      }).catch(e => { err = e.message || String(e); }).finally(() => { planning = false; repaint(); });
    });
    const handoffBtn = root.querySelector('[data-idea-handoff]');
    if (handoffBtn) handoffBtn.addEventListener('click', () => {
      if (handoffing || !result || !(window.EU_API && window.EU_API.handoffIdea)) return;
      if (!planDraft && !(window.EU_IDEA_HANDOFF && window.EU_IDEA_HANDOFF.handoff_plan)) {
        err = t('Generate and review the study plan before freezing an Agent handoff.', '请先生成并审阅研究计划，再冻结交接给 Agent。');
        repaint();
        return;
      }
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
    get status() { return `<span class="pill ok"><span class="dot"></span> ${t('Local-first', '本地优先')}</span>`; },
    get actionHtml() { return `<button class="btn sm" data-nav="agent">${icon('agent', 13)} ${t('Agent Projects', '研究项目')}</button>`; },
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
      takeGuidedHandoff();
      return `
        ${guidedPrefillNote()}
        <div class="page-head" style="margin-bottom:16px;">
          <div class="row" style="justify-content:space-between;align-items:flex-start;gap:16px;">
            <div>
              <div class="eyebrow">${t('DISCOVERY · IDEA MINING · FEASIBILITY', '发现 · IDEA 挖掘 · 可行性')}</div>
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
