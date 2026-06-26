/* Screen: Idea Mining — first-class discovery workflow.
   Local-first Stage67: user-supplied metadata/excerpt -> idea ledger ->
   dictionary/export feasibility -> pre-experiment -> Agent handoff plan. */
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
      source_file_name: (pdfInfo && pdfInfo.filename) || fieldValue('source_file_name'),
      source_file_sha256: (pdfInfo && pdfInfo.sha256) || fieldValue('source_file_sha256'),
      literature_folder: inputValOr(root, '#ideaLiteratureFolder', 'literature_folder'),
      literature_pdf_count: literatureScan && literatureScan.folder ? Number(literatureScan.folder.pdf_count || 0) : Number(fieldValue('literature_pdf_count') || 0),
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
    if (srcType === 'frontier') return t('Frontier topic seed', '前沿主题种子');
    return t('Manual idea seed', '手动想法种子');
  }
  function modePrimaryLabel() {
    if (srcType === 'frontier') return t('Frontier topic / journal scope', '前沿主题 / 期刊范围');
    if (srcType === 'url') return t('Question this article suggests', '这篇文章启发的问题');
    if (srcType === 'pdf') return t('Question from this excerpt', '这段摘录启发的问题');
    if (srcType === 'literature_folder') return t('Review scope for this library', '这批文献的综述范围');
    return t('Idea / research question', '想法 / 研究问题');
  }
  function modeExcerptLabel() {
    if (srcType === 'frontier') return t('Reasoning notes or review theme', '推理备注或 review 主题');
    if (srcType === 'pdf') return t('Bounded PDF excerpt', '有界 PDF 摘录');
    if (srcType === 'url') return t('Abstract / quoted trigger sentence', '摘要 / 触发原文句子');
    if (srcType === 'literature_folder') return t('Library notes', '文献库备注');
    return t('Source quote or rationale sentence', '来源引用或触发句');
  }
  function modePlaceholder() {
    if (srcType === 'frontier') return 'e.g. ICU long-term outcomes after septic shock, editorials/reviews in Intensive Care Medicine';
    if (srcType === 'url') return 'e.g. What ICU-database study could test this trial or review insight?';
    if (srcType === 'pdf') return 'e.g. The excerpt suggests a measurable ICU exposure, outcome, or subgroup.';
    if (srcType === 'literature_folder') return 'e.g. septic shock resuscitation reviews, ARDS ventilation editorials';
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
    if (srcType === 'frontier') {
      return t('Frontier mode can prepare queries without network access. If you opt in, it searches bounded PubMed metadata/abstracts, maps candidate ideas to the EasyICU dictionary, then lets you choose one for the local ledger.', '前沿模式在未联网时只准备检索式；如果你 opt-in，会检索有界 PubMed 元数据/摘要，把候选 idea 映射到 EasyICU 字典，并让你选择其中一个生成本地台账。');
    }
    return t('Manual mode creates a local, evidence-bound idea ledger from the text you provide.', '手动模式会根据你提供的文本生成本地、证据绑定的 idea 台账。');
  }
  function validatePayload(payload) {
    const hasTopic = !!payload.topic;
    const hasExcerpt = !!payload.excerpt;
    const hasTitle = !!payload.title;
    if (srcType === 'url' && !payload.url) return t('Paste the article URL, then add a title, abstract, excerpt, or rationale sentence.', '请先粘贴文章链接，并补充标题、摘要、摘录或触发句。');
    if (srcType === 'url' && !(hasTopic || hasExcerpt || hasTitle || payload.doi)) return t('Resolve the URL or add a title, DOI, abstract, excerpt, or rationale sentence before mining.', '请先解析链接，或补充标题、DOI、摘要、摘录或触发句再挖掘。');
    if (srcType === 'pdf' && !(hasExcerpt || payload.source_file_sha256 || hasTitle)) return t('Choose a local PDF or paste a bounded PDF excerpt before mining.', '请先选择本地 PDF 或粘贴有界 PDF 摘录。');
    if (srcType === 'literature_folder' && !payload.literature_folder) return t('Choose or paste a local literature folder before mining.', '请先选择或粘贴本地文献库文件夹。');
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
    planDraft = data && data.idea_plan ? data.idea_plan : null;
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
      ['evidence', t('Pre-experiment', '预实验'), pre ? esc(pre.status || 'checked') : t('after mining', '挖掘后生成'), pre ? 'beaker' : 'shield'],
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
    return `
      <div class="ideas-discovery mt-12">
        <div class="ideas-prior-top">
          <div>
            <h3>${t('Frontier literature discovery', '前沿文献发现')}</h3>
            <p>${esc((discovery && discovery.reason) || t('Searches PubMed metadata/abstracts only after explicit network opt-in; then maps each article into an EasyICU idea candidate.', '只有明确网络 opt-in 后才检索 PubMed 元数据/摘要；随后把每篇文章映射成 EasyICU idea 候选。'))}</p>
          </div>
          <button class="btn ${fieldValue('allow_network') === true || fieldValue('allow_network') === 'true' ? 'primary' : ''}" data-idea-discover ${discovering ? 'aria-disabled="true"' : ''}>${discovering ? '<span class="spin"></span>' : icon('search', 13)} ${t('Discover papers', '检索文章')}</button>
        </div>
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
    return `
      <details class="ideas-advanced mt-10">
        <summary>${icon('shield', 13)} ${t('Network and provider opt-in', '网络与模型 opt-in')} <span>${t('off by default', '默认关闭')}</span></summary>
        <label class="rtodo-row mt-10 ideas-network-row">
          <input type="checkbox" id="ideaNetworkOptIn" ${fieldValue('allow_network') === 'true' || fieldValue('allow_network') === true ? 'checked' : ''} />
          <span class="rtodo-t">${t('Allow one bounded network metadata/prior-art request for this source', '允许本来源进行一次有界网络元数据 / prior-art 请求')}</span>
          <span class="rtodo-ref mono">opt-in</span>
        </label>
        <div class="muted mt-8">${t('URL/DOI metadata and PubMed prior-art checks stay blocked until this source-level opt-in is checked. Provider calls still require provider readiness.', 'URL/DOI 元数据和 PubMed prior-art 检查在勾选本来源 opt-in 前保持阻断。Provider 调用仍需要 provider readiness。')}</div>
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
  function sourceSpecificForm() {
    if (srcType === 'manual') return `
      <div class="ideas-primary-grid mt-14">
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="4" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        <label class="field ideas-field"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="4" placeholder="${t('Optional: paste the sentence, clinical observation, or rationale that triggered the idea.', '可选：粘贴触发这个想法的句子、临床观察或理由。')}">${esc(fieldValue('excerpt'))}</textarea></label>
      </div>`;
    if (srcType === 'url') return `
      <div class="ideas-url-stack mt-14">
        <label class="field ideas-field"><span>${t('Article URL', '文章链接')}</span><input id="ideaUrl" placeholder="https://www.nejm.org/doi/full/..." value="${esc(fieldValue('url'))}" /></label>
        <div class="ideas-meta-grid mt-10">
          <label class="field ideas-field"><span>DOI / PMID</span><input id="ideaDoi" placeholder="10.xxxx or PMID" value="${esc(fieldValue('doi'))}" /></label>
          <label class="field ideas-field"><span>Title</span><input id="ideaTitle" placeholder="${t('Resolved or manually entered title', '解析或手动输入的标题')}" value="${esc(fieldValue('title'))}" /></label>
          <label class="field ideas-field"><span>Journal</span><input id="ideaJournal" placeholder="e.g. NEJM" value="${esc(fieldValue('journal'))}" /></label>
          <label class="field ideas-field ideas-year"><span>Year</span><input id="ideaYear" placeholder="2026" value="${esc(fieldValue('year'))}" /></label>
        </div>
        <label class="field ideas-field mt-10"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="3" placeholder="${t('Optional: paste the article sentence that should become an ICU-database question.', '可选：粘贴应转化为 ICU 数据库问题的文章句子。')}">${esc(fieldValue('excerpt'))}</textarea></label>
        ${optInBlock()}
      </div>`;
    if (srcType === 'pdf') return `
      <div class="ideas-url-stack mt-14">
        ${pdfPickerBlock()}
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="3" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        <label class="field ideas-field"><span>${modeExcerptLabel()}</span><textarea id="ideaExcerpt" rows="4" placeholder="${t('Optional: paste a bounded passage if the PDF parser did not extract the motivating sentence.', '可选：如果 PDF 解析没有抽到触发句，可以粘贴一段有界摘录。')}">${esc(fieldValue('excerpt'))}</textarea></label>
        <div class="ideas-meta-grid mt-10">
          <label class="field ideas-field"><span>Title</span><input id="ideaTitle" placeholder="${t('Auto-filled from PDF when available', '可由 PDF 自动填充')}" value="${esc(fieldValue('title'))}" /></label>
          <label class="field ideas-field"><span>DOI / PMID</span><input id="ideaDoi" placeholder="10.xxxx or PMID" value="${esc(fieldValue('doi'))}" /></label>
        </div>
      </div>`;
    if (srcType === 'literature_folder') return `
      <div class="ideas-url-stack mt-14">
        ${literatureFolderBlock()}
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="3" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
      </div>`;
    return `
      <div class="ideas-url-stack mt-14">
        <label class="field ideas-field"><span>${modePrimaryLabel()}</span><textarea id="ideaTopic" rows="4" placeholder="${esc(modePlaceholder())}">${esc(fieldValue('topic'))}</textarea></label>
        <div class="ideas-meta-grid mt-10">
          <label class="field ideas-field"><span>${t('Journal scope', '期刊范围')}</span><input id="ideaJournal" placeholder="NEJM, JAMA, ICM..." value="${esc(fieldValue('journal'))}" /></label>
          <label class="field ideas-field ideas-year"><span>${t('Year window', '年份窗口')}</span><input id="ideaYear" placeholder="2024-2026" value="${esc(fieldValue('year'))}" /></label>
        </div>
        ${optInBlock()}
        ${discoveryPanel()}
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
        ${sourceSpecificForm()}
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
    const isEventMetric = (s) => s && s.metric_kind === 'event_rate';
    const riskCount = stats.filter(s => !isEventMetric(s) && (s.low_coverage || pct(s.coverage_pct) < 50)).length;
    const featureRow = (s) => {
      const n = s.numeric_summary || {};
      const eventMetric = isEventMetric(s);
      const metricPct = eventMetric ? pct(s.event_rate_pct) : pct(s.coverage_pct);
      const tone = eventMetric ? 'event' : coverageTone(metricPct);
      const summary = eventMetric
        ? t('binary/event indicator; non-events are not missing', '二分类/事件指标；阴性患者不是缺失')
        : (n.available ? `median ${fmt(n.median)} · min ${fmt(n.min)} · max ${fmt(n.max)}` : t('categorical, non-numeric, or empty', '分类、非数值或为空'));
      const meta = eventMetric
        ? `<span>${t('Events', '事件')} ${fmt(s.event_entities ?? s.records)}</span><span>${t('Non-events', '非事件')} ${fmt(s.non_event_entities)}</span>`
        : `<span>${t('Records', '记录')} ${fmt(s.records)}</span><span>${t('Missing', '缺失')} ${pctLabel(s.missing_pct)}</span>`;
      return `<div class="ideas-feature-row ${tone}">
        <div class="ideas-feature-name">
          <b>${esc(s.label)}</b>
          <span class="mono">${esc(s.module || '')} · ${esc(s.concept_id || '')}</span>
        </div>
        <div class="ideas-feature-cov">
          <div class="ideas-cov-head"><span>${eventMetric ? t('Event rate', '事件率') : t('Coverage', '覆盖')}</span><b>${pctLabel(eventMetric ? s.event_rate_pct : s.coverage_pct)}</b></div>
          <div class="ideas-cov-bar"><i style="width:${metricPct}%"></i></div>
        </div>
        <div class="ideas-feature-meta">
          ${meta}
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
    const plan = (planDraft && planDraft.plan) || (window.EU_IDEA_HANDOFF && window.EU_IDEA_HANDOFF.handoff_plan);
    if (!result) return '';
    if (!plan) {
      return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('agent', 14)}</span>
          <div><h2>${t('Plan / replan before Agent', 'Agent 前计划 / replan')}</h2><p>${t('Generate a study plan from the idea ledger and pre-experiment before freezing an Agent handoff.', '先根据 idea 台账和预实验生成研究计划，然后再冻结交接给 Agent。')}</p></div>
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
    return `
      <div class="card pad ideas-core-card">
        <div class="section-head">
          <span class="sec-ico">${icon('agent', 14)}</span>
          <div><h2>${t('Plan / replan before Agent', 'Agent 前计划 / replan')}</h2><p>${t('Confirm or revise the plan before sending it to Agent Projects. This does not unlock a manuscript draft.', '交给 Agent Projects 前先确认或修订计划。这里不会解锁论文草稿。')}</p></div>
        </div>
        <div class="note ok mt-8"><div class="ico">${icon('check', 14)}</div><div class="body"><div class="t">${esc(plan.research_question || '')}</div><div class="d">${esc((plan.agent_boundary && plan.agent_boundary.reason) || t('Draft analysis plan is locked until human confirmation and evidence checks pass.', '分析计划草稿在人工确认和证据核验通过前保持锁定。'))}</div></div></div>
        <div class="ledger compact mt-12">
          ${(plan.analysis_plan || []).map((x, i) => `<div class="ledger-row"><span class="ledger-ico">${String(i + 1).padStart(2, '0')}</span><div>${esc(x)}</div></div>`).join('')}
        </div>
        ${patterns.length ? `<details class="ideas-compact-details mt-10" open><summary>${icon('book', 13)} ${t('Reference method patterns', '参考方法套路')} <span>${patterns.length}</span></summary>${miniList(patterns)}</details>` : ''}
        ${constraints.length ? `<details class="ideas-compact-details mt-10"><summary>${icon('shield', 13)} ${t('ICU constraints', 'ICU 场景约束')} <span>${constraints.length}</span></summary>${miniList(constraints)}</details>` : ''}
        ${confirmations.length ? `<div class="ideas-interpretation mt-10"><div>${icon('shield', 13)} <span>${t('Still needs confirmation', '仍需确认')}: ${confirmations.map(esc).join(' · ')}</span></div></div>` : ''}
        <label class="field ideas-plan-edits mt-12"><span>${t('Natural-language plan edits', '用自然语言微调计划')}</span><textarea id="ideaPlanEdits" rows="4" placeholder="${t('e.g. use AKI as the endpoint, restrict to first ICU stay, add missingness sensitivity...', '例如:把 AKI 作为结局,限制首次 ICU 入住,增加缺失敏感性分析...')}">${esc(planEdits)}</textarea></label>
        <div class="row gap-8 mt-12">
          <button class="btn" data-idea-replan ${planning ? 'aria-disabled="true"' : ''}>${planning ? '<span class="spin"></span>' : icon('refresh', 14)} ${t('Replan from notes', '根据说明重规划')}</button>
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
      result = null; err = null; planEdits = ''; sourceResolved = null; discovery = null; priorArt = null; planDraft = null; projectSeed = null; selectedRunId = null; selectedRecordKey = null; draft = {}; activeStep = 'source'; window.EU_IDEA_HANDOFF = null; repaint();
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
    root.querySelectorAll('[data-idea-use-discovery]').forEach(btn => btn.addEventListener('click', () => {
      useDiscoveryCandidate(btn.dataset.ideaUseDiscovery || 0);
    }));
    const mineBtn = root.querySelector('[data-idea-mine]');
    if (mineBtn) mineBtn.addEventListener('click', () => {
      if (mining || !(window.EU_API && window.EU_API.mineIdeas)) return;
      const payload = collectPayload(document);
      const validationError = validatePayload(payload);
      if (validationError) { err = validationError; repaint(); return; }
      mining = true; err = null; result = null; priorArt = null; projectSeed = null; window.EU_IDEA_HANDOFF = null;
      repaint();
      window.EU_API.mineIdeas(payload).then(data => {
        result = data; selectedRunId = data.run_id || null; selectedRecordKey = runRecordKey(data, 'current') || selectedRunId; err = null; planEdits = ''; planDraft = null; activeStep = 'ledger'; window.EU_IDEA_LAST_RUN = data; upsertHistoryRun(data);
      }).catch(e => {
        err = e.message || String(e);
      }).finally(() => { mining = false; repaint(); });
    });
    const planBox = root.querySelector('#ideaPlanEdits');
    if (planBox) planBox.addEventListener('input', () => {
      planEdits = planBox.value;
      window.EU_IDEA_HANDOFF = null;
      projectSeed = null;
    });
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
