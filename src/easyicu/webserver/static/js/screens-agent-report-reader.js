/* Owner: registered report figures, manuscript references and evidence lineage.
   Pure renderers shared through AGENT_RENDER. Load before screens-agent-render.js.
   No mutable closure state: escaping/i18n/icons are the page's shared owners;
   the manuscript view receives artifactTable and manuscriptUnavailable from
   the artifact renderer, so this owner never reaches back into its closure. */
(function () {
  const { esc, escAttr } = window.EU_HTML;
  const t = window.t;
  const icon = window.icon;

  function boundedPngDataUrl(value) {
    const source = String(value || '');
    return /^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(source) ? source : '';
  }
  function displayAnchorId(row) {
    const displayId = String(row && row.display_id || '').trim();
    return /^[A-Za-z0-9 _.:-]{1,160}$/.test(displayId) ? displayId : '';
  }
  function displayAnchorAttributes(row) {
    const id = displayAnchorId(row);
    // Anchors are matched through the data attribute (dataset compare in the
    // preview and run-files hosts); no consumer reads an element id, and
    // registered display ids like "Table 1" legitimately contain spaces.
    return id ? ` data-gpi-display-anchor="${escAttr(id)}"` : '';
  }
  function figureReading(row) {
    const charts = new Set(Array.isArray(row.chart_types) ? row.chart_types : []);
    const roles = new Set(Array.isArray(row.panel_roles) ? row.panel_roles : []);
    let title = '';
    let clinical = '';
    // Explain the registered display's scope, never infer a result from its
    // filename or turn a coverage panel into independent sensitivity evidence.
    if (charts.has('dot_interval_absolute_risk') && charts.has('forest')) {
      title = t('Group outcomes and association estimates', '分组结局与关联估计');
      clinical = t('Compare observed group risks and association estimates; observational associations do not establish causation.',
        '比较各组结局发生比例与关联估计；观察性关联不能说明因果。');
    } else if (roles.has('cohort_accounting') || charts.has('cohort_flow')) {
      title = t('Cohort selection', '研究人群筛选');
      clinical = t('Counts show the recorded eligibility stages; the final analysis population may differ.',
        '展示各筛选阶段保留和排除的记录数；最终分析人数可能不同。');
    } else if (roles.has('data_quality')) {
      title = t('Data availability', '数据可用情况');
      clinical = t('Read source availability and record coverage; missingness is not absence of a clinical event.',
        '查看变量缺失和记录覆盖情况；缺少记录不等于未发生临床事件。');
    } else if (charts.has('specification_grid')) {
      title = t('Analysis specifications', '不同分析设定');
      clinical = t('Read each analysis estimate and its limitations separately; rows need not be directly comparable or independent.',
        '逐项查看分析估计及其限制；各行不一定可直接比较，也不一定是独立验证。');
    } else if (roles.has('primary_estimand') && charts.has('forest')) {
      title = t('Primary association estimate', '主要关联估计');
      clinical = t('Read the primary association and its uncertainty; an association does not establish causation.',
        '查看主要关联估计及其不确定性；关联不能说明因果。');
    } else if (roles.has('descriptive_result') || roles.has('distribution')) {
      title = t('Group comparison', '分组结果比较');
      clinical = t('Compare the observed groups; a descriptive difference does not establish causation.',
        '比较各组的观察结果；描述性差异不能说明因果。');
    }
    const prose = value => {
      // A product token ("figure:cohort flow") names the file, not the figure.
      const text = String(value || '').trim().replace(/^figure:\s*/i, '');
      return /^(?:file:\/\/|\/|[A-Za-z]:\\|\.\.?[\\/])|^(?:[\w.-]+[\\/])*[\w.-]+\.(?:png|pdf|svg|tiff?)$/i.test(text) ? '' : text;
    };
    return {
      title: title || prose(row.label) || t('Study figure', '研究结果图'),
      clinical,
      scientific: prose(row.caption),
    };
  }
  function figureGallery(payload) {
    const figs = payload && Array.isArray(payload.figures) ? payload.figures : [];
    const visible = figs
      .map(row => ({ row, imageSource: boundedPngDataUrl(row && (row.data_url || row.image_data_url)) }))
      .filter(item => item.row && item.imageSource);
    if (!visible.length) return '';
    const downloadControl = row => {
      const runRef = String(payload.run_ref || '');
      if (!/^rr_[a-f0-9]{32}$/.test(runRef)) return '';
      const exports = row.export_locators || {};
      const formats = ['pdf', 'svg', 'tiff', 'png'].filter(format => validFigureExport(exports[format]));
      if (!formats.length) return '';
      const labels = formats.map(format => format.toUpperCase()).join(' · ');
      return `<details class="ag-figure-download"><summary class="btn sm" aria-label="${escAttr(t('Download figure', '下载图件') + ': ' + labels)}">${icon('download', 13)} ${esc(labels)}</summary><div class="ag-figure-formats">${formats.map(format => {
        const locator = exports[format];
        return `<button class="btn sm" type="button" data-gpi-figure-download data-figure-run-ref="${escAttr(runRef)}" data-figure-format="${format}" data-figure-evidence-id="${escAttr(locator.evidence_id)}" data-figure-sha256="${escAttr(locator.sha256)}" aria-label="${escAttr(t('Download ', '下载 ') + format.toUpperCase())}">${format.toUpperCase()}</button>`;
      }).join('')}</div></details>`;
    };
    const sourceButton = row => {
      const source = row && row.source_code && typeof row.source_code === 'object' ? row.source_code : {};
      const evidenceId = String(source.evidence_id || '');
      const sha256 = String(source.sha256 || '').toLowerCase();
      if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) return '';
      const focusStart = Number(source.focus_start_line || 0);
      const focusEnd = Number(source.focus_end_line || 0);
      const focus = Number.isInteger(focusStart) && Number.isInteger(focusEnd) && focusStart > 0 && focusEnd >= focusStart
        ? ` data-source-focus-start="${focusStart}" data-source-focus-end="${focusEnd}"` : '';
      return `<button class="ag-figure-source" type="button" data-gpi-source-code data-source-evidence-id="${escAttr(evidenceId)}" data-source-sha256="${escAttr(sha256)}" data-source-title="${escAttr(t('Figure generation source', '图表生成代码'))}" data-source-output="${escAttr(figureReading(row).title)}" data-source-step="${escAttr(source.produced_by_step || '')}" data-source-producer="${escAttr(source.producer || '')}" data-source-generation="${escAttr(source.generation_mode || '')}" data-source-display-name="${escAttr(source.display_name || 'analysis.py')}" data-source-language="${escAttr(source.language || 'text')}"${focus}>${icon('search', 13)} ${esc(t('View source', '查看源代码'))}</button>`;
    };
    return `
      <div class="ag-figure-gallery">
        ${visible.map(({ row, imageSource }, index) => {
          const reading = figureReading(row);
          return `
          <figure${displayAnchorAttributes(row)} class="${index === 0 || row.tier === 'primary_publication' || row.status === 'canonical_main' ? 'is-primary' : 'is-supporting'}">
            <label class="ag-figure-image"><input type="checkbox" aria-label="${escAttr(t('Show figure at original size', '按原尺寸查看图件'))}"><span class="ag-figure-zoom-label">${esc(t('Original size', '原尺寸'))}</span><span class="ag-figure-scroll" tabindex="0"><img src="${escAttr(imageSource)}" alt="${escAttr(reading.title)}" /></span></label>
            <figcaption><strong>${esc(reading.title)}</strong>${reading.clinical ? `<p class="ag-figure-caption">${esc(reading.clinical)}</p>` : ''}${reading.scientific ? reading.clinical
              ? `<details class="ag-figure-scientific-caption"><summary>${esc(t('Scientific caption', '英文图注'))}</summary><p>${esc(reading.scientific)}</p></details>`
              : `<p class="ag-figure-caption">${esc(reading.scientific)}</p>` : ''}${downloadControl(row)}${sourceButton(row)}</figcaption>
          </figure>`;
        }).join('')}
      </div>`;
  }
  function validFigureExport(locator) {
    return locator && typeof locator === 'object'
      && Object.keys(locator).sort().join(',') === 'evidence_id,kind,sha256'
      && locator.kind === 'figure' && /^[A-Za-z0-9_.-]{1,160}$/.test(locator.evidence_id || '')
      && /^[a-f0-9]{64}$/.test(locator.sha256 || '');
  }
  function figureDownloadRequest(button) {
    const data = button && button.dataset || {};
    const locator = { evidence_id: data.figureEvidenceId, sha256: data.figureSha256, kind: 'figure' };
    return /^rr_[a-f0-9]{32}$/.test(data.figureRunRef || '')
      && ['pdf', 'svg', 'tiff', 'png'].includes(data.figureFormat) && validFigureExport(locator)
      ? { run_ref: data.figureRunRef, figure_export: { format: data.figureFormat, ...locator } } : null;
  }
  // One row per lineage node the provenance owner emits, in its fixed order
  // from the reported number back to the data source. Facts arrive as typed
  // counts and identifiers; this owner only words them.
  const LINEAGE_LABELS = {
    statistic: ['Result', '统计结果'],
    analysis_input: ['Analysis table', '分析数据表'],
    cohort: ['Study population', '研究人群'],
    column_definitions: ['Variable definitions', '变量定义'],
    concept_dictionary: ['Concept dictionary', '概念字典'],
    data_source: ['Data source', '数据来源'],
  };
  const LINEAGE_REASONS = {
    upstream_unproven: ['Not checked: the step above could not be verified', '上一环未能核实，这一环无法确认'],
    artifact_unregistered: ['This run did not register the record', '本次运行没有登记这份记录'],
    artifact_unavailable: ['The record is not in this run', '记录文件不在本次运行中'],
    artifact_digest_mismatch: ['The record changed after it was registered', '记录登记后被改动过'],
    artifact_json_too_large: ['The record is too large to read here', '记录过大，无法在此读取'],
    artifact_json_invalid: ['The record could not be read', '记录无法读取'],
    analysis_input_ambiguous: ['More than one analysis table could have been used', '无法确定用的是哪一份分析数据'],
    analysis_input_not_recorded: ['The run did not record which analysis table it used', '运行没有记录所用的分析数据'],
    analysis_input_not_loaded: ['The run did not record reading the analysis table', '运行没有记录读取了分析数据'],
    analysis_input_not_table: ['The registered input is not a data table', '登记的分析输入不是数据表'],
    input_digest_not_recorded: ['The run did not record a checksum for the analysis table', '运行没有记录分析数据的校验值'],
    input_digest_mismatch: ['The table the analysis read differs from the registered one', '分析读取的数据与登记的数据表不一致'],
    cohort_input_unlinked: ['No registered link from the analysis table to the study population', '分析数据与研究人群之间没有登记的关联'],
    context_digest_mismatch: ['The population record differs from the run registration', '研究人群记录与运行登记不一致'],
    cohort_digest_mismatch: ['The population record differs from the run registration', '研究人群记录与运行登记不一致'],
    column_definitions_not_recorded: ['The run did not record its variable definitions', '运行没有记录变量定义'],
    dictionary_digest_not_recorded: ['The run did not record a checksum for the dictionary', '运行没有记录所用字典的校验值'],
    source_identity_not_recorded: ['The run did not record its data source', '运行没有记录数据来源'],
  };
  const DATABASE_NAMES = {
    miiv: 'MIMIC-IV', miiv_demo: 'MIMIC-IV Demo', eicu: 'eICU-CRD', eicu_demo: 'eICU-CRD Demo',
    mimic: 'MIMIC-III', mimic_demo: 'MIMIC-III Demo', aumc: 'AmsterdamUMCdb', hirid: 'HiRID', sic: 'SICdb',
  };
  function lineageCount(value) {
    return Number.isSafeInteger(value) && value >= 0 ? new Intl.NumberFormat().format(value) : '';
  }
  function lineageFacts(node) {
    const facts = node.facts && typeof node.facts === 'object' ? node.facts : {};
    const rows = lineageCount(facts.rows);
    const parts = [];
    if (node.kind === 'statistic' && typeof facts.method === 'string') parts.push(esc(facts.method));
    if (node.kind === 'analysis_input' && rows) parts.push(esc(t(`${rows} rows`, `${rows} 行`)));
    if (node.kind === 'cohort') {
      if (rows) parts.push(esc(t(`${rows} included`, `纳入 ${rows} 例`)));
      const start = facts.window_start_hours;
      const end = facts.window_end_hours;
      if (Number.isFinite(start) && Number.isFinite(end)) parts.push(esc(t(`window ${start}–${end} h`, `时间窗 ${start}–${end} 小时`)));
    }
    if (node.kind === 'column_definitions' && lineageCount(facts.count)) {
      parts.push(esc(t(`${lineageCount(facts.count)} variables`, `${lineageCount(facts.count)} 个变量`)));
    }
    if (node.kind === 'concept_dictionary' && lineageCount(facts.concept_count)) {
      parts.push(esc(t(`${lineageCount(facts.concept_count)} concepts`, `${lineageCount(facts.concept_count)} 个概念`)));
    }
    if (node.kind === 'data_source') {
      const database = typeof facts.database === 'string' ? (DATABASE_NAMES[facts.database] || facts.database) : '';
      const name = [database, facts.version].filter(value => typeof value === 'string' && value).join(' ');
      if (name) parts.push(esc(name));
      const doi = String(facts.doi || '');
      if (facts.official_sha256_matches === true) {
        const label = esc(t('matches the official PhysioNet release', '与 PhysioNet 官方发布一致'));
        parts.push(/^10\.\d{4,9}\/[A-Za-z0-9._;()/:-]{1,120}$/.test(doi)
          ? `<a href="https://doi.org/${escAttr(doi)}" target="_blank" rel="noopener noreferrer">${label}</a>` : label);
      } else if (facts.official_sha256_matches === false) {
        parts.push(`<span class="is-warn">${esc(t('does not match the official release', '与官方发布不一致'))}</span>`);
      }
    }
    return parts.join(' · ');
  }
  function lineageChain(lineage, evidenceButton) {
    const nodes = (Array.isArray(lineage) ? lineage : [])
      .filter(node => node && Object.prototype.hasOwnProperty.call(LINEAGE_LABELS, node.kind)).slice(0, 6);
    if (!nodes.length) return '';
    // A break is explained once; the unverified steps after it show only
    // their hollow marker.
    const firstUnproven = nodes.findIndex(node => node.missing_reason === 'upstream_unproven');
    return `<section class="gpi-lineage-wrap"><h3>${esc(t('Where this number comes from', '这个数字的来源'))}</h3><ol class="gpi-lineage">${nodes.map((node, index) => {
      const status = ['current', 'stale'].includes(node.status) ? node.status : 'missing';
      const [en, zh] = LINEAGE_LABELS[node.kind];
      const reason = LINEAGE_REASONS[node.missing_reason] || ['Not recorded', '未记录'];
      const repeated = node.missing_reason === 'upstream_unproven' && index > firstUnproven;
      const detail = status === 'current' ? lineageFacts(node) : (repeated ? '' : esc(t(reason[0], reason[1])));
      const open = status === 'current' ? evidenceButton(node, t(en, zh)) : '';
      return `<li class="is-${status}" style="--i:${index}"><div><strong>${esc(t(en, zh))}</strong>${status === 'stale' ? ` <span class="pill warn">${esc(t('changed', '已变更'))}</span>` : ''}${detail ? `<p>${detail}</p>` : ''}</div>${/<button/.test(open) ? open : ''}</li>`;
    }).join('')}</ol></section>`;
  }
  function manuscriptProvenanceView(payload, { artifactTable, manuscriptUnavailable }) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const claims = Array.isArray(p.claims) ? p.claims.slice(0, 240) : [];
    const blocks = Array.isArray(p.article_blocks) ? p.article_blocks.slice(0, 240) : [];
    const opening = blocks.slice(0, 4).flatMap(block => Array.isArray(block && block.segments) ? block.segments : [])
      .map(segment => String(segment && segment.text || '')).join('\n').slice(0, 600);
    if (manuscriptUnavailable(opening)) {
      return `<div class="ag-artifact-readable"><div class="ag-artifact-section"><div class="ag-artifact-section-title">${esc(t('Manuscript unavailable', '文章尚未生成'))}</div><p>${esc(t('This run has no readable manuscript. Review the run status and scientific review records for the reason; existing results remain available in the result tables and evidence records.', '本次运行没有可阅读的文章。请在运行状态与科学审阅记录中查看原因；已有结果仍可从结果表与证据记录查看。'))}</p></div></div>`;
    }
    const references = Array.isArray(p.references) ? p.references.slice(0, 120) : [];
    const referenceMap = new Map(references.map(row => [String(row.key || ''), row]));
    const claimMap = new Map(claims.map(row => [String(row && row.claim_id || ''), row || {}]));
    const readableText = value => {
      const source = String(value || '')
        .replace(/\s*\[(?!@)(?:[A-Za-z_][A-Za-z0-9_.-]*|\d+_[A-Za-z0-9_.-]+)\]/g, '')
        .replace(/\s+([,.;:)])/g, '$1');
      const tokens = source.split(/(\*\*[^*]+\*\*|\[@[^\]]+\])/g).filter(Boolean);
      return tokens.map(token => {
        if (/^\*\*[^*]+\*\*$/.test(token)) return `<strong>${esc(token.slice(2, -2))}</strong>`;
        if (/^\[@[^\]]+\]$/.test(token)) {
          const keys = Array.from(token.matchAll(/@([A-Za-z0-9_.:-]+)/g), match => match[1]);
          return keys.map(key => {
            const ref = referenceMap.get(key);
            if (!ref || !Number.isInteger(ref.number) || ref.number < 1) return `<span class="gpi-reader-citation" title="${escAttr(key)}">[ref]</span>`;
            return `<a class="gpi-reader-citation" data-gpi-reference="${ref.number}" href="#gpi-reference-${ref.number}" title="${escAttr(ref.title || key)}">[${ref.number}]</a>`;
          }).join(' ');
        }
        return esc(token);
      }).join('');
    };
    const claimSourceValue = claim => {
      // source_value is the numeric claim's rounded lexical label. Preserve
      // its canonical value for exact comparison with the admitted JSON field.
      if (claim.canonical_value == null) return claim.source_value == null ? '' : claim.source_value;
      return typeof claim.canonical_value === 'number' && Number.isFinite(claim.canonical_value)
        ? claim.canonical_value : '';
    };
    const claimEvidenceAttrs = claim => {
      if (claim.status && claim.status !== 'current') return '';
      const evidence = claim && claim.evidence && typeof claim.evidence === 'object' ? claim.evidence : {};
      const evidenceId = String(evidence.evidence_id || '').trim();
      const sha256 = String(evidence.sha256 || '').trim().toLowerCase();
      if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) return '';
      return ` data-gpi-evidence-open data-evidence-id="${escAttr(evidenceId)}" data-evidence-sha256="${escAttr(sha256)}" data-evidence-kind="${escAttr(String(evidence.kind || 'statistic'))}" data-evidence-label="${escAttr(t('Exact result source', '准确结果来源'))}" data-evidence-pointer="${escAttr(String(claim.source_json_pointer || ''))}" data-evidence-source-value="${escAttr(String(claimSourceValue(claim)))}"`;
    };
    const renderSegments = value => (Array.isArray(value) ? value : []).map(segment => {
      const text = readableText(segment && segment.text || '');
      const claimId = String(segment && segment.claim_id || '');
      if (!segment || segment.kind !== 'claim' || !claimMap.has(claimId)) return text;
      const claim = claimMap.get(claimId) || {};
      const evidenceAttrs = claimEvidenceAttrs(claim);
      return `<button type="button" class="gpi-bound-number" id="claim-${escAttr(claimId)}" data-gpi-claim="${escAttr(claimId)}"${evidenceAttrs} aria-controls="gpi-claim-detail-${escAttr(claimId)}" aria-expanded="false" title="${escAttr(evidenceAttrs ? t('Open result evidence preview', '打开结果证据预览') : t('Open evidence lineage', '查看证据链路'))}">${text}</button>`;
    }).join('');
    const reportFigures = figureGallery(p.figure_gallery || {});
    const displayAnchorButton = row => {
      const displayId = displayAnchorId(row);
      if (!displayId) return '';
      return ` <button type="button" class="gpi-display-link" data-gpi-display="${escAttr(displayId)}" title="${escAttr(t('Jump to the registered display', '跳转到对应图表'))}">${esc(t('Locate display', '定位图表'))}</button>`;
    };
    const tables = (Array.isArray(p.tables) ? p.tables.slice(0, 20) : []).map(table => {
      const anchor = displayAnchorAttributes(table);
      return `<section class="gpi-reader-table"${anchor}>${artifactTable(`${table.label || ''}. ${table.caption || ''}`, table.columns || [], table.rows || [], '', { formattedCells: true })}<details><summary>${esc(t('Table definitions and source', '表格定义与来源'))}</summary>${(Array.isArray(table.notes) ? table.notes : []).map(note => `<p>${esc(note)}</p>`).join('')}</details></section>`;
    }).join('');
    let displaysInserted = false;
    const figureContext = (Array.isArray(p.figure_context) ? p.figure_context : [])
      .map(note => `<p class="gpi-reader-context">${esc(note.text || '')}</p>`).join('');
    const displayInsert = figureContext + tables + (reportFigures
      ? `<section class="gpi-article-figure-insert"><div class="gpi-article-figure-head"><h2>${esc(t('Result figures', '结果图件'))}</h2><p>${esc(p.figure_gallery && p.figure_gallery.presentation_variant ? t('Re-rendered from digest-verified source tables. Original run figures remain unchanged.', '根据摘要核验后的源数据表重新排版；原始运行图件保持不变。') : t('Figures registered by this run.', '本次运行登记的图件。'))}</p></div>${reportFigures}</section>` : '');
    const article = blocks.map(block => {
      if (block && block.kind === 'verification_notes') {
        const notes = Array.isArray(block.notes) ? block.notes.slice(0, 24) : [];
        if (!notes.length) return '';
        return `<section class="gpi-verification-notes"><h2>${esc(t('Verification notes', '核验备注'))}</h2><ul>${notes.map(note => `<li><code>${esc(String(note && note.code || ''))}</code> ${esc(String(note && note.text || ''))}</li>`).join('')}</ul></section>`;
      }
      const content = renderSegments(block && block.segments);
      const headingText = (Array.isArray(block && block.segments) ? block.segments : [])
        .map(segment => String(segment && segment.text || '')).join('').trim();
      const insertHere = !displaysInserted && block && block.kind === 'heading'
        && Number(block.level || 2) === 2 && /^(Discussion|讨论)$/i.test(headingText);
      const figureInsert = insertHere ? displayInsert : '';
      if (insertHere) displaysInserted = true;
      if (block && block.kind === 'heading') {
        const level = Math.max(2, Math.min(4, Number(block.level || 2)));
        return `${figureInsert}<h${level}>${content}</h${level}>`;
      }
      return `${figureInsert}<p>${content}</p>`;
    }).join('');
    const incompleteBibliography = references.some(ref => !Array.isArray(ref.authors) || !ref.authors.length)
      ? `<p class="gpi-reference-notice">${esc(t('Some source records have no author metadata. The bibliography requires verification.', '部分来源记录缺少作者信息，参考文献仍需核验。'))}</p>` : '';
    const referenceList = references.length ? `<section class="gpi-reader-references"><h2>${esc(t('References', '参考文献'))}</h2>${incompleteBibliography}<ol>${references.map(ref => {
      const authors = Array.isArray(ref.authors) ? ref.authors.join(', ') : '';
      const url = ref.doi ? `https://doi.org/${encodeURIComponent(ref.doi)}` : (/^https:\/\//i.test(String(ref.url || '')) ? ref.url : '');
      const title = url ? `<a href="${escAttr(url)}" target="_blank" rel="noopener noreferrer">${esc(ref.title || '')}</a>` : esc(ref.title || '');
      const titleStop = /[.!?。！？]$/.test(String(ref.title || '').trim()) ? '' : '.';
      const notices = Array.isArray(ref.bibliographic_notices) ? ref.bibliographic_notices : [];
      return `<li id="gpi-reference-${Number.isInteger(ref.number) ? ref.number : 0}">${esc(authors)}${authors ? '. ' : ''}${title}${titleStop} ${esc(ref.venue || '')}${ref.venue ? '. ' : ''}${esc(ref.year || '')}.${notices.map(note => `<p class="gpi-reference-notice">${esc(note)}</p>`).join('')}</li>`;
    }).join('')}</ol></section>` : '';
    const revision = p.report_revision && p.report_revision.status === 'pass'
      ? `<p class="gpi-reader-revision">${esc(t('Report revision', '报告修订'))}: ${esc(p.report_revision.revision_id || '')} · ${esc(t('Source analysis unchanged; not publication authorization.', '源分析不变；不代表发表授权。'))}</p>` : '';
    const evidenceButton = (row, label, pointer, sourceValue, text) => {
      const evidenceId = String(row && row.evidence_id || '').trim();
      const sha256 = String(row && row.sha256 || '').trim().toLowerCase();
      if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) {
        return esc(label || evidenceId || t('Unavailable', '不可用'));
      }
      return `<button type="button" class="gpi-evidence-open" data-gpi-evidence-open data-evidence-id="${escAttr(evidenceId)}" data-evidence-sha256="${escAttr(sha256)}" data-evidence-kind="${escAttr(String(row.kind || 'artifact'))}" data-evidence-role="${escAttr(String(row.role || ''))}" data-evidence-label="${escAttr(String(label || evidenceId))}" data-evidence-pointer="${escAttr(String(pointer || ''))}" data-evidence-source-value="${escAttr(String(sourceValue == null ? '' : sourceValue))}">${esc(text || label || evidenceId)}</button>`;
    };
    const lineageTable = (source, rows, pointer, sourceValue) => {
      const entries = [];
      if (source && source.evidence_id) {
        entries.push({ ...source, role: t('Exact result source', '准确结果来源'), kind: source.kind || 'statistic' });
      }
      (Array.isArray(rows) ? rows : []).forEach(row => entries.push(row || {}));
      if (!entries.length) return `<p class="gpi-claim-boundary">${esc(t('No registered evidence artifacts.', '没有登记证据产物。'))}</p>`;
      return `<div class="ag-artifact-section"><div class="ag-artifact-section-title">${esc(t('Open registered evidence', '打开已登记证据'))}</div><div class="ag-artifact-table-wrap"><table class="ag-artifact-table"><thead><tr><th>${esc(t('Role', '角色'))}</th><th>${esc(t('Type', '类型'))}</th><th>${esc(t('Preview', '预览'))}</th><th>SHA-256</th></tr></thead><tbody>${entries.map(row => `<tr><td>${esc(row.role || '')}${row.status && row.status !== 'current' ? ` <span class="pill warn">${esc(row.status)}</span>` : ''}</td><td>${esc(row.kind || '')}</td><td>${evidenceButton(row, row.evidence_id || t('Open', '打开'), pointer, sourceValue)}${displayAnchorButton(row)}</td><td>${esc(row.sha256 || '')}</td></tr>`).join('')}</tbody></table></div></div>`;
    };
    const panels = claims.map(claim => {
      const claimId = String(claim && claim.claim_id || '');
      const evidence = claim && claim.evidence && typeof claim.evidence === 'object' ? claim.evidence : {};
      const artifacts = Array.isArray(claim && claim.related_artifacts) ? claim.related_artifacts : [];
      const claimStatus = String(claim && claim.status || 'current').trim();
      const statusChip = claimStatus && claimStatus !== 'current'
        ? `<span class="pill warn">${esc(claimStatus)}</span>` : '';
      const methodRows = claim && claim.method_summary && typeof claim.method_summary === 'object'
        ? Object.entries(claim.method_summary).slice(0, 6).map(([key, value]) => [
          String(key).replace(/_/g, ' '), String(value == null ? '' : value),
        ]) : [];
      // The preview tab is named after the node; its renderer follows the
      // record's evidence kind, not the lineage step.
      const chain = lineageChain(claim && claim.lineage, (node, name) => evidenceButton({
        ...node, kind: node.kind === 'analysis_input' ? 'table' : (node.kind === 'statistic' ? 'statistic' : 'artifact'),
      }, name, '', '', t('Open', '打开')));
      const technical = `${artifactTable(t('Exact result source', '准确结果来源'), [t('Item', '项目'), t('Value', '值')], [
          [t('JSON field', 'JSON 字段'), claim.source_field || ''],
          [t('JSON pointer', 'JSON 指针'), claim.source_json_pointer || ''],
          [t('Source value', '源数值'), claimSourceValue(claim)],
          [t('Analysis step', '分析步骤'), claim.step_id || ''],
          [t('Evidence ID', '证据 ID'), evidence.evidence_id || ''],
          ['SHA-256', evidence.sha256 || ''],
          ...methodRows,
        ])}
        ${lineageTable(evidence, artifacts, claim.source_json_pointer, claimSourceValue(claim))}
        <p class="gpi-claim-boundary">${esc(t('This view exposes immutable IDs and digests, not patient rows or host file paths. Scientific authority remains analysis-only until Host gates and human review permit more.', '此视图只显示不可变 ID 与摘要，不暴露患者行或主机文件路径。除非 Host 闸门与人工审阅另行许可，科学权限仍为 analysis-only。'))}</p>`;
      // Readers saved before lineage existed keep their tables in view.
      return `<section class="gpi-claim-panel" id="gpi-claim-detail-${escAttr(claimId)}" data-gpi-claim-panel="${escAttr(claimId)}" hidden>
        <div class="gpi-claim-panel-head"><div><span>${esc(t('Bound number', '绑定数字'))}</span><strong>${esc(claim.display_value || '')}</strong>${statusChip}</div><button type="button" data-gpi-claim-close aria-label="${escAttr(t('Close evidence detail', '关闭证据详情'))}">${esc(t('Close', '关闭'))}</button></div>
        ${chain ? `${chain}<details class="gpi-claim-technical"><summary>${esc(t('IDs and checksums', 'ID 与校验值'))}</summary>${technical}</details>` : technical}
      </section>`;
    }).join('');
    return `<div class="ag-artifact-readable ag-manuscript-reader">
      <div class="ag-artifact-readable-head"><div><div class="eyebrow">${esc(t('Evidence-bound article', '证据绑定文章'))}</div><div class="ag-artifact-readable-title">${esc(t('Click a highlighted number to open its exact result evidence preview. Full lineage remains available when needed.', '点击高亮数字，直接打开对应结果证据的可视化；需要时仍可查看完整证据链。'))}</div></div><span class="pill warn">analysis-only</span></div>
      ${revision}<div class="gpi-manuscript-layout" data-gpi-manuscript-layout><article class="gpi-manuscript-article">${article || `<p>${esc(t('No reader blocks are available.', '没有可用的文章阅读内容。'))}</p>`}${displaysInserted ? '' : displayInsert}${referenceList}</article><aside class="gpi-claim-drawer" aria-live="polite"><div class="gpi-claim-empty" data-gpi-claim-empty>${esc(t('Claims without a previewable result source can still open their exact audit lineage here.', '没有可直接预览结果来源的论断，仍可在这里打开准确审计链路。'))}</div>${panels}</aside></div>
    </div>`;
  }

  window.AGENT_REPORT_READER = { figureReading, figureGallery, figureDownloadRequest, manuscriptProvenanceView };
})();
