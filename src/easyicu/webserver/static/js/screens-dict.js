/* Owner: Data Dictionary route. */
/* Screen: Data Dictionary — the real concept catalog, searchable.
   Native catalog browser with search + category
   browser over the 19 concept groups, with units and per-database coverage).
   Its study-type view is js/screens-dict-studies.js (window.EU_DICT_STUDIES). */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  const C = () => window.EU_CATALOG;

  let dictSearch = '';
  let dictCat = 'all';   // 'all' | group key
  let dictSelected = null;
  let dictSelectedDb = null;
  let dictLineage = null;
  let dictLineageLoading = false;
  let dictLineageError = '';
  let dictLineageRequest = 0;
  let dictShowInternal = false;
  let dictView = 'concepts';   // 'concepts' | 'studies'
  // The study-type view has its own address, #dictionary/studies.
  function hashView() {
    const hash = typeof location !== 'undefined' ? String(location.hash || '') : '';
    return hash === '#dictionary/studies' ? 'studies' : hash.startsWith('#dictionary') ? 'concepts' : null;
  }
  function showView(view) {
    dictView = view;
    if (typeof history !== 'undefined' && typeof location !== 'undefined') {
      history.replaceState(null, '', `${location.pathname}${location.search}#dictionary${view === 'studies' ? '/studies' : ''}`);
    }
    window.__euRender();
  }
  const DB_NAMES = () => window.EU_DATABASE_NAMES;

  function L(en, zh) { return t(en, zh); }
  function html(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));
  }
  function nameOf(k) { const d = C().dict[k]; return d ? d[0] : k; }
  function zhOf(k) { const d = C().dict[k]; return d ? d[1] : k; }
  function unitOf(k) { const d = C().dict[k]; return d && d[2] ? d[2] : '—'; }
  function descOf(k) { const d = C().desc[k]; return d ? (window.EU_LANG === 'zh' ? d[1] : d[0]) : ''; }
  // Catalog group names start with a picker emoji ("⭐ SOFA-2 Scores"); the
  // dictionary names a group by its words, as review_labels.plain_label does.
  function plainName(name) { const text = String(name || '').replace(/^[^\p{L}\p{N}]+/u, ''); return text || String(name || ''); }
  function groupNameOf(key) { const g = C().groups.find(g => g[0] === key); return g ? plainName(window.EU_LANG === 'zh' ? g[2] : g[1]) : key; }

  function hostedServer() { return ((window.EU_CAPABILITIES || {}).deployment || {}).hosted === true; }
  // A hosted instance counts coverage over the databases it actually has.
  function instanceDbs() { const dbs = C().instanceDatabases || []; return hostedServer() && dbs.length ? dbs : null; }
  function dbName(key) { return DB_NAMES().name(DB_NAMES().key(key)) || key; }
  // Bookkeeping concepts (receipts, evidence flags, helper series) stay out of
  // the list until asked for; the catalog's audience field decides.
  function isInternal(k) { const meta = (C().conceptCoverage || {})[k]; return !!(meta && meta.audience === 'internal'); }
  function listed(k) { return !!C().dict[k] && (dictShowInternal || !isInternal(k)); }
  // A concept's other definitions and component criteria fold under its
  // entry; the catalog's variants map decides. Selecting the entry or one of
  // its variants opens the family.
  function variantsOf(k) { return ((C().variants || {})[k] || []).filter(listed); }
  function primaryOf(k) {
    const map = C().variants || {};
    return Object.keys(map).find(primary => map[primary].includes(k)) || null;
  }
  function entry(k) { return listed(k) && !primaryOf(k); }
  function openFamily() { return dictSelected && (variantsOf(dictSelected).length ? dictSelected : primaryOf(dictSelected)); }
  function listedIn(gk) { return (C().groupConcepts[gk] || []).filter(entry); }
  function internalCount() { return Object.keys(C().dict || {}).filter(isInternal).length; }

  function supportedDbCount() { return (C().supportedDbs || []).length || 6; }

  // How a database holds a variable only as a substitute: the host's facts
  // (webserver/data_map.py, carried as the catalog's ``substitutes``) — a value
  // recorded in bands or capped above a limit, or a death counted elsewhere.
  const UNIT_WORDS = { years: ['year', '岁'], kg: ['kg', 'kg'] };
  function substituteOf(k, db) { return (((C().conceptCoverage || {})[k] || {}).substitutes || {})[db] || null; }
  function sourceSemanticOf(k, db) {
    const declared = (((C().conceptCoverage || {})[k] || {}).source_semantics || {})[db];
    // The bundled offline catalog must not imply an exact SIC definition.
    return declared || (k === 'los_icu' && db === 'sic' ? { kind: 'care_unit_to_case_closure_proxy' } : null);
  }
  function sourceSemanticText(k, db) {
    const semantic = sourceSemanticOf(k, db);
    if (semantic && semantic.kind === 'care_unit_to_case_closure_proxy') {
      return L('SICdb: first transfer to ICU or intermediate care until case closure; not exact ICU length of stay.',
        'SICdb：从首次转入 ICU 或中间监护病房到病例闭合的时长，不等于精确的 ICU 住院时长。');
    }
    return '';
  }
  function substituteText(k, sub) {
    if (!sub) return '';
    if (sub.meaning) {
      const words = window.EU_DICT_STUDIES ? window.EU_DICT_STUDIES.meaning(sub.meaning) : '';
      return words ? L(`Counts ${words}`, `口径为${/^[\x20-\x7e]/.test(words) ? ' ' : ''}${words}`) : '';
    }
    const unit = UNIT_WORDS[unitOf(k)] ? L(...UNIT_WORDS[unitOf(k)]) : '';
    const zhUnit = unit && /^[\x20-\x7e]+$/.test(unit) ? `${unit} ` : unit;
    if (sub.kind === 'approximate') return L('Rounded value; exact bounds are not specified', '数值经过取整，精确区间边界未说明');
    if (sub.kind === 'binned' && Number(sub.bin_width) > 0) {
      const top = sub.top_code == null ? NaN : Number(sub.top_code);
      return L(`Recorded in ${sub.bin_width}-${unit} bands`, `每 ${sub.bin_width} ${zhUnit}记为一段`)
        + (Number.isFinite(top) ? L(`; ${top} and above form one band`, `；${top} ${zhUnit}及以上为一段`) : '');
    }
    if (sub.kind === 'binned' && Array.isArray(sub.bin_labels) && sub.bin_labels.length) {
      return L(`Recorded in bands: ${sub.bin_labels.join(', ')}`, `按区间记录：${sub.bin_labels.join('、')}`);
    }
    if (sub.kind === 'binned') return L('Recorded in bands', '按区间记录');
    if (sub.kind === 'capped_or_shifted') {
      const through = sub.exact_through == null ? NaN : Number(sub.exact_through);
      return Number.isFinite(through)
        ? L(`Values above ${through} are capped or shifted`, `${through} ${zhUnit}以上的值被截断或平移`)
        : L('Values above a limit are capped or shifted', '超过上限的值被截断或平移');
    }
    return '';
  }

  function hasOwn(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj || {}, key);
  }

  function isDerivedRuleConcept(k) {
    return /^(sofa_|sofa2_|sep3_)/.test(k) || ['susp_inf', 'culture_positive', 'bld_culture_positive'].includes(k);
  }

  function coverageOf(k) {
    const meta = (C().conceptCoverage || {})[k];
    if (meta && !(meta.kind === 'not_audited' && isDerivedRuleConcept(k))) return meta;
    if (isDerivedRuleConcept(k)) {
      return { kind: 'derived', databases: null, basis: 'score_or_rule_component', source: 'rule_based_output' };
    }
    if (hasOwn(C().cov, k)) return { kind: 'audited', databases: C().cov[k], basis: 'CONCEPT_DB_COVERAGE' };
    return { kind: 'not_audited', databases: null, basis: 'missing_from_CONCEPT_DB_COVERAGE' };
  }

  function fallbackCoverageSummary() {
    const total = C().totalConcepts || Object.keys(C().dict || {}).length;
    const dbs = supportedDbCount();
    const cov = C().cov || {};
    const dictKeys = Object.keys(C().dict || {});
    const auditedAll = Object.keys(cov).filter(k => cov[k] === dbs).length;
    const auditedFive = Object.keys(cov).filter(k => cov[k] === dbs - 1).length;
    const auditedPartial = Object.keys(cov).filter(k => cov[k] > 0 && cov[k] < dbs - 1).length;
    const audited = Object.keys(cov).length;
    const derived = dictKeys.filter(k => !hasOwn(cov, k) && isDerivedRuleConcept(k)).length;
    return {
      supportedDatabases: dbs,
      audited,
      auditedAll,
      auditedFive,
      auditedPartial,
      derived,
      notAudited: Math.max(0, total - audited - derived),
    };
  }

  // Which supported databases the dictionary maps a concept to, by name; a
  // derived concept reads its inputs' mappings. An earlier data audit that
  // confirmed fewer of them is marked, since it records a count, not names.
  function declaredBadge(meta) {
    const declared = meta.declared_databases;
    if (!declared.length) {
      return `<span class="cov-badge unaudited" title="${html(L('The dictionary declares no source mapping for this concept.', '字典没有为这个概念声明来源映射。'))}">${L('No mapping', '未声明映射')}</span>`;
    }
    const all = C().supportedDbs && C().supportedDbs.length ? C().supportedDbs : DB_NAMES().keys();
    const names = declared.map(dbName).join(window.EU_LANG === 'zh' ? '、' : ', ');
    const derived = meta.kind === 'derived';
    const title = derived
      ? L(`Derived from its input concepts, which are mapped in: ${names}`, `由输入概念计算；输入映射所在库：${names}`)
      : L(`Dictionary mapping in: ${names}`, `字典映射所在库：${names}`);
    const audited = Number(meta.data_audit_databases);
    const audit = Number.isFinite(audited) && audited < declared.length
      ? `<span class="cov-audit" title="${html(L(`An earlier data audit confirmed usable data in ${audited} of these ${declared.length} databases.`, `早期数据核对只在这 ${declared.length} 个库中的 ${audited} 个确认有可用数据。`))}">${L(`data ${audited}/${declared.length}`, `数据核对 ${audited}/${declared.length}`)}</span>`
      : '';
    const mark = db => {
      if (!declared.includes(db)) return `<i>${html(DB_NAMES().abbr(db) || dbName(db))}</i>`;
      const note = sourceSemanticText(meta.key, db) || substituteText(meta.key, substituteOf(meta.key, db));
      return note
        ? `<i class="on sub" title="${html(`${dbName(db)}${L(': ', '：')}${note}`)}">${html(DB_NAMES().abbr(db) || dbName(db))}</i>`
        : `<i class="on">${html(DB_NAMES().abbr(db) || dbName(db))}</i>`;
    };
    return `<span class="cov-badge cov-declared${derived ? ' is-derived' : ''}" title="${html(title)}"><span class="cov-dbs">${all.map(mark).join('')}</span>${derived ? `<span class="cov-kind">${L('derived', '派生')}</span>` : ''}${audit}</span>`;
  }

  // An outcome the outcome owner (outcome_availability.py) supports in no
  // database yet: what it would need, short for the row and in full for the
  // detail panel. A concept without its own entry reads the general reason.
  const UNSUPPORTED_NEEDS = {
    vent_free_days_28: [['Needs complete ventilation and day-28 survival', '需完整通气记录和第 28 天生存'], ['every ventilation episode of a patient and whether they were alive on day 28', '患者的全部机械通气记录和第 28 天时是否存活']],
  };
  function unsupportedNeeds(k) {
    const [short, full] = UNSUPPORTED_NEEDS[k] || [];
    return {
      short: short ? L(...short) : L('Not computed in any database yet', '暂未在任何库中计算'),
      full: full
        ? L(`EasyICU does not compute this outcome yet. It needs ${full[0]}; an export's stay length, hospital death flag or row order cannot stand in for that, and no database's export has been shown to carry it.`,
          `EasyICU 暂不计算这个结局。它需要${full[1]}；导出里的住院时长、院内死亡标记或记录顺序都替代不了，目前也没有哪个库的导出被证明具备这些。`)
        : L('EasyICU does not compute this outcome in any database yet.', 'EasyICU 暂未在任何库中计算这个结局。'),
    };
  }
  function unsupportedBadge(k) {
    const needs = unsupportedNeeds(k);
    return `<span class="cov-unsupported" title="${html(needs.full)}"><span class="cov-badge unaudited">${L('Not supported yet', '暂不支持')}</span><span class="cov-why">${html(needs.short)}</span></span>`;
  }

  function covBadge(k) {
    if (coverageOf(k).kind === 'unsupported') return unsupportedBadge(k);
    const inst = instanceDbs();
    if (inst) {
      const covered = ((C().conceptCoverage || {})[k] || {}).instance_databases_covered || [];
      const cls = covered.length === inst.length ? 'hi' : covered.length ? 'mid' : 'lo';
      const notes = covered.map(db => [db, sourceSemanticText(k, db) || substituteText(k, substituteOf(k, db))]).filter(([, note]) => note)
        .map(([db, note]) => `${dbName(db)}${L(': ', '：')}${note}`);
      return `<span class="cov-badge ${cls}"${notes.length ? ` title="${html(notes.join(L('; ', '；')))}"` : ''}><span class="cov-n">${covered.length ? covered.map(dbName).join(' · ') : L('Not in this instance', '本实例无')}</span></span>`;
    }
    const meta = coverageOf(k);
    if (Array.isArray(meta.declared_databases)) return declaredBadge({ ...meta, key: k });
    if (meta.kind === 'derived') {
      return `<span class="cov-badge derived" title="${L('Derived or rule-based output. It is not a separate raw database mapping count.', '派生或规则输出,不是单独的原始数据库映射数量。')}">${L('derived', '派生')}</span>`;
    }
    if (meta.kind === 'not_audited') {
      return `<span class="cov-badge unaudited" title="${L('No static six-database mapping audit is recorded for this catalog concept. This is not active-export missingness.', '该字典概念没有静态六库映射审计记录。这不是当前导出数据缺失率。')}">${L('no catalog audit', '无字典审计')}</span>`;
    }
    const n = Number(meta.databases || 0);
    const cls = n >= 5 ? 'hi' : n >= 3 ? 'mid' : 'lo';
    const dbs = supportedDbCount();
    return `<span class="cov-badge ${cls}" title="${L('Mapping audited in ' + n + ' of ' + dbs + ' ICU databases', dbs + ' 个 ICU 数据库中已审计映射 ' + n + ' 个')}">
      <span class="cov-dots">${Array.from({ length: dbs }, (_, idx) => `<i class="${idx < n ? 'on' : ''}"></i>`).join('')}</span>
      <span class="cov-n">${n}/${dbs} ${L('DBs', '库')}</span></span>`;
  }

  function matchedRows() {
    const cat = C();
    const q = dictSearch.trim().toLowerCase();
    const matches = k => !q || `${k} ${nameOf(k)} ${zhOf(k)} ${descOf(k)}`.toLowerCase().includes(q);
    const open = openFamily();
    const groupsToScan = dictCat === 'all' ? cat.groups.map(g => g[0]) : [dictCat];
    const rows = [];
    groupsToScan.forEach(gk => {
      (cat.groupConcepts[gk] || []).forEach(k => {
        if (!entry(k)) return;
        const variants = variantsOf(k);
        const hits = q ? variants.filter(matches) : [];
        if (!matches(k) && !hits.length) return;
        rows.push({ k, group: gk, variants: variants.length });
        (k === open ? variants : hits).forEach(v => rows.push({ k: v, group: gk, variantOf: k }));
      });
    });
    return rows;
  }

  function resultsHtml(rows) {
    const hidden = internalCount();
    const toggle = hidden ? `<button type="button" class="btn sm ghost" data-dict-internal>${dictShowInternal ? L('Hide internal fields', '隐藏内部字段') : L(`Show internal fields (${hidden})`, `显示内部字段（${hidden}）`)}</button>` : '';
    const n = rows.filter(row => !row.variantOf).length;
    return `<div class="dict-resultline row" style="justify-content:space-between;align-items:center;"><span>${t(n + ' concept' + (n === 1 ? '' : 's'), n + ' 个概念')}${dictCat !== 'all' ? ` · ${groupNameOf(dictCat)}` : ''}${dictSearch ? ` · "${dictSearch.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]))}"` : ''}</span>${toggle}</div>${tableHtml(rows)}`;
  }
  function tableHtml(rows) {
    if (!rows.length) return `<div class="dict-empty">${icon('search', 22)}<div>${L('No matching concepts.', '未找到匹配的概念。')}</div></div>`;
    const coverageLabel = instanceDbs() ? L('Declared mappings here', '本实例声明映射') : L('Declared mappings', '字典声明映射');
    const head = `
      <div class="dict-row dict-head">
        <span class="dc-code">${L('Code', '代码')}</span>
        <span class="dc-name">${L('Concept', '概念')}</span>
        <span class="dc-unit">${L('Unit', '单位')}</span>
        <span class="dc-cov">${coverageLabel}</span>
        ${dictCat === 'all' ? `<span class="dc-cat">${L('Group', '分组')}</span>` : ''}
      </div>`;
    const open = openFamily();
    const body = rows.map(({ k, group, variants, variantOf }) => {
      const d = descOf(k);
      const expanded = variants ? ` aria-expanded="${open === k ? 'true' : 'false'}"` : '';
      const fold = variants ? `<span class="dn-fold">${L(`${variants} related variables`, `${variants} 个相关变量`)}</span>` : '';
      return `
      <button type="button" class="dict-row dict-concept-row${variantOf ? ' is-variant' : ''} ${dictSelected === k ? 'selected' : ''}" data-dict-concept="${html(k)}" aria-pressed="${dictSelected === k ? 'true' : 'false'}"${expanded}>
        <span class="dc-code mono">${k}</span>
        <span class="dc-name"><span class="dn-t">${window.EU_LANG === 'zh' ? zhOf(k) : nameOf(k)}</span>${d ? `<span class="dn-d">${d}</span>` : ''}${sourceSemanticText(k, 'sic') ? `<span class="dn-d">${html(sourceSemanticText(k, 'sic'))}</span>` : ''}${fold}</span>
        <span class="dc-unit mono">${unitOf(k)}</span>
        <span class="dc-cov">${covBadge(k)}</span>
        ${dictCat === 'all' ? `<span class="dc-cat">${variantOf ? '' : groupNameOf(group)}</span>` : ''}
      </button>`;
    }).join('');
    return `<div class="dict-table">${head}${body}</div>`;
  }

  function field(value) { return value == null || value === '' ? L('not declared', '未声明') : String(value); }
  function selector(mapping, full = false) {
    const values = Array.isArray(mapping.selector_values) ? mapping.selector_values : [];
    if (!mapping.selector_field || !values.length) return L('no row selector declared', '未声明行选择条件');
    const shown = (full ? values : values.slice(0, 4)).join(', ');
    return `${mapping.selector_field} ∈ {${shown}${!full && values.length > 4 ? ', …' : ''}}`;
  }
  function mappingSummary(lane) {
    return (lane.mappings || []).map(mapping => `${field(mapping.table)}.${field(mapping.value_field)}`).join(' + ');
  }
  function lineagePanel() {
    if (!dictSelected) return '';
    if (coverageOf(dictSelected).kind === 'unsupported') {
      return `<section class="dict-lineage"><div class="note warn"><div class="body"><div class="t">${L('Not supported yet', '暂不支持')}</div><div class="d">${html(unsupportedNeeds(dictSelected).full)}</div></div></div></section>`;
    }
    if (dictLineageLoading) return `<section class="dict-lineage"><div class="state loading"><span class="spin"></span><div class="t">${L('Loading declared lineage…', '正在加载声明的血缘…')}</div></div></section>`;
    if (dictLineageError) return `<section class="dict-lineage"><div class="note warn"><div class="body"><div class="t">${L('Lineage unavailable', '血缘暂不可用')}</div><div class="d">${html(dictLineageError)}</div></div></div></section>`;
    if (!dictLineage) return '';
    const lanes = dictLineage.lanes || [];
    if (!lanes.length) return `<section class="dict-lineage"><div class="note info"><div class="body"><div class="t">${L('No raw-source mapping declared', '未声明原始来源映射')}</div><div class="d">${L('This concept may be derived or rule-based. No table or field is inferred.', '该概念可能是派生或规则输出；这里不会猜测表或字段。')}</div></div></div></section>`;
    const active = lanes.find(lane => lane.database === dictSelectedDb) || lanes[0];
    dictSelectedDb = active.database;
    const canonical = dictLineage.canonical || {};
    const primary = active.mappings[0] || {};
    const callbacks = Array.from(new Set((active.mappings || []).map(mapping => mapping.callback).filter(Boolean)));
    const mappings = (active.mappings || []).map(mapping => `<div style="margin-top:12px;overflow-wrap:anywhere">
      <b>${html(field(mapping.table))}</b>
      <div>${html(`${L('Row selection', '行选择')}: ${selector(mapping, true)}`)}</div>
      <div>${html(`${L('value', '值')} ${field(mapping.value_field)} · ${L('unit', '单位')} ${field(mapping.unit_field)} · ${L('time', '时间')} ${field(mapping.time_field)}`)}</div>
      <div>${html(`${L('Transformation', '转换处理')}: ${mapping.callback || L('no callback declared', '未声明回调')}`)}</div>
    </div>`).join('');
    const steps = [
      [L('Source table', '来源表'), (active.mappings || []).map(mapping => field(mapping.table)).join(' + ')],
      [L('Row selection', '行选择'), (active.mappings || []).map(mapping => selector(mapping)).join('；')],
      [L('Raw fields', '原始字段'), (active.mappings || []).map(mapping => `${field(mapping.table)}: ${L('value', '值')} ${field(mapping.value_field)} · ${L('unit', '单位')} ${field(mapping.unit_field)} · ${L('time', '时间')} ${field(mapping.time_field)}`).join('；') || `${L('value', '值')} ${field(primary.value_field)} · ${L('unit', '单位')} ${field(primary.unit_field)} · ${L('time', '时间')} ${field(primary.time_field)}`],
      [L('Transformation', '转换处理'), callbacks.length ? callbacks.join('；') : L('No dictionary callback declared; identity cannot be assumed beyond the declared fields.', '字典未声明回调；除已声明字段外，不假定额外处理。')],
      [L('Bounds', '边界'), `${canonical.minimum == null ? '−∞' : canonical.minimum} – ${canonical.maximum == null ? '+∞' : canonical.maximum}`],
      [L('Canonical output', '标准输出'), `${dictLineage.concept}${canonical.unit ? ` · ${canonical.unit}` : ''}`],
    ];
    const substitute = substituteText(dictLineage.concept, substituteOf(dictLineage.concept, active.database));
    return `<section class="dict-lineage" data-dict-lineage>
      <div class="dict-lineage-head"><div><div class="eyebrow">${L('Feature provenance', '特征血缘')}</div><h2>${html(window.EU_LANG === 'zh' ? zhOf(dictLineage.concept) : dictLineage.name)} <span class="mono">${html(dictLineage.concept)}</span></h2><p>${L('Declared catalog mappings only. Fields marked “not declared” are left explicit instead of guessed.', '仅展示字典声明的映射；未声明的字段保持明确空缺，不做猜测。')}</p></div><span class="pill ok">${lanes.length} ${L('databases', '个数据库')}</span></div>
      <div class="dict-lineage-tabs" role="tablist">${lanes.map(lane=>`<button type="button" class="${lane.database===active.database?'active':''}" data-dict-lineage-db="${html(lane.database)}">${html(lane.label)}</button>`).join('')}</div>
      <div class="dict-audit-track" data-dict-audit-track>${steps.map(([label,value],index)=>`<div class="dict-audit-step"><i>${index+1}</i><div><b>${html(label)}</b><span>${html(value)}</span></div></div>`).join('')}</div>
      <details data-dict-mapping-details><summary>${L('View complete source mappings', '查看完整来源映射')}</summary>${mappings}</details>
      ${substitute ? `<p class="dict-lineage-sub" data-dict-substitute>${html(`${active.label}${L(': ', '：')}${substitute}`)}</p>` : ''}
      ${sourceSemanticText(dictLineage.concept, active.database) ? `<p class="dict-lineage-sub" data-dict-source-semantics>${html(sourceSemanticText(dictLineage.concept, active.database))}</p>` : ''}
      <div class="dict-lineage-lanes" data-dict-lineage-lanes>${lanes.map((lane,index)=>`<article style="--dict-lane-color:${window.EU_PALETTE.color(index)}"><div><i></i><b>${html(lane.label)}</b><span>${html(lane.database)}</span></div><strong>${html(mappingSummary(lane))}</strong><em>${html(Array.from(new Set((lane.mappings||[]).map(mapping=>mapping.callback).filter(Boolean))).join('；') || L('no callback declared', '未声明回调'))}</em><small>→ ${html(dictLineage.concept)}${canonical.unit?` · ${html(canonical.unit)}`:''}</small></article>`).join('')}</div>
    </section>`;
  }

  function catChips() {
    const cat = C();
    const chip = (key, label, n) => `<button class="dict-chip ${dictCat === key ? 'on' : ''}" data-dict-cat="${key}">${label}${n != null ? ` <span class="dch-n">${n}</span>` : ''}</button>`;
    return `<div class="dict-chips">
      ${chip('all', L('All', '全部'), Object.keys(cat.dict || {}).filter(entry).length)}
      ${cat.groups.filter(g => listedIn(g[0]).length).map(g => chip(g[0], groupNameOf(g[0]), listedIn(g[0]).length)).join('')}
    </div>`;
  }

  S.dictionary = {
    section: 'dictionary', nav: 'dictionary',
    get crumbs() { return [t('Home', '首页'), t('Data Dictionary', '数据字典')]; },
    actionHtml: '',
    rail() {
      const cat = C();
      const inst = instanceDbs();
      if (inst) {
        return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Catalog', '目录')}</span></div>
        <div class="col gap-6" style="font-size:12px;">
          <div class="setup-row"><span class="k">${t('Concepts', '概念')}</span><span class="vv">${Object.keys(cat.dict || {}).filter(entry).length}</span></div>
          <div class="setup-row"><span class="k">${t('Databases', '数据库')}</span><span class="vv">${inst.map(dbName).join(' · ')}</span></div>
        </div>
      </div>`;
      }
      const summary = cat.coverageSummary || fallbackCoverageSummary();
      const dbs = summary.supportedDatabases || supportedDbCount();
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Catalog', '目录')}</span></div>
        <div class="col gap-6" style="font-size:12px;">
          <div class="setup-row"><span class="k">${t('Groups', '分组')}</span><span class="vv">${cat.groups.length}</span></div>
          <div class="setup-row"><span class="k">${t('Concepts', '概念')}</span><span class="vv">${cat.totalConcepts}</span></div>
          <div class="setup-row"><span class="k">${t('Databases', '数据库')}</span><span class="vv">${dbs}</span></div>
        </div>
        ${summary.mappedAll == null ? '' : `<div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Dictionary mappings', '字典映射')}</div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('In all ' + dbs + ' databases', dbs + ' 个库都有映射')}</span><span class="mono">${summary.mappedAll}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('In some databases', '部分库有映射')}</span><span class="mono">${summary.mappedPartial}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('No mapping declared', '未声明映射')}</span><span class="mono">${summary.unmapped}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('Of which derived from inputs', '其中由输入计算')}</span><span class="mono">${summary.derived}</span></div>
        </div>`}
      </div>`;
    },
    render() {
      const cat = C();
      // The retired Data map's links open the study-type view (app.js).
      if (window.__euDictView) { dictView = window.__euDictView; window.__euDictView = null; }
      else if (hashView()) dictView = hashView();
      const studies = dictView === 'studies' && window.EU_DICT_STUDIES;
      const tab = (view, en, zh) => `<button type="button" role="tab" class="tab${dictView === view ? ' active' : ''}" aria-selected="${dictView === view}" data-dict-view="${view}">${t(en, zh)}</button>`;
      const head = `
      <div class="page-head" style="margin-bottom:16px;">
        <div class="eyebrow">${t('Data', '数据')}</div>
        <h1 style="margin-top:6px;">${t('Data dictionary', '数据字典')}</h1>
        ${instanceDbs() ? '' : `<p class="lead">${t(cat.totalConcepts + ' concepts across ' + cat.groups.length + ' feature groups — abbreviations, full names, units, and database coverage from the EasyICU concept dictionary.', cat.totalConcepts + ' 个概念，分属 ' + cat.groups.length + ' 个特征分组：缩写、全称、单位，以及 EasyICU 字典中的数据库覆盖。')}</p>`}
      </div>
      ${window.EU_DICT_STUDIES ? `<div class="tabs dict-views" role="tablist" aria-label="${t('Dictionary views', '字典视图')}">${tab('concepts', 'Variables', '变量')}${tab('studies', 'Study types', '研究类型')}</div>` : ''}`;
      if (studies) return head + window.EU_DICT_STUDIES.html();
      const rows = matchedRows();
      return `${head}

      ${instanceDbs() ? '' : `<div class="dict-catalog-note">
        ${icon('help', 14)}
        <span>${t('These are declared dictionary mappings, not proof of equivalent clinical meaning, exact research capability, or coverage in the selected data.', '这一列是字典声明的映射，不代表各库临床口径相同、具备精确研究能力或当前所选数据实际覆盖。')}</span>
      </div>`}

      <div class="dict-search">
        <span class="ds-ico">${icon('search', 16)}</span>
        <input id="dictSearchInput" type="text" placeholder="${t('Search by code, name or description… (e.g. hr, lactate, 乳酸)', '按代码、名称或说明搜索… (如 hr、lactate、乳酸)')}" value="${dictSearch.replace(/"/g, '&quot;')}" autocomplete="off" spellcheck="false" />
        <button class="ds-clear" data-dict-clear style="${dictSearch ? '' : 'display:none;'}">${icon('close', 14)}</button>
      </div>

      ${catChips()}

      <div id="dictResults">${resultsHtml(rows)}</div>
      <div id="dictLineage">${lineagePanel()}</div>
      <div class="nextbar mt-16">
        <div class="nb-ico">${icon('arrow', 16)}</div>
        <div class="grow"><div class="nb-t">${t('Found the concepts you need?', '找到需要的概念了？')}</div><div class="nb-d">${t('These concepts become columns of your dataset — select their modules in Data Extraction.', '这些概念会成为你数据表的列 —— 到「数据抽取」勾选它们所属的模块。')}</div></div>
        <button class="btn primary" data-nav="extraction">${icon('extract', 13)} ${t('Open Data Extraction', '打开数据抽取')}</button>
      </div>`;
    },
    afterRender(root) {
      root.querySelectorAll('[data-dict-view]').forEach(b => b.addEventListener('click', () => {
        if (dictView !== b.dataset.dictView) showView(b.dataset.dictView);
      }));
      if (dictView === 'studies' && window.EU_DICT_STUDIES) { window.EU_DICT_STUDIES.bind(root); return; }
      const input = root.querySelector('#dictSearchInput');
      const clr = root.querySelector('[data-dict-clear]');
      // Never replace the search input from its own input event: a full
      // re-render destroys the IME composition buffer mid-pinyin and resets
      // the caret. Keep the input node stable and repaint only the results.
      const paintResults = () => {
        const res = document.getElementById('dictResults');
        if (res) res.innerHTML = resultsHtml(matchedRows());
        if (clr) clr.style.display = dictSearch ? '' : 'none';
      };
      const paintLineage = () => {
        const panel = document.getElementById('dictLineage');
        if (panel) panel.innerHTML = lineagePanel();
      };
      if (input) {
        input.addEventListener('input', () => {
          dictSearch = input.value;
          paintResults();
        });
      }
      if (clr) clr.addEventListener('click', () => {
        dictSearch = '';
        if (input) { input.value = ''; input.focus(); }
        paintResults();
      });
      root.querySelectorAll('[data-dict-cat]').forEach(b => b.addEventListener('click', () => {
        dictCat = b.dataset.dictCat; window.__euRender();
      }));
      root.addEventListener('click', async event => {
        if (event.target.closest('[data-dict-internal]')) { dictShowInternal = !dictShowInternal; window.__euRender(); return; }
        const dbButton = event.target.closest('[data-dict-lineage-db]');
        if (dbButton) { dictSelectedDb = dbButton.dataset.dictLineageDb; paintLineage(); return; }
        const conceptButton = event.target.closest('[data-dict-concept]');
        if (!conceptButton) return;
        const request = ++dictLineageRequest;
        // A second press on an open entry folds its variants away again.
        if (conceptButton.dataset.dictConcept === dictSelected && variantsOf(dictSelected).length) {
          dictSelected = null; dictLineage = null; dictLineageError = '';
          paintResults(); paintLineage();
          return;
        }
        dictSelected = conceptButton.dataset.dictConcept;
        const selected = dictSelected;
        dictSelectedDb = null;
        dictLineage = null;
        dictLineageError = '';
        // An unsupported outcome's panel is its reason; it has no lineage to fetch.
        if (coverageOf(dictSelected).kind === 'unsupported') { paintResults(); paintLineage(); return; }
        dictLineageLoading = true;
        paintResults(); paintLineage();
        try {
          const api = window.EU_API;
          if (!api || typeof api.loadConceptLineage !== 'function') throw new Error(L('Lineage API is unavailable.', '血缘 API 不可用。'));
          const lineage = await api.loadConceptLineage(selected);
          if (request !== dictLineageRequest) return;
          dictLineage = lineage;
        } catch (error) {
          if (request !== dictLineageRequest) return;
          // A listed concept the lineage owner does not declare (the SOFA-2
          // scores keep their own dictionary) has no raw-source mapping to show.
          if (error && error.code === 'concept_not_found') dictLineage = { concept: dictSelected, lanes: [] };
          else dictLineageError = String(error && error.message || error);
        } finally {
          if (request === dictLineageRequest) {
            dictLineageLoading = false;
            paintResults(); paintLineage();
          }
        }
      });
    },
  };
})();
