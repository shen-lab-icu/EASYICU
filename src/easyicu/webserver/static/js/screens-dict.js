/* Screen: Data Dictionary — the real concept catalog, searchable.
   Native catalog browser with search + category
   browser over the 19 concept groups, with units and per-database coverage). */
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
  function groupNameOf(key) { const g = C().groups.find(g => g[0] === key); return g ? (window.EU_LANG === 'zh' ? g[2] : g[1]) : key; }

  function supportedDbCount() { return (C().supportedDbs || []).length || 6; }

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

  function covBadge(k) {
    const meta = coverageOf(k);
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
    const groupsToScan = dictCat === 'all' ? cat.groups.map(g => g[0]) : [dictCat];
    const rows = [];
    groupsToScan.forEach(gk => {
      (cat.groupConcepts[gk] || []).forEach(k => {
        if (!cat.dict[k]) return;
        if (q) {
          const hay = `${k} ${nameOf(k)} ${zhOf(k)} ${descOf(k)}`.toLowerCase();
          if (!hay.includes(q)) return;
        }
        rows.push({ k, group: gk });
      });
    });
    return rows;
  }

  function resultsHtml(rows) {
    return `<div class="dict-resultline">${t(rows.length + ' concept' + (rows.length === 1 ? '' : 's'), rows.length + ' 个概念')}${dictCat !== 'all' ? ` · ${groupNameOf(dictCat)}` : ''}${dictSearch ? ` · "${dictSearch.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]))}"` : ''}</div>${tableHtml(rows)}`;
  }
  function tableHtml(rows) {
    if (!rows.length) return `<div class="dict-empty">${icon('search', 22)}<div>${L('No matching concepts.', '未找到匹配的概念。')}</div></div>`;
    const coverageLabel = L('Database coverage', '数据库覆盖');
    const head = `
      <div class="dict-row dict-head">
        <span class="dc-code">${L('Code', '代码')}</span>
        <span class="dc-name">${L('Concept', '概念')}</span>
        <span class="dc-unit">${L('Unit', '单位')}</span>
        <span class="dc-cov">${coverageLabel}</span>
        ${dictCat === 'all' ? `<span class="dc-cat">${L('Group', '分组')}</span>` : ''}
      </div>`;
    const body = rows.map(({ k, group }) => {
      const d = descOf(k);
      return `
      <button type="button" class="dict-row dict-concept-row ${dictSelected === k ? 'selected' : ''}" data-dict-concept="${html(k)}" aria-pressed="${dictSelected === k ? 'true' : 'false'}">
        <span class="dc-code mono">${k}</span>
        <span class="dc-name"><span class="dn-t">${window.EU_LANG === 'zh' ? zhOf(k) : nameOf(k)}</span>${d ? `<span class="dn-d">${d}</span>` : ''}</span>
        <span class="dc-unit mono">${unitOf(k)}</span>
        <span class="dc-cov">${covBadge(k)}</span>
        ${dictCat === 'all' ? `<span class="dc-cat">${groupNameOf(group)}</span>` : ''}
      </button>`;
    }).join('');
    return `<div class="dict-table">${head}${body}</div>`;
  }

  function field(value) { return value == null || value === '' ? L('not declared', '未声明') : String(value); }
  function selector(mapping) {
    const values = Array.isArray(mapping.selector_values) ? mapping.selector_values : [];
    if (!mapping.selector_field || !values.length) return L('no row selector declared', '未声明行选择条件');
    const shown = values.slice(0, 4).join(', ');
    return `${mapping.selector_field} ∈ {${shown}${values.length > 4 ? ', …' : ''}}`;
  }
  function mappingSummary(lane) {
    return (lane.mappings || []).map(mapping => `${field(mapping.table)}.${field(mapping.value_field)}`).join(' + ');
  }
  function lineagePanel() {
    if (!dictSelected) return '';
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
    const steps = [
      [L('Source table', '来源表'), (active.mappings || []).map(mapping => field(mapping.table)).join(' + ')],
      [L('Row selection', '行选择'), (active.mappings || []).map(selector).join('；')],
      [L('Raw fields', '原始字段'), `${L('value', '值')} ${field(primary.value_field)} · ${L('unit', '单位')} ${field(primary.unit_field)} · ${L('time', '时间')} ${field(primary.time_field)}`],
      [L('Transformation', '转换处理'), callbacks.length ? callbacks.join('；') : L('No dictionary callback declared; identity cannot be assumed beyond the declared fields.', '字典未声明回调；除已声明字段外，不假定额外处理。')],
      [L('Bounds', '边界'), `${canonical.minimum == null ? '−∞' : canonical.minimum} – ${canonical.maximum == null ? '+∞' : canonical.maximum}`],
      [L('Canonical output', '标准输出'), `${dictLineage.concept}${canonical.unit ? ` · ${canonical.unit}` : ''}`],
    ];
    return `<section class="dict-lineage" data-dict-lineage>
      <div class="dict-lineage-head"><div><div class="eyebrow">${L('Feature provenance', '特征血缘')}</div><h2>${html(window.EU_LANG === 'zh' ? zhOf(dictLineage.concept) : dictLineage.name)} <span class="mono">${html(dictLineage.concept)}</span></h2><p>${L('Declared catalog mappings only. Fields marked “not declared” are left explicit instead of guessed.', '仅展示字典声明的映射；未声明的字段保持明确空缺，不做猜测。')}</p></div><span class="pill ok">${lanes.length} ${L('databases', '个数据库')}</span></div>
      <div class="dict-lineage-tabs" role="tablist">${lanes.map(lane=>`<button type="button" class="${lane.database===active.database?'active':''}" data-dict-lineage-db="${html(lane.database)}">${html(lane.label)}</button>`).join('')}</div>
      <div class="dict-audit-track" data-dict-audit-track>${steps.map(([label,value],index)=>`<div class="dict-audit-step"><i>${index+1}</i><div><b>${html(label)}</b><span>${html(value)}</span></div></div>`).join('')}</div>
      <div class="dict-lineage-lanes" data-dict-lineage-lanes>${lanes.map((lane,index)=>`<article style="--dict-lane-color:${window.EU_PALETTE.color(index)}"><div><i></i><b>${html(lane.label)}</b><span>${html(lane.database)}</span></div><strong>${html(mappingSummary(lane))}</strong><em>${html(Array.from(new Set((lane.mappings||[]).map(mapping=>mapping.callback).filter(Boolean))).join('；') || L('no callback declared', '未声明回调'))}</em><small>→ ${html(dictLineage.concept)}${canonical.unit?` · ${html(canonical.unit)}`:''}</small></article>`).join('')}</div>
    </section>`;
  }

  function catChips() {
    const cat = C();
    const chip = (key, label, n) => `<button class="dict-chip ${dictCat === key ? 'on' : ''}" data-dict-cat="${key}">${label}${n != null ? ` <span class="dch-n">${n}</span>` : ''}</button>`;
    return `<div class="dict-chips">
      ${chip('all', L('All', '全部'), cat.totalConcepts)}
      ${cat.groups.map(g => chip(g[0], window.EU_LANG === 'zh' ? g[2] : g[1], cat.groupConcepts[g[0]].length)).join('')}
    </div>`;
  }

  S.dictionary = {
    section: 'dictionary', nav: 'dictionary',
    get crumbs() { return [t('Home', '首页'), t('Data Dictionary', '数据字典')]; },
    actionHtml: '',
    rail() {
      const cat = C();
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
        <div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Dictionary database coverage', '字典数据库覆盖')}</div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('All ' + dbs + ' databases', '全部 ' + dbs + ' 库')}</span><span class="mono">${summary.auditedAll || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t((dbs - 1) + ' databases', (dbs - 1) + ' 库')}</span><span class="mono">${summary.auditedFive || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('≤ ' + Math.max(1, dbs - 2) + ' databases', '≤ ' + Math.max(1, dbs - 2) + ' 库')}</span><span class="mono">${summary.auditedPartial || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('Derived / rule outputs', '派生/规则输出')}</span><span class="mono">${summary.derived || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('No catalog audit', '无字典审计')}</span><span class="mono">${summary.notAudited || 0}</span></div>
        </div>
      </div>`;
    },
    render() {
      const cat = C();
      const rows = matchedRows();
      return `
      <div class="page-head" style="margin-bottom:16px;">
        <div class="eyebrow">${t('Data dictionary · 数据字典', '数据字典 · Data dictionary')}</div>
        <h1 style="margin-top:6px;">${t('Data dictionary', '数据字典')}</h1>
        <p class="lead">${t(cat.totalConcepts + ' concepts across ' + cat.groups.length + ' feature groups — abbreviations, full names, units, and database coverage from the EasyICU concept dictionary.', cat.totalConcepts + ' 个概念,分属 ' + cat.groups.length + ' 个特征分组 —— 缩写、全称、单位,以及 EasyICU 字典中的数据库覆盖。')}</p>
      </div>

      <div class="dict-catalog-note">
        ${icon('help', 14)}
        <span>${t('This column counts how many supported ICU databases have a dictionary mapping for each feature. It is independent of the currently registered local export.', '这一列统计字典中每个特征覆盖了多少个支持的 ICU 数据库,与当前注册的本地导出无关。')}</span>
      </div>

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
        const dbButton = event.target.closest('[data-dict-lineage-db]');
        if (dbButton) { dictSelectedDb = dbButton.dataset.dictLineageDb; paintLineage(); return; }
        const conceptButton = event.target.closest('[data-dict-concept]');
        if (!conceptButton) return;
        dictSelected = conceptButton.dataset.dictConcept;
        dictSelectedDb = null;
        dictLineage = null;
        dictLineageError = '';
        dictLineageLoading = true;
        paintResults(); paintLineage();
        try {
          const api = window.EU_API;
          if (!api || typeof api.loadConceptLineage !== 'function') throw new Error(L('Lineage API is unavailable.', '血缘 API 不可用。'));
          dictLineage = await api.loadConceptLineage(dictSelected);
        } catch (error) {
          dictLineageError = String(error && error.message || error);
        } finally {
          dictLineageLoading = false;
          paintResults(); paintLineage();
        }
      });
    },
  };
})();
