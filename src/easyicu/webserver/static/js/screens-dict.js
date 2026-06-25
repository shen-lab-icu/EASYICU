/* Screen: Data Dictionary — the real concept catalog, searchable.
   Native catalog browser with search + category
   browser over the 19 concept groups, with units and per-database coverage). */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  const C = () => window.EU_CATALOG;

  let dictSearch = '';
  let dictCat = 'all';   // 'all' | group key

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
    const active = C().activeExportCoverage || {};
    if (active.status === 'ready') {
      const meta = (active.concepts || {})[k];
      if (meta) return meta;
      return { kind: 'not_in_active_export', databases: null, basis: 'missing_from_active_export', active: true };
    }
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
    if (meta.kind === 'active_export' || meta.kind === 'active_event') {
      const pct = Number(meta.coverage_pct);
      const cls = Number.isFinite(pct) ? (pct >= 80 ? 'hi' : pct >= 50 ? 'mid' : 'lo') : 'none';
      const label = Number.isFinite(pct) ? `${pct}%` : L('available', '已包含');
      const prefix = meta.kind === 'active_event' ? L('event ', '事件 ') : '';
      return `<span class="cov-badge ${cls}" title="${L('Active export aggregate coverage: ' + (meta.observed_entities ?? 'n/a') + '/' + (meta.denominator ?? 'n/a') + ' entities from ' + (meta.module || 'module') + '.', '当前导出聚合覆盖: ' + (meta.observed_entities ?? 'n/a') + '/' + (meta.denominator ?? 'n/a') + ' 个实体,来自 ' + (meta.module || '模块') + '。')}"><span class="cov-n">${prefix}${label}</span></span>`;
    }
    if (meta.kind === 'active_unreadable') {
      return `<span class="cov-badge unaudited" title="${L('The column exists in the active export, but aggregate coverage could not be computed without a readable stay_id column.', '该列存在于当前导出,但缺少可读 stay_id 列,无法计算聚合覆盖。')}">${L('present, no coverage', '已包含,无覆盖')}</span>`;
    }
    if (meta.kind === 'not_in_active_export') {
      return `<span class="cov-badge unaudited" title="${L('This concept is in the EasyICU dictionary but is not included in the active registered export.', '该概念存在于 EasyICU 字典,但不在当前注册导出中。')}">${L('not extracted', '未提取')}</span>`;
    }
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
      <span class="cov-n">${n}/${dbs}</span></span>`;
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

  function tableHtml(rows) {
    if (!rows.length) return `<div class="dict-empty">${icon('search', 22)}<div>${L('No matching concepts.', '未找到匹配的概念。')}</div></div>`;
    const active = C().activeExportCoverage || {};
    const coverageLabel = active.status === 'ready' ? L('Active export coverage', '当前导出覆盖') : L('Mapping audit', '映射审计');
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
      <div class="dict-row">
        <span class="dc-code mono">${k}</span>
        <span class="dc-name"><span class="dn-t">${window.EU_LANG === 'zh' ? zhOf(k) : nameOf(k)}</span>${d ? `<span class="dn-d">${d}</span>` : ''}</span>
        <span class="dc-unit mono">${unitOf(k)}</span>
        <span class="dc-cov">${covBadge(k)}</span>
        ${dictCat === 'all' ? `<span class="dc-cat">${groupNameOf(group)}</span>` : ''}
      </div>`;
    }).join('');
    return `<div class="dict-table">${head}${body}</div>`;
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
      const active = cat.activeExportCoverage || {};
      const activeSummary = active.summary || {};
      const dbs = summary.supportedDatabases || supportedDbCount();
      const activeReady = active.status === 'ready';
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Catalog', '目录')}</span></div>
        <div class="col gap-6" style="font-size:12px;">
          <div class="setup-row"><span class="k">${t('Groups', '分组')}</span><span class="vv">${cat.groups.length}</span></div>
          <div class="setup-row"><span class="k">${t('Concepts', '概念')}</span><span class="vv">${cat.totalConcepts}</span></div>
          <div class="setup-row"><span class="k">${t('Databases', '数据库')}</span><span class="vv">${dbs}</span></div>
        </div>
        ${activeReady ? `
        <div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Active export coverage', '当前导出覆盖')}</div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('Source', '来源')}</span><span class="mono">${html((active.source || {}).label || 'active')}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('Included concepts', '已提取概念')}</span><span class="mono">${activeSummary.included || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('≥80% coverage', '≥80% 覆盖')}</span><span class="mono">${activeSummary.high || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('50-79% coverage', '50-79% 覆盖')}</span><span class="mono">${activeSummary.medium || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('<50% coverage', '<50% 覆盖')}</span><span class="mono">${activeSummary.low || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('Not in active export', '当前未提取')}</span><span class="mono">${activeSummary.notInExport || 0}</span></div>
        </div>` : `
        <div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Catalog mapping audit', '字典映射审计')}</div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('All ' + dbs + ' databases', '全部 ' + dbs + ' 库')}</span><span class="mono">${summary.auditedAll || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t((dbs - 1) + ' databases', (dbs - 1) + ' 库')}</span><span class="mono">${summary.auditedFive || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('≤ ' + Math.max(1, dbs - 2) + ' databases', '≤ ' + Math.max(1, dbs - 2) + ' 库')}</span><span class="mono">${summary.auditedPartial || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('Derived / rule outputs', '派生/规则输出')}</span><span class="mono">${summary.derived || 0}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('No catalog audit', '无字典审计')}</span><span class="mono">${summary.notAudited || 0}</span></div>
        </div>`}
      </div>`;
    },
    render() {
      const cat = C();
      const rows = matchedRows();
      return `
      <div class="page-head" style="margin-bottom:16px;">
        <div class="eyebrow">${t('Data dictionary · 数据字典', '数据字典 · Data dictionary')}</div>
        <h1 style="margin-top:6px;">${t('Data dictionary', '数据字典')}</h1>
        <p class="lead">${t(cat.totalConcepts + ' concepts across ' + cat.groups.length + ' feature groups — abbreviations, full names, units, and active-export coverage when a local export is registered.', cat.totalConcepts + ' 个概念,分属 ' + cat.groups.length + ' 个特征分组 —— 缩写、全称、单位,以及注册本地导出后的真实覆盖。')}</p>
      </div>

      <div class="dict-catalog-note">
        ${icon('help', 14)}
        <span>${(cat.activeExportCoverage || {}).status === 'ready'
          ? t('This column uses the active registered export and reports aggregate entity coverage. Concepts not selected during extraction are labeled not extracted.', '这一列来自当前注册导出,显示聚合实体覆盖率。提取时未选择的概念标为未提取。')
          : t('No active registered export is available, so this page falls back to the static cross-database catalog mapping audit.', '当前没有可用的注册导出,因此本页回退显示静态跨库字典映射审计。')}</span>
      </div>

      <div class="dict-search">
        <span class="ds-ico">${icon('search', 16)}</span>
        <input id="dictSearchInput" type="text" placeholder="${t('Search by code, name or description… (e.g. hr, lactate, 乳酸)', '按代码、名称或说明搜索… (如 hr、lactate、乳酸)')}" value="${dictSearch.replace(/"/g, '&quot;')}" autocomplete="off" spellcheck="false" />
        ${dictSearch ? `<button class="ds-clear" data-dict-clear>${icon('close', 14)}</button>` : ''}
      </div>

      ${catChips()}

      <div class="dict-resultline">${t(rows.length + ' concept' + (rows.length === 1 ? '' : 's'), rows.length + ' 个概念')}${dictCat !== 'all' ? ` · ${groupNameOf(dictCat)}` : ''}${dictSearch ? ` · "${dictSearch}"` : ''}</div>
      ${tableHtml(rows)}`;
    },
    afterRender(root) {
      const input = root.querySelector('#dictSearchInput');
      if (input) {
        input.addEventListener('input', () => {
          dictSearch = input.value;
          // re-render just the result region for snappy typing
          window.__euRender();
          const again = document.querySelector('#dictSearchInput');
          if (again) { again.focus(); again.setSelectionRange(again.value.length, again.value.length); }
        });
      }
      const clr = root.querySelector('[data-dict-clear]');
      if (clr) clr.addEventListener('click', () => { dictSearch = ''; window.__euRender(); });
      root.querySelectorAll('[data-dict-cat]').forEach(b => b.addEventListener('click', () => {
        dictCat = b.dataset.dictCat; window.__euRender();
      }));
    },
  };
})();
