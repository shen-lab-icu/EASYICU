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
