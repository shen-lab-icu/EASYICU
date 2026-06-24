/* Screen: Data Dictionary — the real concept catalog, searchable.
   Native catalog browser with search + category
   browser over the 19 concept groups, with units and per-database coverage). */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  const C = () => window.EU_CATALOG;

  let dictSearch = '';
  let dictCat = 'all';   // 'all' | group key

  function L(en, zh) { return t(en, zh); }
  function nameOf(k) { const d = C().dict[k]; return d ? d[0] : k; }
  function zhOf(k) { const d = C().dict[k]; return d ? d[1] : k; }
  function unitOf(k) { const d = C().dict[k]; return d && d[2] ? d[2] : '—'; }
  function descOf(k) { const d = C().desc[k]; return d ? (window.EU_LANG === 'zh' ? d[1] : d[0]) : ''; }
  function covOf(k) { return C().cov[k] || 0; }
  function groupNameOf(key) { const g = C().groups.find(g => g[0] === key); return g ? (window.EU_LANG === 'zh' ? g[2] : g[1]) : key; }

  function covBadge(n) {
    if (!n) return `<span class="cov-badge none" title="${L('coverage unknown', '覆盖未知')}">·</span>`;
    const cls = n >= 5 ? 'hi' : n >= 3 ? 'mid' : 'lo';
    return `<span class="cov-badge ${cls}" title="${L(n + ' of 6 databases', '6 个数据库中的 ' + n + ' 个')}">
      <span class="cov-dots">${[1,2,3,4,5,6].map(i => `<i class="${i <= n ? 'on' : ''}"></i>`).join('')}</span>
      <span class="cov-n">${n}/6</span></span>`;
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
    const head = `
      <div class="dict-row dict-head">
        <span class="dc-code">${L('Code', '代码')}</span>
        <span class="dc-name">${L('Concept', '概念')}</span>
        <span class="dc-unit">${L('Unit', '单位')}</span>
        <span class="dc-cov">${L('Databases', '数据库')}</span>
        ${dictCat === 'all' ? `<span class="dc-cat">${L('Group', '分组')}</span>` : ''}
      </div>`;
    const body = rows.map(({ k, group }) => {
      const d = descOf(k);
      return `
      <div class="dict-row">
        <span class="dc-code mono">${k}</span>
        <span class="dc-name"><span class="dn-t">${window.EU_LANG === 'zh' ? zhOf(k) : nameOf(k)}</span>${d ? `<span class="dn-d">${d}</span>` : ''}</span>
        <span class="dc-unit mono">${unitOf(k)}</span>
        <span class="dc-cov">${covBadge(covOf(k))}</span>
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
      const dbCov = {};
      Object.keys(cat.cov).forEach(k => { const n = cat.cov[k]; dbCov[n] = (dbCov[n] || 0) + 1; });
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">${t('Catalog', '目录')}</span></div>
        <div class="col gap-6" style="font-size:12px;">
          <div class="setup-row"><span class="k">${t('Groups', '分组')}</span><span class="vv">${cat.groups.length}</span></div>
          <div class="setup-row"><span class="k">${t('Concepts', '概念')}</span><span class="vv">${cat.totalConcepts}</span></div>
          <div class="setup-row"><span class="k">${t('Databases', '数据库')}</span><span class="vv">6</span></div>
        </div>
        <div class="eyebrow mt-16" style="margin-bottom:8px;">${t('Harmonization', '跨库统一')}</div>
        <div class="col gap-6" style="font-size:11.5px;color:var(--ink-3);">
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('All 6 databases', '全部 6 库')}</span><span class="mono">${Object.keys(cat.cov).filter(k => cat.cov[k] === 6).length}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('5 databases', '5 库')}</span><span class="mono">${Object.keys(cat.cov).filter(k => cat.cov[k] === 5).length}</span></div>
          <div class="row gap-6" style="justify-content:space-between;"><span>${t('≤ 4 databases', '≤ 4 库')}</span><span class="mono">${Object.keys(cat.cov).filter(k => cat.cov[k] > 0 && cat.cov[k] <= 4).length}</span></div>
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
        <p class="lead">${t(cat.totalConcepts + ' concepts across ' + cat.groups.length + ' feature groups — abbreviations, full names, units, and how many of the six ICU databases each is harmonized across.', cat.totalConcepts + ' 个概念,分属 ' + cat.groups.length + ' 个特征分组 —— 缩写、全称、单位,以及每个概念在六个 ICU 数据库中已统一的数量。')}</p>
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
