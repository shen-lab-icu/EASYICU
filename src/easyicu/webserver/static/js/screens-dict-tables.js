/* Owner: the Data Dictionary's source-table view — which raw tables and
   fields each variable is read from, one database at a time.
   The host owns every row (webserver/catalog_source_map.py,
   GET /api/catalog/source-map/{database}): declared dictionary provenance,
   never patient rows. This view lays the table → field → variable map out;
   it decides nothing itself. screens-dict.js hosts it and owns the page.
   Interface: window.EU_DICT_TABLES = { html(), bind(host), open(database, table) }. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function catalog() { return window.EU_CATALOG || {}; }
  function conceptName(key) {
    const row = (catalog().dict || {})[key];
    return row ? (window.EU_LANG === 'zh' ? row[1] : row[0]) : key;
  }
  function groupOf(key) {
    const groups = catalog().groupConcepts || {};
    return Object.keys(groups).find(group => (groups[group] || []).includes(key)) || '';
  }
  function groupName(key) {
    const row = (catalog().groups || []).find(group => group[0] === key);
    const name = row ? (window.EU_LANG === 'zh' ? row[2] : row[1]) : '';
    return String(name || tr('Other', '其他')).replace(/^[^\p{L}\p{N}]+/u, '') || tr('Other', '其他');
  }
  const ROLES = {
    selector_field: ['row selector', '行选择'], value_field: ['value', '值'],
    unit_field: ['unit', '单位'], time_field: ['time', '时间'],
  };
  const DERIVED = '~derived';
  const FUNCTIONS = '~functions';

  const state = { database: 'miiv', table: '', group: '', filter: '', maps: {}, loading: '', failed: '' };

  function current() { return state.maps[state.database] || null; }
  function selectedTable(map) {
    const tables = map.tables || [];
    if (state.table === DERIVED || state.table === FUNCTIONS) return null;
    return tables.find(table => table.table === state.table) || tables[0] || null;
  }

  function load(host) {
    const database = state.database;
    if (state.maps[database] || state.loading === database) return;
    if (typeof api().loadCatalogSourceMap !== 'function') { state.failed = tr('The source-table API is unavailable.', '来源表 API 不可用。'); return; }
    state.loading = database; state.failed = '';
    api().loadCatalogSourceMap(database).then(map => {
      state.maps[database] = map;
    }).catch(error => {
      state.failed = String((error && error.message) || error);
    }).finally(() => {
      if (state.loading === database) state.loading = '';
      repaint();
    });
  }

  function databaseTabs(map) {
    const list = (map && map.databases) || [{ database: state.database, label: state.database }];
    return `<div class="dtb-dbs" role="tablist" aria-label="${esc(tr('Database', '数据库'))}">${list.map(row => `
      <button type="button" role="tab" aria-selected="${row.database === state.database}" class="${row.database === state.database ? 'on' : ''}" data-dtb-db="${esc(row.database)}">${esc(row.label)}</button>`).join('')}</div>`;
  }

  function tableList(map) {
    const tables = map.tables || [];
    const most = Math.max(1, ...tables.map(table => table.concepts.length));
    const active = selectedTable(map);
    const item = (key, label, count, extra) => `
      <button type="button" class="dtb-table${extra}" data-dtb-table="${esc(key)}" aria-pressed="${extra.includes(' on')}">
        <span class="dtb-table-name">${label}</span><span class="dtb-table-n">${count}</span>
        ${key === DERIVED || key === FUNCTIONS ? '' : `<i style="--dtb-share:${Math.round((count / most) * 100)}%"></i>`}
      </button>`;
    return `<nav class="dtb-tables" aria-label="${esc(tr('Source tables', '来源表'))}">
      ${tables.map(table => item(table.table, `<span class="mono">${esc(table.table)}</span>`, table.concepts.length,
        active && active.table === table.table ? ' on' : '')).join('')}
      ${(map.derived || []).length ? item(DERIVED, esc(tr('Derived from other variables', '由其他变量计算')), map.derived.length, state.table === DERIVED ? ' on is-derived' : ' is-derived') : ''}
      ${(map.functions || []).length ? item(FUNCTIONS, esc(tr('Computed from the stay record', '由住院记录计算')), map.functions.length, state.table === FUNCTIONS ? ' on is-derived' : ' is-derived') : ''}
    </nav>`;
  }

  // Table → value field → variable group, as weighted bands. Every band is a
  // count of declared rows; nothing is measured from data.
  function flow(table) {
    const rows = table.rows || [];
    const fields = new Map();
    const groups = new Map();
    const links = new Map();
    const seen = new Set();
    for (const row of rows) {
      const fieldKey = row.value_field || tr('(callback output)', '（回调输出）');
      // A variable with two rows through the same field counts once.
      if (seen.has(`${fieldKey}\u0001${row.concept}`)) continue;
      seen.add(`${fieldKey}\u0001${row.concept}`);
      const group = groupOf(row.concept);
      fields.set(fieldKey, (fields.get(fieldKey) || 0) + 1);
      groups.set(group, (groups.get(group) || 0) + 1);
      const key = `${fieldKey}\u0001${group}`;
      links.set(key, (links.get(key) || 0) + 1);
    }
    const total = [...fields.values()].reduce((sum, n) => sum + n, 0) || 1;
    const groupList = [...groups.entries()].sort((a, b) => b[1] - a[1]);
    const fieldList = [...fields.entries()].sort((a, b) => b[1] - a[1]);
    const gap = 6, unit = Math.max(4, Math.min(14, 300 / total));
    const columnHeight = list => list.reduce((sum, [, n]) => sum + Math.max(16, n * unit) + gap, -gap);
    const height = Math.max(columnHeight(fieldList), columnHeight(groupList), Math.max(16, total * unit)) + 8;
    const width = 760, x0 = 8, x1 = 300, x2 = 520, nodeW = 10;
    const place = list => { let y = 4; return new Map(list.map(([key, n]) => { const h = Math.max(16, n * unit); const box = { y, h, n, used: 0, usedIn: 0 }; y += h + gap; return [key, box]; })); };
    const fieldBox = place(fieldList), groupBox = place(groupList);
    const tableBox = { y: (height - Math.max(16, total * unit)) / 2, h: Math.max(16, total * unit), used: 0 };
    const palette = window.EU_PALETTE;
    const colorOf = new Map(groupList.map(([key], index) => [key, palette ? palette.color(index) : '#6b8f8a']));
    const band = (xa, ya, ha, xb, yb, hb, color, title, attrs) => {
      const mid = (xa + xb) / 2;
      return `<path d="M${xa},${ya} C${mid},${ya} ${mid},${yb} ${xb},${yb} L${xb},${yb + hb} C${mid},${yb + hb} ${mid},${ya + ha} ${xa},${ya + ha} Z" fill="${color}" ${attrs || ''}><title>${esc(title)}</title></path>`;
    };
    let paths = '';
    for (const [field, box] of fieldBox) {
      paths += band(x0 + nodeW, tableBox.y + tableBox.used, tableBox.h * (box.n / total), x1, box.y, box.h, 'var(--dtb-band)',
        `${table.table} → ${field}: ${box.n}`, 'class="dtb-band"');
      tableBox.used += tableBox.h * (box.n / total);
    }
    for (const [key, n] of [...links.entries()].sort((a, b) => b[1] - a[1])) {
      const [field, group] = key.split('\u0001');
      const fb = fieldBox.get(field), gb = groupBox.get(group);
      const ha = fb.h * (n / fb.n), hb = gb.h * (n / gb.n);
      const dim = state.group && state.group !== group ? ' is-dim' : '';
      paths += band(x1 + nodeW, fb.y + fb.usedIn, ha, x2, gb.y + gb.used, hb, colorOf.get(group),
        `${field} → ${groupName(group)}: ${n}`, `class="dtb-band is-group${dim}" data-dtb-group="${esc(group)}"`);
      fb.usedIn += ha; gb.used += hb;
    }
    const label = (x, y, text, anchor, cls) => `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls || ''}">${esc(text)}</text>`;
    let nodes = `<rect x="${x0}" y="${tableBox.y}" width="${nodeW}" height="${tableBox.h}" class="dtb-node"/>`;
    nodes += label(x0 + nodeW + 6, tableBox.y - 4 < 10 ? 12 : tableBox.y - 4, table.table, 'start', 'dtb-t mono');
    for (const [field, box] of fieldBox) {
      nodes += `<rect x="${x1}" y="${box.y}" width="${nodeW}" height="${box.h}" class="dtb-node"/>`;
      nodes += label(x1 - 6, box.y + box.h / 2 + 4, `${field} · ${box.n}`, 'end', 'dtb-f mono');
    }
    for (const [group, box] of groupBox) {
      const dim = state.group && state.group !== group ? ' is-dim' : '';
      nodes += `<rect x="${x2}" y="${box.y}" width="${nodeW}" height="${box.h}" fill="${colorOf.get(group)}" class="dtb-node is-group${dim}" data-dtb-group="${esc(group)}"/>`;
      nodes += `<text x="${x2 + nodeW + 8}" y="${box.y + box.h / 2 + 4}" class="dtb-g${dim}" data-dtb-group="${esc(group)}">${esc(groupName(group))} · ${box.n}</text>`;
    }
    return `<figure class="dtb-flow" data-dtb-flow>
      <figcaption>${esc(tr('Table → value field → variable group (number of variables). Press a group to filter the list below.', '表 → 值字段 → 变量分组（变量数）。点击分组可筛选下方列表。'))}</figcaption>
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(tr(`${table.table}: fields and variable groups`, `${table.table}：字段与变量分组`))}" preserveAspectRatio="xMinYMin meet" style="max-height:${Math.min(height, 560)}px">${paths}${nodes}</svg>
    </figure>`;
  }

  function columnStrip(table) {
    const roles = new Map((table.used_fields || []).map(row => [row.field, row.roles]));
    const columns = table.columns && table.columns.length ? table.columns : [...roles.keys()];
    const chip = column => {
      const used = roles.get(column);
      const title = used ? used.map(role => tr(...ROLES[role])).join(' · ') : tr('not read by the dictionary', '字典未读取');
      return `<span class="dtb-col${used ? ` is-used is-${esc(used[0])}` : ''}" title="${esc(title)}">${esc(column)}${used ? `<small>${esc(used.map(role => tr(...ROLES[role])).join(' · '))}</small>` : ''}</span>`;
    };
    const usedCount = columns.filter(column => roles.has(column)).length;
    return `<div class="dtb-cols">
      <div class="dtb-sub">${esc(tr(`Columns: ${usedCount} of ${columns.length} read by the dictionary`, `字段：字典读取 ${columns.length} 个中的 ${usedCount} 个`))}${table.declared_in_profile ? '' : ` · ${esc(tr('table not in the packaged schema', '打包结构中无此表'))}`}</div>
      <div class="dtb-col-list">${columns.map(chip).join('')}</div>
    </div>`;
  }

  function selection(row) {
    const values = Array.isArray(row.selector_values) ? row.selector_values : [];
    if (!row.selector_field || !values.length) return `<span class="dtb-muted">${esc(tr('all rows', '全部行'))}</span>`;
    const shown = values.slice(0, 3).join(', ');
    return `<span class="mono" title="${esc(values.join(', '))}">${esc(row.selector_field)} ∈ {${esc(shown)}${values.length > 3 ? ', …' : ''}}</span>${values.length > 3 ? ` <span class="dtb-muted">${esc(tr(`${values.length} codes`, `${values.length} 个代码`))}</span>` : ''}`;
  }
  function filterMatch(concept) {
    const query = state.filter.trim().toLowerCase();
    return !query || concept.toLowerCase().includes(query) || conceptName(concept).toLowerCase().includes(query);
  }
  function conceptButton(concept) {
    return `<button type="button" class="dtb-concept" data-dtb-concept="${esc(concept)}"><b>${esc(conceptName(concept))}</b> <span class="mono">${esc(concept)}</span></button>`;
  }

  function rowsTable(table) {
    const rows = (table.rows || []).filter(row => (!state.group || groupOf(row.concept) === state.group) && filterMatch(row.concept));
    const field = value => value ? `<span class="mono">${esc(value)}</span>` : `<span class="dtb-muted">${esc(tr('not declared', '未声明'))}</span>`;
    return `<div class="dtb-scroll"><table class="dtb-rows">
      <thead><tr><th>${esc(tr('Variable', '变量'))}</th><th>${esc(tr('Row selection', '行选择'))}</th><th>${esc(tr('Value', '值'))}</th><th>${esc(tr('Unit', '单位'))}</th><th>${esc(tr('Time', '时间'))}</th><th>${esc(tr('Transformation', '转换'))}</th></tr></thead>
      <tbody>${rows.map(row => `<tr><td>${conceptButton(row.concept)}</td><td>${selection(row)}</td><td>${field(row.value_field)}</td><td>${field(row.unit_field)}</td><td>${field(row.time_field)}</td><td>${row.callback ? `<span class="mono">${esc(row.callback)}</span>` : '<span class="dtb-muted">—</span>'}</td></tr>`).join('')
        || `<tr><td colspan="6" class="dtb-muted">${esc(tr('No variable matches.', '没有匹配的变量。'))}</td></tr>`}</tbody>
    </table></div>`;
  }

  function derivedView(map) {
    const rows = (map.derived || []).filter(row => filterMatch(row.concept));
    return `<div class="dtb-detail-head"><h3>${esc(tr('Derived from other variables', '由其他变量计算'))}</h3><p>${esc(tr('These variables read no table themselves; their callback computes them from the input variables.', '这些变量本身不读表，由回调根据输入变量计算。'))}</p></div>
      <div class="dtb-scroll"><table class="dtb-rows"><thead><tr><th>${esc(tr('Variable', '变量'))}</th><th>${esc(tr('Computed from', '输入变量'))}</th><th>${esc(tr('Transformation', '转换'))}</th></tr></thead>
      <tbody>${rows.map(row => `<tr><td>${conceptButton(row.concept)}</td><td><div class="dtb-inputs">${row.inputs.map(input => `<button type="button" class="dtb-input" data-dtb-concept="${esc(input)}">${esc(conceptName(input))}</button>`).join('')}</div></td><td>${row.callback ? `<span class="mono">${esc(row.callback)}</span>` : '<span class="dtb-muted">—</span>'}</td></tr>`).join('')}</tbody></table></div>`;
  }
  function functionView(map) {
    const rows = (map.functions || []).filter(row => filterMatch(row.concept));
    return `<div class="dtb-detail-head"><h3>${esc(tr('Computed from the stay record', '由住院记录计算'))}</h3><p>${esc(tr('The dictionary names no single table for these; the callback reads the stay record itself, for example the admission and discharge times.', '字典没有为这些变量指定单一的表；回调直接读取住院记录，例如入院和出院时间。'))}</p></div>
      <div class="dtb-scroll"><table class="dtb-rows"><thead><tr><th>${esc(tr('Variable', '变量'))}</th><th>${esc(tr('Transformation', '转换'))}</th></tr></thead>
      <tbody>${rows.map(row => `<tr><td>${conceptButton(row.concept)}</td><td><span class="mono">${esc(row.callback || '—')}</span></td></tr>`).join('')}</tbody></table></div>`;
  }

  function detail(map) {
    if (state.table === DERIVED) return derivedView(map);
    if (state.table === FUNCTIONS) return functionView(map);
    const table = selectedTable(map);
    if (!table) return `<p class="dtb-muted">${esc(tr('No source table is declared for this database.', '这个数据库没有声明来源表。'))}</p>`;
    return `<div class="dtb-detail-head"><h3><span class="mono">${esc(table.table)}</span></h3>
        <p>${esc(tr(`${table.concepts.length} variables read from this table`, `从这张表读取 ${table.concepts.length} 个变量`))}${state.group ? ` · ${esc(groupName(state.group))} <button type="button" class="btn sm ghost" data-dtb-group="">${esc(tr('Show all groups', '显示全部分组'))}</button>` : ''}</p></div>
      ${flow(table)}
      ${columnStrip(table)}
      ${rowsTable(table)}`;
  }

  function body() {
    const map = current();
    if (state.failed && !map) return `<div class="note warn"><div class="body"><div class="t">${esc(tr('Source tables unavailable', '来源表暂不可用'))}</div><div class="d">${esc(state.failed)}</div></div></div>`;
    if (!map) return `<div class="state loading"><span class="spin"></span><div class="t">${esc(tr('Loading declared source tables…', '正在加载声明的来源表…'))}</div></div>`;
    const mapped = new Set((map.tables || []).flatMap(table => table.concepts)).size;
    return `<p class="dtb-summary">${esc(tr(
      `${map.label}: ${(map.tables || []).length} source tables · ${mapped} variables read from them · ${(map.derived || []).length} derived · ${(map.functions || []).length} from the stay record`,
      `${map.label}：${(map.tables || []).length} 张来源表 · 从中读取 ${mapped} 个变量 · ${(map.derived || []).length} 个派生 · ${(map.functions || []).length} 个由住院记录计算`))}</p>
      <div class="dtb-grid">
        ${tableList(map)}
        <section class="dtb-detail" data-dtb-detail>
          <label class="dtb-filter">${window.icon ? window.icon('search', 14) : ''}<input type="search" data-dtb-filter value="${esc(state.filter)}" placeholder="${esc(tr('Filter variables in this view…', '筛选当前视图中的变量…'))}" autocomplete="off" spellcheck="false"/></label>
          <div data-dtb-detail-body>${detail(map)}</div>
        </section>
      </div>`;
  }

  function html() {
    return `<div class="dtb" data-dict-tables>
      <p class="dtb-note">${esc(tr('Declared dictionary rows only: which table and field each variable is read from, and how its rows are selected. Nothing here is read from patient data.', '只展示字典声明的映射：每个变量从哪张表、哪个字段读取，以及如何筛选行。这里的内容都不来自患者数据。'))}</p>
      ${databaseTabs(current())}
      <div data-dtb-body>${body()}</div>
    </div>`;
  }

  // The page re-renders while a map loads; paint whichever view is live.
  function liveRoot() { return typeof document !== 'undefined' ? document.querySelector('[data-dict-tables]') : null; }
  function repaint() {
    const root = liveRoot();
    if (!root) return;
    const tabs = root.querySelector('.dtb-dbs');
    if (tabs) tabs.outerHTML = databaseTabs(current());
    const target = root.querySelector('[data-dtb-body]');
    if (target) target.innerHTML = body();
  }
  function repaintDetail() {
    const map = current();
    const root = liveRoot();
    const target = root && root.querySelector('[data-dtb-detail-body]');
    if (map && target) target.innerHTML = detail(map);
  }

  let boundRoot = null;
  function bind(host) {
    load(host);
    const root = host.querySelector('[data-dict-tables]');
    if (!root || boundRoot === root) return;
    boundRoot = root;
    root.addEventListener('click', event => {
      const db = event.target.closest('[data-dtb-db]');
      if (db) { state.database = db.dataset.dtbDb; state.table = ''; state.group = ''; repaint(); load(host); return; }
      const table = event.target.closest('[data-dtb-table]');
      if (table) { state.table = table.dataset.dtbTable; state.group = ''; repaint(); return; }
      const group = event.target.closest('[data-dtb-group]');
      if (group) { const key = group.getAttribute('data-dtb-group'); state.group = state.group === key ? '' : key; repaintDetail(); return; }
      const concept = event.target.closest('[data-dtb-concept]');
      if (concept && typeof window.EU_DICT_OPEN_CONCEPT === 'function') window.EU_DICT_OPEN_CONCEPT(concept.dataset.dtbConcept, state.database);
    });
    root.addEventListener('input', event => {
      if (!event.target.matches('[data-dtb-filter]')) return;
      state.filter = event.target.value;
      repaintDetail();
    });
  }

  // The variable view's lineage opens one database's table here.
  function open(database, table) {
    if (database) state.database = database;
    state.table = table || '';
    state.group = '';
  }

  window.EU_DICT_TABLES = Object.freeze({ html, bind, open });
})();
