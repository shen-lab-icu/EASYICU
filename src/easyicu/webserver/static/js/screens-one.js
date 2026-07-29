/* screens-one.js — owner: the single-surface controller.
 *
 * There is one screen. The conversation is always the middle column; every
 * other destination is a panel the conversation summons into the right column
 * by rendering that screen's existing render() — no screen was rewritten to
 * become a panel, which is the whole point of hosting them this way.
 *
 * The summon contract is one attribute: [data-panel="<screenId>"]. Anything the
 * conversation emits carrying it opens that panel, so the assistant can offer
 * "看看这个病人" as a control without this file knowing what it means.
 */
(function () {
  const t = (en, zh) => (window.t ? window.t(en, zh) : en);
  const icon = (n, s) => (window.icon ? window.icon(n, s) : '');

  /* Routes that name a panel rather than a page. Everything else leaves the
     conversation alone with no panel open. */
  const PANEL_ROUTES = [
    'patient', 'cohort', 'crossdb', 'extraction',
    'ideas', 'agent', 'dictionary', 'states', 'settings', 'tutorial',
  ];

  /* Panels wide enough that a 480px column would misrepresent them. */
  const WIDE_PANELS = ['crossdb', 'agent', 'ideas', 'dictionary'];

  const QUICK = [
    { id: 'patient', label: ['Patients', '患者'] },
    { id: 'cohort', label: ['Cohort', '队列'] },
    { id: 'crossdb', label: ['Cross-database', '跨库'] },
    { id: 'extraction', label: ['Extract', '抽取'] },
    { id: 'agent', label: ['Projects', '研究项目'] },
    { id: 'ideas', label: ['Ideas', '想法'] },
  ];

  const L = (v) => (Array.isArray(v) ? t(v[0], v[1]) : v);

  let tasks = null;      /* null = not fetched yet, [] = fetched and empty */
  let tasksError = '';

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
    ));
  }

  function panelTitle(id) {
    const scr = window.SCREENS[id];
    if (scr && scr.crumbs && scr.crumbs.length) {
      const last = scr.crumbs[scr.crumbs.length - 1];
      return window.EU_CRUMB_LABEL ? window.EU_CRUMB_LABEL(last) : last;
    }
    const q = QUICK.find((x) => x.id === id);
    return q ? L(q.label) : id;
  }

  /* ---- task rail ---------------------------------------------------- */

  function dotClass(status) {
    if (status === 'running') return 'o-run';
    if (status === 'blocked' || status === 'error') return 'o-stop';
    if (status === 'paused' || status === 'waiting') return 'o-wait';
    return 'o-ok';
  }

  function taskRows() {
    if (tasks === null) {
      return `<div class="one-sec">${t('Loading…', '加载中…')}</div>`;
    }
    if (tasksError) {
      /* An unreachable feed is reported, never rendered as "no tasks" — an
         empty rail and a broken rail must not look the same. */
      return `<div class="one-sec">${t('Task list unavailable', '任务列表读取失败')}<br>${esc(tasksError)}</div>`;
    }
    if (!tasks.length) {
      return `<div class="one-sec">${t('No runs yet. Ask a question below to start one.', '还没有任务。在下面提个问题就能开始。')}</div>`;
    }
    return tasks.map((task) => `
      <button type="button" class="one-task" data-open-run="${esc(task.id)}">
        <span class="dot ${dotClass(task.status)}"></span>
        <span class="body">
          <span class="t">${esc(task.title)}</span>
          <span class="s">${esc(task.detail || '')}</span>
        </span>
      </button>`).join('');
  }

  function railHtml() {
    return `
    <aside class="one-rail" aria-label="${t('Tasks', '任务')}">
      <div class="one-brand">${icon('flask', 17)}<span>EasyICU</span></div>
      <button type="button" class="one-new" data-panel="">
        ${icon('spark', 15)}<span>${t('New question', '新问题')}</span>
      </button>
      <div class="one-sec">${t('Tasks', '任务')}</div>
      ${taskRows()}
      <div class="one-railfoot">
        <button type="button" class="one-iconbtn" data-panel="settings"
          title="${t('Settings', '设置')}" aria-label="${t('Settings', '设置')}">${icon('gear', 16)}</button>
        <button type="button" class="one-iconbtn" data-panel="tutorial"
          title="${t('Get Started', '快速上手')}" aria-label="${t('Get Started', '快速上手')}">${icon('help', 16)}</button>
        <button type="button" class="one-iconbtn" data-lang-toggle
          title="${t('Switch language', '切换语言')}" aria-label="${t('Switch language', '切换语言')}">${icon('globe', 16)}</button>
      </div>
    </aside>`;
  }

  function loadTasks() {
    if (tasks !== null) return;
    const done = (rows, err) => {
      tasks = rows;
      tasksError = err || '';
      if (window.__euRender) window.__euRender();
    };
    if (!window.fetch) { done([], ''); return; }
    fetch('/api/agent-runs/history', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ limit: 20 }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
      .then((data) => {
        const runs = (data && (data.runs || data.history)) || [];
        done(runs.map((run) => ({
          id: run.project_dir || run.run_id || '',
          title: run.question || run.study_id || run.run_id || t('Untitled run', '未命名任务'),
          detail: run.status || '',
          status: run.status || 'ok',
        })), '');
      })
      .catch((e) => done([], e && e.message ? e.message : 'error'));
  }

  /* ---- panel -------------------------------------------------------- */

  function panelHtml(id) {
    const scr = window.SCREENS[id];
    if (!scr) return '';
    let body;
    try {
      body = scr.render();
    } catch (e) {
      body = `<div class="one-sec">${t('This panel failed to render.', '该面板渲染失败。')} ${esc(e && e.message)}</div>`;
    }
    return `
    <section class="one-panel" aria-label="${esc(panelTitle(id))}">
      <div class="one-panelbar">
        <span class="t">${esc(panelTitle(id))}</span>
        <span class="sp"></span>
        <button type="button" class="one-iconbtn" data-panel=""
          title="${t('Close panel', '关闭面板')}" aria-label="${t('Close panel', '关闭面板')}">${icon('x', 16)}</button>
      </div>
      <div class="one-panelbody"><div class="content">${body}</div></div>
    </section>`;
  }

  function quickHtml(active) {
    return `
    <div class="one-quick" role="group" aria-label="${t('Open a panel', '打开面板')}">
      ${QUICK.map((q) => `
        <button type="button" class="one-chip" data-panel="${q.id}"
          aria-pressed="${active === q.id}">${L(q.label)}</button>`).join('')}
    </div>`;
  }

  window.EU_ONE = {
    PANEL_ROUTES,
    isPanel: (id) => PANEL_ROUTES.indexOf(id) >= 0,
    isWide: (id) => WIDE_PANELS.indexOf(id) >= 0,
    rail: railHtml,
    panel: panelHtml,
    quick: quickHtml,
    load: loadTasks,
  };

  /* One delegated listener owns the summon contract. data-panel="" closes. */
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-panel]');
    if (!el) return;
    const id = el.getAttribute('data-panel');
    e.preventDefault();
    if (!id) { location.hash = '#guided'; return; }
    if (window.SCREENS[id]) location.hash = '#' + id;
  });
}());
