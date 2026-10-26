/* Owner: Cross-DB source choice widget. */
/* Cross-DB source choice owner: prepared data (own exports and public demos) or raw ICU roots. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;

  function text(en, zh) {
    return window.EU_LANG === 'zh' ? zh : en;
  }

  function registeredPaths() {
    const host = window.EU_CROSSDB_SOURCE_HOST;
    if (!host || typeof host.registeredPaths !== 'function') return [];
    const paths = host.registeredPaths();
    return Array.from(new Set((Array.isArray(paths) ? paths : []).map(path => String(path || '').trim()).filter(Boolean)));
  }

  function render(options) {
    const registryHtml = String(options && options.registryHtml || '');
    const offersHtml = String(options && options.offersHtml || '');
    const count = registeredPaths().length;
    const ready = count >= 2;
    const sourceLabel = count === 1
      ? text('1 selected export', '已选择 1 个导出')
      : text(`${count} selected exports`, `已选择 ${count} 个导出`);
    return `
      <section data-crossdb-source-choice>
        ${registryHtml}
        <div class="gate-strip mt-14">
          <span class="pill ${ready ? 'ok' : 'warn'}">${sourceLabel}</span>
          <span class="panel-sub">${ready
            ? text('Ready for an aggregate-only consistency check.', '已可运行仅聚合的一致性检查。')
            : text('Select at least two prepared datasets; public demos count.', '请选择至少两份已准备的数据，公开 Demo 也可以。')}</span>
          <div class="grow"></div>
          <button class="btn primary" type="button" data-crossdb-run-registered ${ready ? '' : 'aria-disabled="true"'}>${text('Start consistency check', '开始一致性检查')}</button>
        </div>
        ${offersHtml}
      </section>`;
  }

  function officialPaths() {
    const host = window.EU_CROSSDB_SOURCE_HOST;
    if (!host || typeof host.officialPaths !== 'function') return [];
    return Array.from(new Set(host.officialPaths().map(path => String(path || '').trim()).filter(Boolean)));
  }

  function renderLoading() {
    const context = window.EU_DATA_MODE_CONTEXT;
    const officialPair = context && context.kind === 'official_demo_pair';
    const count = officialPair ? officialPaths().length : registeredPaths().length;
    const title = officialPair
      ? text('Loading official demo summaries…', '正在加载官方 Demo 摘要…')
      : text('Loading registered export summaries…', '正在加载已注册导出的摘要…');
    const meta = officialPair
      ? text(`${count} official demos · aggregate-only`, `${count} 个官方 Demo · 仅聚合`)
      : text(`${count} local exports · aggregate-only`, `${count} 个本地导出 · 仅聚合`);
    return `<div class="card pad" data-crossdb-registered-loading role="status" aria-live="polite">
      <div class="load-strip">
        <span class="spin accent"></span>
        <div class="grow">
          <div style="font-weight:600;font-size:12.75px;">${title}</div>
          <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${meta}</div>
        </div>
        <button class="btn sm" type="button" data-crossdb-cancel>${text('Cancel', '取消')}</button>
      </div>
      <div class="indet mt-12"></div>
      <div class="panel-sub mt-8">${text('Reading summaries of the selected exports; raw databases are not read.', '正在读取所选数据的汇总；不会读取原始数据库。')}</div>
    </div>`;
  }

  function wire(root) {
    if (!root || typeof root.querySelectorAll !== 'function') return;
    root.querySelectorAll('[data-crossdb-run-registered]').forEach(button => {
      if (button.dataset.crossdbRegisteredBound === '1') return;
      button.dataset.crossdbRegisteredBound = '1';
      button.addEventListener('click', event => {
        event.preventDefault();
        event.stopPropagation();
        if (button.getAttribute('aria-disabled') === 'true' || registeredPaths().length < 2) return;
        const host = window.EU_CROSSDB_SOURCE_HOST;
        if (host && typeof host.runRegistered === 'function') host.runRegistered();
      });
    });
    const sourceOwner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
    const host = window.EU_CROSSDB_SOURCE_HOST;
    const demoChoice = sourceOwner && typeof root.querySelector === 'function'
      ? root.querySelector('[data-crossdb-source-choice]')
      : null;
    if (sourceOwner && demoChoice) {
      sourceOwner.ensureLoaded(() => {
        if (host && typeof host.repaint === 'function') host.repaint();
      });
      sourceOwner.bind(root, {
        refresh: () => {
          if (host && typeof host.repaint === 'function') host.repaint();
        },
        openPrepared: sourceId => {
          if (host && typeof host.openOfficial === 'function') host.openOfficial(sourceId);
        },
      });
    }
  }

  window.EU_CROSSDB_SOURCE_CHOICE = { render, renderLoading, wire };
})();
