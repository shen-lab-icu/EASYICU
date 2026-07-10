/* Cross-DB source choice owner: registered EasyICU exports vs raw ICU roots. */
(function () {
  'use strict';

  function text(en, zh) {
    return window.EU_LANG === 'zh' ? zh : en;
  }

  function registeredPaths() {
    const host = window.EU_CROSSDB_SOURCE_HOST;
    if (!host || typeof host.registeredPaths !== 'function') return [];
    const paths = host.registeredPaths();
    return Array.from(new Set((Array.isArray(paths) ? paths : []).map(path => String(path || '').trim()).filter(Boolean)));
  }

  function render() {
    const count = registeredPaths().length;
    const ready = count >= 2;
    const sourceLabel = count === 1
      ? text('1 selected export', '已选择 1 个导出')
      : text(`${count} selected exports`, `已选择 ${count} 个导出`);
    return `
      <section data-crossdb-source-choice>
        <div class="note info">
          <div class="body">
            <div class="t">${text('Choose one local Cross-DB source', '选择一种本地跨库来源')}</div>
            <div class="d">${text('Use prepared EasyICU exports for a bounded aggregate comparison, or read raw ICU database folders with local sampling.', '使用已准备好的 EasyICU 导出进行有界聚合对比，或通过本地抽样读取原始 ICU 数据库文件夹。')}</div>
          </div>
        </div>
        <div class="card pad mt-16" data-crossdb-registered-option>
          <div class="row between gap-12" style="align-items:flex-start;">
            <div>
              <div class="eyebrow">${text('Source A', '来源 A')}</div>
              <div class="panel-title mt-4">${text('Registered EasyICU exports', '已注册的 EasyICU 导出')}</div>
              <div class="panel-sub mt-4">${text('Compare the aggregate summaries of at least two exports already selected in the source rail. This path does not scan raw database folders.', '对比来源栏中已选择的至少两个导出的聚合摘要；此路径不会扫描原始数据库文件夹。')}</div>
            </div>
            <span class="pill ${ready ? 'ok' : 'warn'}">${sourceLabel}</span>
          </div>
          <div class="row gap-8 mt-14" style="align-items:center;flex-wrap:wrap;">
            <button class="btn primary" type="button" data-crossdb-run-registered ${ready ? '' : 'aria-disabled="true"'}>${text('Run registered exports', '运行已注册导出')}</button>
            ${ready
              ? `<span class="panel-sub">${text('Ready for aggregate-only comparison.', '已可运行仅聚合对比。')}</span>`
              : `<span class="panel-sub">${text('Add and select at least two EasyICU exports from the source rail on the left.', '请从左侧来源栏添加并选择至少两个 EasyICU 导出。')}</span>`}
          </div>
        </div>
        <div class="sec-stack"><div class="lbl">${text('Or · Source B · Raw ICU database root', '或者 · 来源 B · 原始 ICU 数据库根目录')}</div></div>
      </section>`;
  }

  function renderLoading() {
    const count = registeredPaths().length;
    return `<div class="card pad" data-crossdb-registered-loading role="status" aria-live="polite">
      <div class="load-strip">
        <span class="spin accent"></span>
        <div class="grow">
          <div style="font-weight:600;font-size:12.75px;">${text('Loading registered export summaries…', '正在加载已注册导出的摘要…')}</div>
          <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${text(`${count} local exports · aggregate-only`, `${count} 个本地导出 · 仅聚合`)}</div>
        </div>
      </div>
      <div class="indet mt-12"></div>
      <div class="panel-sub mt-8">${text('EasyICU is reading bounded summaries from the selected exports. Raw ICU folders are not scanned.', 'EasyICU 正在读取所选导出的有界摘要；不会扫描原始 ICU 文件夹。')}</div>
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
  }

  window.EU_CROSSDB_SOURCE_CHOICE = { render, renderLoading, wire };
})();
