/* Owner: Guided pipeline summary widget. */
/* Guided study-pipeline summary owner.
   Owns the aside "step overview + study item list" rendering and its
   collapsed state; screens-guided.js owns the conversation shell and
   delegates here. The shell hands its live readers through init() because it
   reassigns studyStatus/studyVal/thread on reset — capturing the objects
   would leave this owner reading detached state (same accessor pattern as
   the idea/extract/review splits). */
(function () {
  'use strict';

  let host = null;
  let pipelineOpen = false;

  function normalizedStudyRows() {
    const gi = host.goalIdx();
    return host.studyTable().map(([id, label, ico, labelZh], idx) => {
      let stt = host.studyStatus()[id] || 'pending';
      // steps past the chosen finish line are optional — dim them unless already reached
      if (idx > gi && (stt === 'pending')) stt = 'beyond';
      let v = host.studyVal()[id]; if (typeof v === 'function') v = v();
      return { id, label, ico, labelZh, idx, stt, v };
    });
  }

  function renderStudyPipelineSummary() {
    const rows = normalizedStudyRows();
    let activeIdx = rows.findIndex(r => r.stt === 'active');
    if (activeIdx < 0) activeIdx = rows.findIndex(r => r.stt !== 'done' && r.stt !== 'beyond');
    if (activeIdx < 0) activeIdx = 0;
    const active = rows[activeIdx] || rows[0];
    const next = rows.slice(activeIdx + 1).find(r => r.stt !== 'beyond');
    const done = rows.filter(r => r.stt === 'done').length;
    const total = Math.max(1, Math.min(host.goalIdx() + 1, host.studyTable().length));
    const pct = Math.max(0, Math.min(100, Math.round(done / total * 100)));
    const currentValue = active && active.v ? `<div class="gd-pipeline-value">${host.esc(active.v)}</div>` : '';
    const nextLine = next
      ? `<span>${host.t('Next', '下一步')}</span><strong>${host.t(next.label, next.labelZh || next.label)}</strong>`
      : `<span>${host.t('Next', '下一步')}</span><strong>${host.t('Ready for sign-off', '等待核验')}</strong>`;
    return `
      <div class="gd-pipeline-summary" data-gd-pipeline-summary>
        <div class="gd-pipeline-summary-head">
          <div>
            <div class="eyebrow">${host.t('Step overview', '步骤总览')}</div>
            <strong>${host.t(active.label, active.labelZh || active.label)}</strong>
            ${currentValue}
          </div>
          <button class="gd-pipeline-toggle" type="button" data-gd-pipeline-toggle aria-controls="gdPipelineList" aria-expanded="${pipelineOpen ? 'true' : 'false'}">
            ${pipelineOpen ? host.t('Hide steps', '收起步骤') : host.t('Show all steps', '展开步骤')}
          </button>
        </div>
        <div class="gd-pipeline-bar" aria-label="${host.t('Guided Copilot progress', '研究引导进度')}"><span style="width:${pct}%;"></span></div>
        <div class="gd-pipeline-meta">
          <span><strong>${done}/${total}</strong> ${host.t('required steps done', '个必需步骤完成')}</span>
          <span>${host.t('Goal', '目标')} · ${host.studyDepth().label}</span>
        </div>
        <div class="gd-pipeline-next">${nextLine}</div>
      </div>`;
  }

  function renderStudyItemList() {
    const gi = host.goalIdx();
    return `<div class="gd-pipeline-list ${pipelineOpen ? 'open' : 'collapsed'}" id="gdPipelineList" ${pipelineOpen ? '' : 'hidden'} data-gd-pipeline-list>` + normalizedStudyRows().map(({ id, label, ico, labelZh, idx, stt, v }) => {
      const dot = stt === 'done' ? host.icon('check', 11, 3) : stt === 'locked' ? host.icon('lock', 10) : host.icon(ico, 12);
      const badge = stt === 'active' ? '<span class="si-state"><span class="spin sm" style="width:11px;height:11px;"></span></span>'
        : stt === 'locked' ? `<span class="si-state pill warn" style="height:18px;"><span class="dot"></span></span>`
        : stt === 'beyond' ? `<span class="si-state si-opt">${host.t('optional', '可选')}</span>` : '';
      const clickable = host.thread().some(t => t.card && t.step === id);
      const row = `<div class="study-item ${stt}${clickable ? ' nav' : ''}" ${clickable ? `data-study="${id}" role="button" tabindex="0"` : ''}><span class="si-dot">${dot}</span><div class="si-txt"><div class="si-t">${host.t(label, labelZh || label)}</div>${v ? `<div class="si-v">${host.esc(v)}</div>` : ''}</div>${badge}</div>`;
      // draw the finish line right after the goal step (only when stopping short of the full study)
      const fin = (idx === gi && host.depthName() !== 'full')
        ? `<div class="study-finishline"><span class="fl-flag">${host.icon('check', 10, 3)}</span><span class="fl-t">${host.t('Finish line', '终点线')} · ${host.studyDepth().label}</span></div>`
        : '';
      return row + fin;
    }).join('') + '</div>';
  }

  window.EU_GUIDED_PIPELINE = {
    init(options) { host = options; },
    resetState() { pipelineOpen = false; },
    // Shell event delegation calls this first; true means the toggle was
    // consumed here and the shell should re-render the aside.
    handleClick(target) {
      if (target && typeof target.closest === 'function' && target.closest('[data-gd-pipeline-toggle]')) {
        pipelineOpen = !pipelineOpen;
        return true;
      }
      return false;
    },
    normalizedStudyRows,
    renderStudyPipelineSummary,
    renderStudyItemList,
  };
})();
