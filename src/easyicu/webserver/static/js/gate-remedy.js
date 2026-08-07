/* Owner: turning a fail-closed refusal into something the user can act on.

   The gates in this product are the point. What they were not doing is saying
   what would clear them. The Agent's cross-DB blocker, for example, explained
   that "the current Agent runner consumes one export and cannot execute the
   aggregate multi-database payload yet" — an accurate statement of our
   implementation limit that tells a clinician nothing about what to do next.

   Worse, screens-agent.js recovered the remedy by running regular expressions
   over the backend's English prose ("/prior-art/i", "/re-extract/i"), even
   though the backend had decided from typed conditions and had a stable code
   in hand the whole time. Prose is not a contract; the code is.

   This module owns exactly one mapping: reason code -> { requirement, route,
   cta }. It renders nothing about WHY (the caller already has that) and
   everything about WHAT NOW. A code with no entry here yields no remedy —
   never a guessed one. */
(function () {
  'use strict';

  function T(en, zh) { return (window.t || (a => a))(en, zh); }

  // route is a hash target; cta labels the button that goes there.
  const REMEDIES = {
    export_not_real: () => ({
      requirement: T('A real EasyICU export is required. Demo fixtures cannot back a reportable result.',
        '需要一个真实的 EasyICU 导出。演示夹具不能支撑可报告结果。'),
      route: 'extraction',
      cta: T('Prepare or select an export', '准备或选择导出'),
    }),
    required_concepts_missing: () => ({
      requirement: T('Some required concepts are missing from the active export. Re-extract them, or confirm the study can proceed without them.',
        '当前导出缺少部分必要概念。请重新抽取，或确认研究可以在缺少它们的情况下进行。'),
      route: 'extraction',
      cta: T('Open Data Extraction', '打开数据抽取'),
    }),
    prior_art_not_reviewed: () => ({
      requirement: T('A prior-art review has to complete before execution — a blocked or failed search does not count as reviewed.',
        '执行前需要完成 prior-art 审阅 —— 被拦截或失败的检索不算已审阅。'),
      route: 'ideas',
      cta: T('Run prior-art review', '进行 prior-art 审阅'),
    }),
    idea_not_recommended: () => ({
      requirement: T('This idea is not marked "recommend" yet. Resolve its feasibility findings first.',
        '这个 idea 尚未标记为 recommend，请先解决它的可行性问题。'),
      route: 'ideas',
      cta: T('Open Idea Mining', '打开想法挖掘'),
    }),
    seed_gate_missing: () => ({
      requirement: T('This project was created before the preflight checks existed. Refresh it from Idea Mining to generate them.',
        '这个项目创建于预检条件之前，请回到想法挖掘刷新它以生成预检。'),
      route: 'ideas',
      cta: T('Refresh from Idea Mining', '回到想法挖掘刷新'),
    }),
    active_export_changed: () => ({
      requirement: T('The active export is not the one this project was planned against. Select the original export, or replan against the current one.',
        '当前导出与这个项目当初规划所用的不是同一个。请选回原导出，或基于当前导出重新规划。'),
      route: 'extraction',
      cta: T('Choose the active export', '选择当前导出'),
    }),
    crossdb_plan_only: () => ({
      requirement: T('A cross-database comparison can shape an analysis plan, but a run needs one reviewed cohort from a single export. Review a cohort, then come back.',
        '跨库对比可以用来制定分析计划，但运行需要来自单个导出的、已审阅的队列。先审阅一个队列，再回到这里。'),
      route: 'cohort',
      cta: T('Review a cohort', '去审阅队列'),
    }),
  };

  function forCode(code) {
    const build = REMEDIES[String(code || '')];
    return build ? Object.assign({ code: String(code) }, build()) : null;
  }

  /* Accepts a backend error payload. Prefers blocker_codes; falls back to the
     top-level error code. Unknown codes are dropped, not guessed. */
  function forPayload(payload) {
    const source = payload || {};
    const codes = Array.isArray(source.blocker_codes) ? source.blocker_codes : [];
    const resolved = codes.map(forCode).filter(Boolean);
    if (resolved.length) return resolved;
    const single = forCode(source.error);
    return single ? [single] : [];
  }

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  /* One row per remedy: what is required, and the button that goes there. */
  function render(remedies) {
    const rows = (Array.isArray(remedies) ? remedies : [remedies]).filter(Boolean);
    if (!rows.length) return '';
    const ic = (name, size) => (window.icon ? window.icon(name, size) : '');
    return `<div class="gate-remedy">
      <div class="gate-remedy-head">${ic('arrow', 13)} ${T('What clears this', '如何解除')}</div>
      ${rows.map(row => `<div class="gate-remedy-row" data-gate-remedy="${esc(row.code)}">
        <span class="gate-remedy-req">${esc(row.requirement)}</span>
        ${row.route ? `<button type="button" class="btn sm" data-nav="${esc(row.route)}">${esc(row.cta)}</button>` : ''}
      </div>`).join('')}
    </div>`;
  }

  window.EU_GATE_REMEDY = { forCode, forPayload, render, CODES: Object.keys(REMEDIES) };
})();
