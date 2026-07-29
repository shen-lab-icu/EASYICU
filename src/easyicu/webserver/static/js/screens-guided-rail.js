/* screens-guided-rail.js — owner: the study-folder list in the left rail.
 *
 * Split out of screens-guided.js (5,789 lines, far past the ~1,500-line budget)
 * when the rail became the surface's only persistent navigation and had to be
 * rebuilt. This is a real split, not a move: the list no longer reads guided's
 * closure. Everything it needs arrives as one explicit state object, so the two
 * files share a contract instead of a scope.
 *
 *   render(host, {
 *     loading, error,          // registry fetch state
 *     rows,                    // the raw draft rows, in their original order
 *     selectedId,              // which one the conversation is bound to
 *     compactPath, fmtRunTime, // guided's own formatters, passed in
 *   })
 *
 * Row indices are the caller's array indices. Grouping must never renumber
 * them: `data-localdraft` / `data-remove-localdraft` are how guided finds the
 * row again, and a regrouped index would open or delete the wrong folder.
 *
 * Shape follows Codex's task list rather than a file tree: one title line, one
 * meta line, grouped by recency, actions hidden until hover. The four stacked
 * meta lines it replaced (status · depth · mode, then path, then timestamp) put
 * three folders in the space the whole list gets.
 */
(function () {
  const t = (en, zh) => (window.t ? window.t(en, zh) : en);
  const icon = (n, s) => (window.icon ? window.icon(n, s) : '');

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
    ));
  }

  /* Recency buckets, computed against local midnight so "yesterday" means the
     calendar day and not "24 hours ago". Unparseable timestamps fall to the
     last bucket rather than being dropped — a folder you cannot date is still
     a folder you may need to open. */
  const BUCKETS = [
    { key: 'today', label: ['Today', '今天'] },
    { key: 'yesterday', label: ['Yesterday', '昨天'] },
    { key: 'week', label: ['Previous 7 days', '过去 7 天'] },
    { key: 'older', label: ['Older', '更早'] },
  ];

  function bucketOf(value) {
    const d = new Date(String(value || ''));
    if (Number.isNaN(d.getTime())) return 'older';
    const midnight = new Date();
    midnight.setHours(0, 0, 0, 0);
    const days = Math.floor((midnight.getTime() - d.getTime()) / 86400000);
    if (days <= 0) return 'today';
    if (days === 1) return 'yesterday';
    if (days < 7) return 'week';
    return 'older';
  }

  function rowHtml(row, index, state) {
    const active = state.selectedId && state.selectedId === row.id;
    const demo = String(row.data_mode || 'demo') === 'demo';
    const dir = row.project_dir ? state.compactPath(row.project_dir) : '';
    /* The path moved from a third line into the tooltip: still available, no
       longer costing every row a line of monospace it could not fit anyway. */
    const hint = dir
      ? `${t('Open this study folder', '打开这个研究文件夹')} · ${dir}`
      : t('Open this study folder', '打开这个研究文件夹');
    const meta = [
      demo ? t('Demo', '演示') : t('Real', '真实'),
      esc(row.depth || 'full'),
      esc(state.fmtRunTime(row.updated_at || row.created_at)),
    ].filter(Boolean).join(' · ');
    return `
      <div class="gd-sessline">
        <button class="gd-sess draft ${active ? 'active' : ''}" data-localdraft="${index}"
          title="${esc(hint)}">
          <span class="gr-dot ${demo ? 'demo' : 'real'}"></span>
          <span class="gr-txt">
            <span class="ss-t">${esc(row.title || t('Untitled study', '未命名研究'))}</span>
            <span class="ss-m">${meta}</span>
          </span>
        </button>
        <button class="gd-sess-action danger" type="button" data-remove-localdraft="${index}"
          title="${t('Remove from the study folder list', '从研究文件夹列表移除')}"
          aria-label="${t('Remove from the study folder list', '从研究文件夹列表移除')}">${icon('close', 12)}</button>
      </div>`;
  }

  function listHtml(state) {
    if (state.loading) {
      return `<div class="gd-empty-local"><div class="ss-t">${t('Loading study folders', '正在加载研究文件夹')}</div><div class="ss-m">${t('Reading metadata-only Guided folder registry.', '正在读取仅元数据的 Guided 文件夹 registry。')}</div></div>`;
    }
    if (state.error) {
      return `<div class="gd-empty-local warn"><div class="ss-t">${t('Study folders unavailable', '研究文件夹不可用')}</div><div class="ss-m">${esc(state.error)}</div></div>`;
    }
    const rows = Array.isArray(state.rows) ? state.rows : [];
    if (!rows.length) {
      return `<div class="gd-empty-local">
        <div class="ss-t">${t('No study folders yet', '还没有研究文件夹')}</div>
        <div class="ss-m">${t('Use New / open study folder to bind this conversation to a local project folder first.', '先使用“新建/打开研究文件夹”把这条对话绑定到本地项目文件夹。')}</div>
      </div>`;
    }
    /* Carry the original index through the grouping. */
    const indexed = rows.slice(0, 12).map((row, index) => ({ row, index }));
    return BUCKETS.map((bucket) => {
      const inBucket = indexed.filter(
        (item) => bucketOf(item.row.updated_at || item.row.created_at) === bucket.key,
      );
      if (!inBucket.length) return '';
      return `<div class="gr-group">${t(bucket.label[0], bucket.label[1])}</div>`
        + inBucket.map((item) => rowHtml(item.row, item.index, state)).join('');
    }).join('');
  }

  function render(host, state) {
    if (!host) return;
    host.innerHTML = `
      <div class="gd-rail-sec in-list">${t('Study folders', '研究文件夹')}
        <button class="gd-refresh-mini" data-refreshdrafts
          title="${t('Refresh study folders', '刷新研究文件夹')}">${icon('refresh', 10)}</button>
      </div>
      <div class="gd-rail-note"><span>${t('Each folder keeps this conversation’s memory. Open one to continue where it stopped.', '每个文件夹保存一段对话记忆，打开即可从中断处继续。')}</span></div>
      ${listHtml(state)}`;
  }

  window.EU_GUIDED_RAIL = { render };
}());
