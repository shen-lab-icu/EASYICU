/* Owner: what the next message mentions with "@" -- a database, a variable or
   a project file. The host resolves each one to an exact identity and owns
   the draft (/api/copilot/pi/sessions/{id}/draft-mentions); this file searches
   the host, shows the draft as chips beside the attached Skills and sends
   references, never names or file text. A variable can be pinned as the
   exposure, outcome or a covariate. A mention says what the question is about;
   it grants no run, extraction or substitution approval. Skills attach
   through "/" (screens-guided-pi-skill-attach.js). */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }

  const KINDS = ['database', 'concept', 'file'];
  const ROLES = ['candidate', 'exposure', 'outcome', 'covariate'];
  const PAGE = 30;
  const kindLabel = kind => ({
    database: tr('Databases', '数据库'), concept: tr('Variables', '变量'), file: tr('Project files', '项目文件'),
  })[kind] || kind;
  const roleLabel = role => ({
    candidate: tr('Any role', '未指定'), exposure: tr('Exposure', '暴露'),
    outcome: tr('Outcome', '结局'), covariate: tr('Covariate', '协变量'),
  })[role] || role;

  const labels = new Map();
  const hydrating = new Set();
  let results = { key: '', items: [], hasMore: false, loading: false, failed: false };
  let sequence = 0;
  let timer = null;
  let busy = false;
  let error = '';
  let max = 16;
  let listener = null;

  const keyOf = ref => (ref.kind === 'database' ? `database:${ref.source || 'database'}:${ref.id}` : `${ref.kind}:${ref.id}`);
  const changed = () => { if (typeof listener === 'function') listener(); };
  const all = session => (session && Array.isArray(session.draft_mentions) ? session.draft_mentions : []);
  // A Skill mention stays in the draft; it shows as an attached Skill.
  const shown = session => all(session).filter(ref => KINDS.includes(ref.kind));

  function databaseName(key) {
    const names = window.EU_DATABASE_NAMES;
    return (names && typeof names.name === 'function' && names.name(key)) || key;
  }
  function conceptName(id) {
    const row = window.EU_CATALOG && window.EU_CATALOG.dict && window.EU_CATALOG.dict[id];
    return row ? tr(row[0], row[1] || row[0]) : '';
  }
  function label(ref) {
    if (ref.kind === 'concept') return conceptName(ref.id) || ref.id;
    if (ref.kind === 'database' && ref.source !== 'registered_source') return databaseName(ref.id);
    return labels.get(keyOf(ref)) || '';
  }
  function remember(row) {
    const ref = row && row.reference;
    const object = (row && row.object) || {};
    if (!ref) return;
    if (ref.kind === 'file' && object.file) labels.set(keyOf(ref), object.file);
    if (ref.kind === 'database' && object.database) labels.set(keyOf(ref), databaseName(object.database));
  }
  function errorText(failure) {
    const code = failure && failure.code;
    const text = {
      mention_unknown: tr('That item is no longer available. Choose it again.', '这一项已不可用，请重新选择。'),
      mention_stale: tr('That item changed after you chose it. Choose it again.', '这一项在选择后有变化，请重新选择。'),
      mention_invalid: tr('That item cannot be mentioned.', '这一项不能引用。'),
      mention_duplicate: tr('That item is already mentioned.', '这一项已经引用过了。'),
      mention_database_conflict: tr('Mention one database per message.', '一条消息只能引用一个数据库。'),
      mention_source_conflict: tr('Mention one local dataset per message.', '一条消息只能引用一份本机数据。'),
      mention_role_conflict: tr('Only one variable can be the exposure, and one the outcome.', '暴露和结局各只能指定一个变量。'),
      mention_context_too_large: tr('The mentioned files are too long to read together. Remove one.', '引用的文件合在一起太长，请去掉一个。'),
      mention_search_invalid: tr('That search is too long.', '搜索内容太长。'),
      mention_server_owned: tr('EasyICU manages that item; it cannot be mentioned.', '这一项由 EasyICU 管理，不能引用。'),
      pi_session_busy: tr('Wait until the current reply finishes.', '请等当前回复结束后再改。'),
    }[code];
    if (text) return text;
    if (failure && failure.status === 403) {
      return tr('The current model connection may not receive this data. Choose a qualifying connection first.',
        '当前模型连接不能接收这份数据，请先选择符合数据使用协议的模型连接。');
    }
    return String((failure && failure.message) || failure || '');
  }

  // One host search per kind and query; a newer search replaces an older one.
  function search(projectId, session, kind, query) {
    if (!session || !KINDS.includes(kind) || typeof api().searchPiMentions !== 'function') return;
    const text = String(query || '').trim().slice(0, 120);
    const key = JSON.stringify([projectId, session.session_id, kind, text]);
    if (key === results.key && !results.failed) return;
    clearTimeout(timer);
    results = { key, items: results.key.startsWith(JSON.stringify([projectId, session.session_id, kind]).slice(0, -1)) ? results.items : [],
      hasMore: false, loading: true, failed: false };
    const mine = ++sequence;
    timer = setTimeout(async () => {
      try {
        const payload = await api().searchPiMentions(session.session_id, { project_id: projectId, kind, query: text, limit: PAGE });
        if (mine !== sequence) return;
        if (payload && Number.isInteger(payload.max_mentions)) max = payload.max_mentions;
        results = { key, items: Array.isArray(payload && payload.items) ? payload.items : [],
          hasMore: Boolean(payload && payload.has_more), loading: false, failed: false };
      } catch (_) {
        if (mine !== sequence) return;
        results = { ...results, items: [], loading: false, failed: true };
      }
      changed();
    }, text ? 200 : 0);
  }
  const rowTitle = row => (row.kind === 'concept' ? conceptName(row.id) || row.label
    : row.kind === 'database' ? databaseName(row.source === 'registered_source' ? row.label : row.id) : row.label);
  const rowDetail = row => (row.kind === 'concept' ? row.id
    : row.kind === 'database' && row.source === 'registered_source' ? tr('Registered on this computer', '本机已登记') : '');
  function listHtml(projectId, session, kind) {
    const prefix = JSON.stringify([projectId, session && session.session_id, kind]).slice(0, -1);
    if (!results.key.startsWith(prefix) || (results.loading && !results.items.length)) {
      return `<p class="gpi-material-empty" role="status">${esc(tr('Searching…', '正在查找…'))}</p>`;
    }
    if (results.failed) {
      return `<p class="gpi-material-empty" role="alert">${esc(tr('Could not search right now.', '暂时无法查找。'))} <button type="button" data-gpi-mention-retry>${esc(tr('Try again', '重试'))}</button></p>`;
    }
    if (!results.items.length) return `<p class="gpi-material-empty" role="status">${esc(tr('Nothing matches.', '没有匹配的项。'))}</p>`;
    const taken = new Set(all(session).map(keyOf));
    return results.items.map((row, index) => {
      const used = taken.has(keyOf(row));
      const detail = rowDetail(row);
      return `<button type="button" class="gpi-composer-drawer-row" data-gpi-mention-add="${index}"${used ? ' aria-pressed="true" disabled' : ''}><strong>${esc(rowTitle(row))}</strong>${detail ? `<span>${esc(detail)}</span>` : ''}</button>`;
    }).join('') + (results.hasMore ? `<p class="gpi-material-empty" role="status">${esc(tr('Type more to narrow the list.', '继续输入以缩小范围。'))}</p>` : '');
  }

  async function write(projectId, session, next) {
    if (busy || typeof api().setPiDraftMentions !== 'function') return false;
    busy = true; error = '';
    try {
      const mentions = await next();
      const payload = await api().setPiDraftMentions(session.session_id, { project_id: projectId, mentions });
      session.draft_mentions = Array.isArray(payload && payload.mentions) ? payload.mentions : mentions;
      (Array.isArray(payload && payload.mentioned_objects) ? payload.mentioned_objects : []).forEach(remember);
      return true;
    } catch (failure) {
      error = errorText(failure);
      return false;
    } finally { busy = false; }
  }
  function resolve(projectId, session, ref, role) {
    return api().resolvePiMention(session.session_id, {
      project_id: projectId, kind: ref.kind, id: ref.id,
      ...(ref.kind === 'database' ? { source: ref.source || 'database' } : {}),
      ...(ref.kind === 'concept' && role ? { role } : {}),
      ...(ref.digest ? { digest: ref.digest } : {}),
    });
  }
  function add(projectId, session, index) {
    const row = results.items[Number(index)];
    if (!session || !row || !KINDS.includes(row.kind)) return Promise.resolve(false);
    if (all(session).length >= max) {
      error = tr(`Mention at most ${max} items in one message.`, `一条消息最多引用 ${max} 项。`);
      return Promise.resolve(false);
    }
    return write(projectId, session, async () => {
      const resolved = await resolve(projectId, session, { kind: row.kind, id: row.id, source: row.source });
      remember(resolved);
      return [...all(session), resolved.reference];
    });
  }
  function remove(projectId, session, key) {
    return write(projectId, session, async () => all(session).filter(ref => keyOf(ref) !== key));
  }
  function setRole(projectId, session, key, role) {
    const ref = all(session).find(item => keyOf(item) === key);
    if (!ref || ref.kind !== 'concept' || !ROLES.includes(role) || (ref.role || 'candidate') === role) return Promise.resolve(false);
    return write(projectId, session, async () => {
      const resolved = await resolve(projectId, session, { kind: 'concept', id: ref.id }, role);
      return all(session).map(item => (keyOf(item) === key ? resolved.reference : item));
    });
  }
  // After a reload a file or local dataset is known only by its identity; one
  // resolution names it.
  function hydrate(projectId, session) {
    if (typeof api().resolvePiMention !== 'function') return;
    shown(session).filter(ref => !label(ref) && !hydrating.has(keyOf(ref))).forEach(ref => {
      hydrating.add(keyOf(ref));
      resolve(projectId, session, ref, ref.role).then(row => { remember(row); changed(); }, () => {});
    });
  }

  function chips(projectId, session) {
    const rows = shown(session);
    if (rows.length) hydrate(projectId, session);
    if (!rows.length && !error) return '';
    return rows.map(ref => {
      const key = keyOf(ref);
      const kind = ref.kind === 'concept' ? tr('Variable', '变量') : ref.kind === 'file' ? tr('File', '文件')
        : ref.source === 'registered_source' ? tr('Local data', '本机数据') : tr('Database', '数据库');
      const role = ref.kind === 'concept'
        ? `<select data-gpi-mention-role="${esc(key)}" aria-label="${esc(tr('Role in the study', '在研究中的角色'))}"${busy ? ' disabled' : ''}>${ROLES.map(value => `<option value="${value}"${(ref.role || 'candidate') === value ? ' selected' : ''}>${esc(roleLabel(value))}</option>`).join('')}</select>`
        : '';
      return `<div class="gpi-composer-skill gpi-mention-chip"><span>${esc(kind)}</span><strong>${esc(label(ref) || '…')}</strong>${role}<button type="button" data-gpi-mention-remove="${esc(key)}" aria-label="${esc(tr('Remove mention', '移除引用'))}"${busy ? ' disabled' : ''}>×</button></div>`;
    }).join('') + (error ? `<div class="gpi-attached-skill-error" role="alert">${esc(error)}</div>` : '');
  }

  function reset() { results = { key: '', items: [], hasMore: false, loading: false, failed: false }; error = ''; }

  window.EasyICU.guidedPi.declare('mentions', {
    KINDS, kindLabel, search, listHtml, add, remove, setRole, chips, reset,
    onChange(fn) { listener = fn; },
    hasDraft: session => shown(session).length > 0,
  });
})();
