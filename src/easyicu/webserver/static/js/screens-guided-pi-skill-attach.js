/* Owner: Skills attached to the next message, and a project's default Skills.
   The host owns both lists (/api/copilot/pi/sessions/{id}/draft-skills and
   /projects/{id}/default-skills): this file shows them and sends exact
   identities. An attached built-in Skill fixes the study design the plan must
   use; an attached personal Skill is read for that turn as advice only. The
   message itself carries no Skill text. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }

  const catalogs = new Map();
  const loading = new Map();
  let busy = false;
  let error = '';
  let defaults = { projectId: '', items: null };

  const keyOf = row => `${row.source}:${row.id}`;
  function identity(row) {
    if (row.source === 'user') return { source: 'user', id: row.id, digest: row.digest };
    return { source: 'builtin', id: row.id, version: row.version, ...(row.digest ? { digest: row.digest } : {}) };
  }
  function draft(session) {
    return session && Array.isArray(session.draft_attached_skills) ? session.draft_attached_skills : [];
  }
  function catalogRows(session) {
    const loaded = session && catalogs.get(session.session_id);
    return loaded ? loaded.items : [];
  }
  function maxAttached(session) {
    const loaded = session && catalogs.get(session.session_id);
    return loaded && Number.isInteger(loaded.max) ? loaded.max : 8;
  }
  function title(row, session) {
    const known = catalogRows(session).find(item => keyOf(item) === keyOf(row)) || row;
    return (window.EU_LANG === 'zh' && known.title_zh) || known.title || row.id;
  }
  function errorText(failure) {
    const code = failure && failure.code;
    return ({
      skill_attachment_disabled: tr('That Skill is turned off. Enable it in Skills first.', '这个技能未启用，请先在「技能」里启用。'),
      skill_attachment_stale: tr('That Skill changed after this conversation started. Start a new conversation to use the new version.', '这个技能在本对话开始后有改动，请新建对话后使用新版本。'),
      skill_attachment_unknown: tr('That Skill is not available in this conversation.', '这个技能在本对话中不可用。'),
      skill_attachment_family_conflict: tr('These Skills are for different study types. Keep one study type per message.', '这些技能属于不同的研究类型，一条消息只能选一种。'),
      skill_attachment_context_too_large: tr('These Skills are too long to read together. Remove one.', '这些技能合在一起太长，请去掉一个。'),
      pi_session_busy: tr('Wait until the current reply finishes.', '请等当前回复结束后再改。'),
    })[code] || String(failure && failure.message || failure || '');
  }

  // The conversation's attachable Skills: enabled built-ins and the personal
  // Skills frozen into this session, with their exact identities.
  async function load(projectId, session) {
    if (!session || !projectId || !api().loadPiSessionSkills) return [];
    if (catalogs.has(session.session_id)) return catalogs.get(session.session_id).items;
    if (!loading.has(session.session_id)) {
      loading.set(session.session_id, api().loadPiSessionSkills(session.session_id, projectId).then(payload => {
        const items = Array.isArray(payload && payload.items) ? payload.items : [];
        catalogs.set(session.session_id, { items, max: payload && payload.max_attached_skills });
        return items;
      }).finally(() => loading.delete(session.session_id)));
    }
    return loading.get(session.session_id);
  }
  function rowFor(session, source, id) {
    return catalogRows(session).find(row => row.source === source && row.id === id) || null;
  }
  function attached(session, source, id) {
    return draft(session).some(row => row.source === source && row.id === id);
  }

  async function saveDraft(projectId, session, rows) {
    if (busy || !api().setPiDraftSkills) return false;
    busy = true; error = '';
    try {
      const payload = await api().setPiDraftSkills(session.session_id, { project_id: projectId, attached_skills: rows });
      session.draft_attached_skills = Array.isArray(payload && payload.attached_skills) ? payload.attached_skills : rows;
      return true;
    } catch (failure) {
      error = errorText(failure);
      return false;
    } finally { busy = false; }
  }
  async function toggle(projectId, session, source, id) {
    await load(projectId, session);
    const current = draft(session);
    if (attached(session, source, id)) {
      return saveDraft(projectId, session, current.filter(row => !(row.source === source && row.id === id)));
    }
    const row = rowFor(session, source, id);
    if (!row) { error = errorText({ code: 'skill_attachment_unknown' }); return false; }
    if (row.enabled === false) { error = errorText({ code: 'skill_attachment_disabled' }); return false; }
    if (current.length >= maxAttached(session)) {
      error = tr(`Attach at most ${maxAttached(session)} Skills to one message.`, `一条消息最多挂 ${maxAttached(session)} 个技能。`);
      return false;
    }
    return saveDraft(projectId, session, [...current, identity(row)]);
  }
  function remove(projectId, session, key) {
    return saveDraft(projectId, session, draft(session).filter(row => keyOf(row) !== key));
  }
  // A hub start keeps a matching project default; picking in the menu still toggles it.
  function ensure(projectId, session, source, id) {
    return attached(session, source, id) ? Promise.resolve(true) : toggle(projectId, session, source, id);
  }

  function chips(session) {
    const rows = draft(session);
    if (!rows.length && !error) return '';
    return `${rows.map(row => `<div class="gpi-composer-skill gpi-attached-skill" role="status"><span>${row.source === 'user' ? tr('Your Skill', '我的技能') : tr('Skill', '技能')}</span><strong>${esc(title(row, session))}</strong><button type="button" data-gpi-attached-remove="${esc(keyOf(row))}" aria-label="${esc(tr('Remove Skill', '移除技能'))}" ${busy ? 'disabled' : ''}>×</button></div>`).join('')}${error ? `<div class="gpi-attached-skill-error" role="alert">${esc(error)}</div>` : ''}`;
  }

  // Project defaults are frozen into each new conversation when it is created;
  // changing them never touches an existing conversation.
  async function loadDefaults(projectId) {
    if (!projectId || !api().loadProjectDefaultSkills) return [];
    if (defaults.projectId !== projectId || !defaults.items) {
      const payload = await api().loadProjectDefaultSkills(projectId);
      defaults = { projectId, items: Array.isArray(payload && payload.attached_skills) ? payload.attached_skills : [] };
    }
    return defaults.items;
  }
  const sameSet = (left, right) => left.length === right.length
    && left.every(row => right.some(other => keyOf(other) === keyOf(row)));
  function defaultsControl(projectId, session) {
    if (!projectId || !session || defaults.projectId !== projectId || !defaults.items) return '';
    const current = draft(session);
    const isDefault = sameSet(current, defaults.items);
    if (!current.length && !defaults.items.length) return '';
    if (isDefault) {
      return `<span class="gpi-attached-defaults is-set">${esc(tr('Default for new conversations in this project', '本项目新对话默认带上'))}</span>`;
    }
    return `<button type="button" class="gpi-attached-defaults" data-gpi-attached-defaults ${busy ? 'disabled' : ''}>${esc(current.length
      ? tr('Use these Skills by default in this project', '设为本项目新对话的默认技能')
      : tr('Stop adding default Skills in this project', '本项目新对话不再默认带技能'))}</button>`;
  }
  async function saveDefaults(projectId, session) {
    if (busy || !api().setProjectDefaultSkills) return false;
    busy = true; error = '';
    try {
      const payload = await api().setProjectDefaultSkills(projectId, { attached_skills: draft(session) });
      defaults = { projectId, items: Array.isArray(payload && payload.attached_skills) ? payload.attached_skills : draft(session) };
      return true;
    } catch (failure) {
      error = errorText(failure);
      return false;
    } finally { busy = false; }
  }

  function reset(sessionId) { if (sessionId) catalogs.delete(sessionId); error = ''; }

  window.EasyICU.guidedPi.declare('skillAttach', {
    load, loadDefaults, toggle, ensure, remove, chips, defaultsControl, saveDefaults, attached, reset,
    hasDraft: session => draft(session).length > 0,
  });
})();
