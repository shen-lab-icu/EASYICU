/* Owner: Guided Pi "遇到问题" menu in the conversation header. */
/* A clinician who hits a problem needs two things: a reference the
   administrator can look up, and whom to ask. The reference names the
   project, conversation, run and job by their ids, with the version and time;
   it carries no path, question text, credential or patient data. The
   administrator contact comes from the hosted-instance owner
   (settings.support_contact). */
(function () {
  'use strict';

  const COPIED_MS = 1400;

  async function copyText(text) {
    try {
      if (window.navigator && window.navigator.clipboard && window.navigator.clipboard.writeText) {
        await window.navigator.clipboard.writeText(text);
        return true;
      }
    } catch (_) { /* fall back below */ }
    const doc = window.document;
    if (!doc || !doc.body || typeof doc.execCommand !== 'function') return false;
    const area = doc.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    doc.body.appendChild(area);
    area.select();
    let copied = false;
    try { copied = doc.execCommand('copy'); } catch (_) { copied = false; }
    area.remove();
    return copied;
  }

  const ID = /^[A-Za-z0-9_.:-]{1,120}$/;
  function id(value) {
    const text = String(value || '').trim();
    return ID.test(text) ? text : '';
  }

  function create(options) {
    const { tr, esc, context } = options;

    function referenceText() {
      const facts = (typeof context === 'function' ? context() : null) || {};
      const version = String(((window.EU_SETTINGS || {}).about || {}).version || '').slice(0, 40);
      const rows = [
        [tr('Project', '项目'), id(facts.projectId)],
        [tr('Conversation', '对话'), id(facts.sessionId)],
        [tr('Run', '运行'), id(facts.runId)],
        [tr('Job', '任务'), id(facts.jobId)],
      ].filter(([, value]) => value).map(([label, value]) => `${label}: ${value}`);
      return [
        `EasyICU ${version}`.trim(),
        ...rows,
        `${tr('Time', '时间')}: ${new Date().toISOString().replace(/\.\d+Z$/, 'Z')}`,
      ].join('\n');
    }

    function contact() {
      const hosted = window.EasyICU.guidedPi.optional('hosted');
      return hosted ? hosted.supportContact() : '';
    }

    function render() {
      const who = contact();
      return `<details class="gpi-help-control" data-popover-menu data-gpi-help>
        <summary>${esc(tr('Having a problem?', '遇到问题'))}</summary>
        <div class="gpi-help-menu" role="menu">
          <button type="button" role="menuitem" data-gpi-help-copy>${esc(tr('Copy problem reference', '复制问题编号'))}</button>
          ${who ? `<p class="gpi-help-contact">${esc(tr('Send it to', '发给管理员'))}：${esc(who)}</p>` : ''}
        </div>
      </details>`;
    }

    // Returns true when the click belonged to this menu.
    function handleClick(event) {
      const button = event.target.closest && event.target.closest('[data-gpi-help-copy]');
      if (!button) return false;
      void copyText(referenceText()).then(copied => {
        button.textContent = copied ? tr('Copied', '已复制') : tr('Copy failed', '复制失败');
        setTimeout(() => {
          const menu = button.closest('details[open]');
          if (menu) menu.removeAttribute('open');
          button.textContent = tr('Copy problem reference', '复制问题编号');
        }, COPIED_MS);
      });
      return true;
    }

    return Object.freeze({ render, handleClick, referenceText });
  }

  window.EasyICU.guidedPi.declare('help', { create });
})();
