/* Owner: read-only projection of assistant-authored research discussion drafts.
   These drafts live in the existing conversation history. They are never
   StudyContext, an Agent plan, a data-source confirmation, or run authority. */
(function () {
  'use strict';

  const SCHEMA = 'easyicu.research-draft/1';
  const TEXT_FIELDS = { title: 140, question: 1200, population: 600, exposure: 600,
    outcome: 600, candidate_data: 800, approach: 1200 };
  const LIST_FIELDS = { data_requirements: 12, open_questions: 10, changes: 10 };
  const KEYS = ['schema', ...Object.keys(TEXT_FIELDS), ...Object.keys(LIST_FIELDS)];
  function blocks() { return /(^|\n)```(?:easyicu-research-draft|json)\r?\n([\s\S]*?)\r?\n```(?=\r?\n|$)/g; }

  function validate(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).length !== KEYS.length || !KEYS.every(key => Object.hasOwn(value, key))
      || value.schema !== SCHEMA) return null;
    for (const [key, limit] of Object.entries(TEXT_FIELDS)) {
      if (typeof value[key] !== 'string' || value[key].length > limit
        || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value[key])) return null;
    }
    if (!value.title.trim() || !value.question.trim()) return null;
    for (const [key, limit] of Object.entries(LIST_FIELDS)) {
      if (!Array.isArray(value[key]) || value[key].length > limit
        || value[key].some(item => typeof item !== 'string' || !item.trim() || item.length > 300
          || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(item))) return null;
    }
    return value;
  }

  function parse(body) {
    if (body.length > 8000) return null;
    try { return validate(JSON.parse(body)); } catch (_) { return null; }
  }

  function read(text) {
    const matches = Array.from(String(text || '').matchAll(blocks()));
    return matches.map(match => parse(match[2])).filter(Boolean).at(-1) || null;
  }

  function strip(text) {
    // Malformed/unsupported blocks remain visible instead of silently losing
    // the assistant's text. Only this exact validated advisory format is lifted.
    return String(text || '').replace(blocks(), (whole, prefix, body) => parse(body) ? prefix : whole).trim();
  }

  function stripCopied(text, renderedText) {
    // Only remove exact blocks that were present and validated in the text
    // passed through the display projection. Preserve every other byte.
    const lifted = new Map();
    for (const match of String(renderedText || '').matchAll(blocks())) {
      if (!parse(match[2])) continue;
      const block = match[0].slice(match[1].length);
      lifted.set(block, (lifted.get(block) || 0) + 1);
    }
    return String(text || '').replace(blocks(), (whole, prefix, body) => {
      const block = whole.slice(prefix.length);
      const count = lifted.get(block) || 0;
      if (!count || !parse(body)) return whole;
      lifted.set(block, count - 1);
      return prefix;
    });
  }

  function streaming(text) {
    let preparing = false;
    const visible = String(text || '').replace(
      /(^|\n)```(easyicu-research-draft|json)\r?\n([\s\S]*?)(?:\r?\n```(?=\r?\n|$)|$)/g,
      (whole, prefix, label, body) => {
        // This is a temporary rendering hint, never an accepted draft. Ordinary
        // JSON stays visible. A json fence must identify this exact schema at
        // the start (or already validate fully) before its streamed body hides.
        const identified = label === 'easyicu-research-draft'
          || /^\s*\{\s*"schema"\s*:\s*"easyicu\.research-draft\/1"/.test(body)
          || Boolean(parse(body));
        if (!identified) return whole;
        preparing = true;
        return prefix;
      },
    );
    return { text: visible.trim(), preparing };
  }

  function isStreaming(row) {
    return Boolean(row && row.role === 'assistant' && row.complete === false
      && !row.errorCode && !row.hostFailedAttempt
      && !['error', 'aborted', 'cancelled', 'interrupted'].includes(String(row.stopReason || row.stop_reason || '')));
  }

  function accepts(row) {
    return Boolean(row && row.role === 'assistant' && row.complete !== false
      && !row.errorCode && !row.hostFailedAttempt
      && !['error', 'aborted', 'cancelled', 'interrupted'].includes(String(row.stopReason || row.stop_reason || '')));
  }

  function current(messages) {
    const rows = Array.isArray(messages) ? messages : [];
    const drafts = [];
    const seen = new Set();
    rows.forEach((row, index) => {
      if (!accepts(row)) return;
      const draft = read(row.text);
      if (!draft) return;
      const identity = String(row.entryId || row.id || '') || JSON.stringify(draft);
      if (seen.has(identity)) return;
      seen.add(identity);
      drafts.push({ draft, index, at: row.timelineAt || row.timestamp || null });
    });
    const latest = drafts.at(-1);
    if (!latest) return null;
    return { ...latest, history: drafts.slice(0, -1).slice(-5),
      pendingUpdate: rows.slice(latest.index + 1).some(row => row && row.role === 'user') };
  }

  function render(snapshot, ctx) {
    if (!snapshot || !validate(snapshot.draft)) return '';
    const { tr, esc } = ctx;
    const draft = snapshot.draft;
    const labels = {
      question: tr('Research question', '研究问题'), population: tr('Population', '研究人群'),
      exposure: tr('Exposure', '暴露'), outcome: tr('Primary outcome', '主要结局'),
      candidate_data: tr('Candidate data', '候选数据'), approach: tr('Proposed approach', '拟议方法'),
    };
    const list = (title, items) => items.length ? `<section><h4>${title}</h4><ul>${items.map(item => `<li>${esc(item)}</li>`).join('')}</ul></section>` : '';
    const at = Number(snapshot.at);
    const date = at > 0 ? new Date(at) : null;
    const when = date && Number.isFinite(date.getTime())
      ? date.toLocaleString(window.EU_LANG === 'zh' ? 'zh-CN' : 'en-US') : '';
    return `<section class="gpi-research-draft" data-gpi-research-draft aria-label="${tr('Discussion draft', '研究讨论草案')}">
      <div class="gpi-research-draft-status">${snapshot.pendingUpdate ? tr('Previous draft · awaiting an update', '上一份草案 · 待更新') : tr('Discussion draft · not approved', '讨论草案 · 未批准')}</div>
      <h3>${esc(draft.title)}</h3>
      <p class="gpi-research-draft-boundary">${tr('Assistant-authored discussion only; data and design are unverified. This does not approve preparation or analysis.', '助手整理的讨论记录，数据与设计尚未核验；不代表批准准备数据或执行分析。')}</p>
      ${when ? `<small>${tr('Draft updated: ', '草案更新：')}${esc(when)}</small>` : ''}
      <dl>${Object.entries(labels).filter(([key]) => draft[key].trim()).map(([key, label]) => `<div><dt>${label}</dt><dd>${esc(draft[key])}</dd></div>`).join('')}</dl>
      ${list(tr('Data needed', '需要的数据'), draft.data_requirements)}
      ${list(tr('Open decisions', '待定事项'), draft.open_questions)}
      ${list(tr('Changes in this draft', '本次草案变更'), draft.changes)}
      ${snapshot.history && snapshot.history.length ? `<details class="gpi-research-draft-history"><summary>${tr('Earlier drafts in this conversation', '本对话中的较早草案')}</summary>${snapshot.history.filter(previous => previous && validate(previous.draft)).slice().reverse().map(previous => `<article><h4>${esc(previous.draft.title)}</h4><p>${esc(previous.draft.question)}</p>${list(tr('Recorded changes', '当时的变更'), previous.draft.changes)}</article>`).join('')}</details>` : ''}
    </section>`;
  }

  window.EasyICU.guidedPi.declare('researchDraft', { read, strip, stripCopied, streaming, isStreaming, accepts, current, render });
})();
