/* Owner: Guided Copilot model-suggested follow-up questions.
   The model ends a completed answer with a '可以继续问：' / 'Follow-up
   questions:' block of 2-3 questions (prompt contract in the Pi bridge).
   This owner lifts that block out of the reply text and renders it as the
   reference-style suggestion list under the latest reply. Clicking one sends
   it as the researcher's own next message; it carries no grant. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  const HEADING = /(?:^|\n)[ \t]*(?:\*\*)?[ \t]*(?:可以继续问|继续追问|追问建议|Follow-up questions?)[ \t]*[:：][ \t]*(?:\*\*)?[ \t]*\n((?:[ \t]*[-*•][ \t]+[^\n]+\n?)+)[ \t]*$/i;
  const MAX_QUESTIONS = 3;

  function split(text) {
    const value = String(text || '');
    const match = value.match(HEADING);
    if (!match) return { text: value, questions: [] };
    const questions = match[1].split('\n')
      .map(line => line.replace(/^[ \t]*[-*•][ \t]+/, '').trim())
      .filter(Boolean)
      .slice(0, MAX_QUESTIONS);
    if (!questions.length) return { text: value, questions: [] };
    return { text: value.slice(0, match.index).replace(/\s+$/, ''), questions };
  }

  function render(questions, options) {
    const rows = Array.isArray(questions) ? questions.filter(Boolean) : [];
    if (!rows.length) return '';
    const tr = options && typeof options.tr === 'function' ? options.tr : en => en;
    const iconHtml = options && typeof options.iconHtml === 'function' ? options.iconHtml : () => '';
    const disabled = options && options.disabled ? ' disabled' : '';
    return `<details class="gpi-followups gpi-model-followups" open>
      <summary class="gpi-followups-head">${iconHtml('help', 14)}<span>${esc(tr('Follow-up questions', '继续追问'))}</span></summary>
      <div class="gpi-followups-list">${rows.map(question => `<div class="gpi-followup-row"><button type="button" class="gpi-followup-prompt" data-gpi-model-followup="${esc(question)}"${disabled}><span aria-hidden="true">↳</span>${esc(question)}</button></div>`).join('')}</div>
    </details>`;
  }

  window.EasyICU.guidedPi.declare('followUps', { split, render });
})();
