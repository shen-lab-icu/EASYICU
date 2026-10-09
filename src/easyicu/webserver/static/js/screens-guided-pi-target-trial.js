/* Owner: a causal study's target trial card in the conversation.

   The host computes the card (workflow projection target_trial_card, owned by
   webserver/target_trial_card.py) and names the study's step as the
   workflow's next action: target_trial_statement_needed (no trial stated),
   target_trial_review (compiling, stopped, blocked or approvable) or
   target_trial_plan_ready (approved, so the plan reads this study's data).
   This owner turns them into the plan card's confirmation and its body, keeps
   the researcher's line-by-line confirmations, and builds the approval
   request.  The host decides every state; only card.approvable lets the card
   approve, and only once every line is ticked.  The protocol, confirmations
   and limitations are English and inside the record's digest, so they are
   shown as written; every value is escaped.  Wording:
   screens-guided-pi-target-trial-copy.js. */
(function () {
  'use strict';

  const CODES = new Set([
    'target_trial_statement_needed',
    'target_trial_review',
    'target_trial_plan_ready',
  ]);
  const DIGEST = /^[0-9a-f]{64}$/;
  // The lines the researcher has ticked, per compiled record.  Kept in this
  // page only: a click approves what the card shows, and a reload asks again.
  const ticked = new Map();

  function copyLine(code, tr, fallback) {
    const copy = window.EasyICU.guidedPi.optional('targetTrialCopy');
    return (copy && code && copy.line(code, tr)) || String(fallback || '');
  }

  function label(kind, value, tr) {
    const copy = window.EasyICU.guidedPi.optional('targetTrialCopy');
    if (!copy) return String(value || '');
    return kind === 'item' ? copy.itemLabel(value, tr) : copy.limitationLabel(value, tr);
  }

  function cardOf(workflow) {
    const card = workflow && workflow.target_trial_card;
    return card && typeof card === 'object' ? card : null;
  }

  function rows(value) {
    return Array.isArray(value) ? value.filter(row => row && typeof row === 'object') : [];
  }

  function digestOf(card) {
    const digest = String((card && card.compile_sha256) || '');
    return DIGEST.test(digest) ? digest : '';
  }

  // The record's count of lines to confirm, when the card lists exactly that many.
  function lineCount(card) {
    const count = Number(card && card.confirmation_lines);
    return Number.isInteger(count) && count >= 0 && count === rows(card && card.confirmations).length
      ? count : -1;
  }

  function tickedLines(card) {
    return ticked.get(digestOf(card)) || new Set();
  }

  function allTicked(card) {
    const count = lineCount(card);
    return Boolean(digestOf(card)) && count >= 0 && tickedLines(card).size === count;
  }

  function approvable(card) {
    return Boolean(card) && card.state === 'approvable' && card.approvable === true && card.stale !== true;
  }

  function stopReason(card) {
    const latest = (card && card.latest_compile) || {};
    return String((card && card.reason_code) || latest.reason_code || '');
  }

  function confirmation(workflow, tr) {
    const code = String((workflow && workflow.next_action_code) || '');
    if (!CODES.has(code)) return null;
    const card = cardOf(workflow);
    const base = { code, grants: [], trialCard: true, editLabel: tr('Revise in the conversation', '在对话里修改') };
    if (code === 'target_trial_statement_needed') return {
      ...base, nonApprovable: true,
      title: tr('State the target trial this study emulates', '说明本研究模拟的目标试验'),
      note: copyLine(code, tr),
      editLabel: tr('State it in the conversation', '在对话里说明'),
    };
    if (code === 'target_trial_plan_ready') return {
      ...base, grants: ['provider_run', 'literature'], hideEdit: true,
      title: tr('The target trial is approved', '目标试验已批准'),
      note: tr('Generating the plan reads this study’s data package; it pauses for your review before any analysis.', '生成计划会读取本研究的数据包，分析前仍会停下等你审核。'),
      approve: tr('Generate the plan on this study’s data', '按本研究数据生成计划'),
    };
    if (approvable(card)) return {
      ...base, approveTrial: true, approveDisabled: !allTicked(card),
      title: tr('Check and approve the target trial', '核对并批准目标试验'),
      note: tr(
        'Tick each line to confirm it, then approve. Approving records this version of the trial; no analysis starts.',
        '请逐行勾选确认，再批准。批准只记录这一版试验，不会开始分析。',
      ),
      approve: tr('Approve the target trial', '批准目标试验'),
    };
    const state = String((card && card.state) || '');
    if (state === 'blocked') return {
      ...base, nonApprovable: true,
      title: tr('The target trial cannot be approved yet', '目标试验还不能批准'),
      note: tr('Each item below holds the approval. Revise the trial in the conversation.', '下列各项挡住了批准，请在对话里修改试验。'),
    };
    if (state === 'stopped') {
      const latest = (card && card.latest_compile) || {};
      const reason = stopReason(card);
      return {
        ...base, nonApprovable: true,
        title: tr('The target trial was not compiled', '目标试验没有编成'),
        note: copyLine(reason, tr, latest.detail) || copyLine('target_trial_review', tr),
      };
    }
    if (state === 'compiling') return {
      ...base, nonApprovable: true, hideEdit: true,
      title: tr('Compiling the target trial', '正在编译目标试验'),
      note: copyLine('easyicu_target_trial_compile_submitted', tr),
    };
    return { ...base, nonApprovable: true, title: tr('The target trial is not approved', '目标试验尚未批准'), note: copyLine(code, tr) };
  }

  function reasonHtml(card, { tr, esc }) {
    // The card's own reason (a record it cannot find) is not the compile's.
    if (!card || card.state !== 'stopped' || card.reason_code) return '';
    const latest = card.latest_compile || {};
    const missing = Array.isArray(latest.missing_concepts) ? latest.missing_concepts.slice(0, 16) : [];
    const cause = String(latest.cause_code || '');
    const detail = String(latest.detail || '');
    const parts = [
      detail && copyLine(stopReason(card), tr) ? `<p class="gpi-trial-detail" lang="en">${esc(detail)}</p>` : '',
      missing.length ? `<p class="gpi-trial-detail"><span>${esc(tr('Concepts the package lacks:', '数据包缺少的概念：'))}</span> <code>${missing.map(esc).join('</code> <code>')}</code></p>` : '',
      cause ? `<p class="gpi-trial-detail"><span>${esc(tr('Cause code:', '原因码：'))}</span> <code>${esc(cause)}</code></p>` : '',
    ];
    return parts.join('');
  }

  function blockingHtml(card, { tr, esc }) {
    const blocking = rows(card && card.blocking).slice(0, 24);
    if (!blocking.length) return '';
    return `<ul class="gpi-trial-blocking">${blocking.map(row => {
      const line = copyLine(String(row.reason || ''), tr, row.detail);
      return `<li><strong>${esc(String(row.name || row.source || ''))}</strong><span>${esc(line)}</span>${row.detail && line !== row.detail ? `<small lang="en">${esc(String(row.detail))}</small>` : ''}</li>`;
    }).join('')}</ul>`;
  }

  function protocolHtml(card, { tr, esc }, open) {
    const protocol = rows(card && card.protocol);
    if (!protocol.length) return '';
    const items = protocol.map(row => `<div><dt>${esc(label('item', row.item, tr))}</dt><dd lang="en">${esc(String(row.text || ''))}</dd></div>`).join('');
    return `<details class="gpi-trial-section"${open ? ' open' : ''}><summary>${esc(tr('Protocol', '试验规程'))}</summary><dl class="gpi-trial-protocol">${items}</dl></details>`;
  }

  function confirmationsHtml(card, { tr, esc }, editable) {
    const lines = rows(card && card.confirmations);
    if (!lines.length) return '';
    const digest = digestOf(card);
    const chosen = tickedLines(card);
    const heading = tr(`Lines to confirm (${lines.length})`, `需要确认的 ${lines.length} 行`);
    const items = lines.map((row, index) => editable && digest
      ? `<li><label><input type="checkbox" data-gpi-trial-line="${index}" data-gpi-trial-digest="${esc(digest)}"${chosen.has(index) ? ' checked' : ''}><span lang="en">${esc(String(row.text || ''))}</span></label></li>`
      : `<li><span lang="en">${esc(String(row.text || ''))}</span></li>`).join('');
    return `<details class="gpi-trial-section"${editable ? ' open' : ''}><summary>${esc(heading)}</summary><ul class="gpi-trial-confirmations" data-gpi-trial-lines="${lines.length}">${items}</ul></details>`;
  }

  function limitationsHtml(card, { tr, esc }) {
    const limitations = rows(card && card.limitations).slice(0, 16);
    const ceiling = String((card && card.evidence_ceiling) || '');
    if (!limitations.length && !ceiling) return '';
    const items = limitations.map(row => {
      const name = label('limitation', row.code, tr);
      return `<li>${name ? `<strong>${esc(name)}</strong>` : ''}<span lang="en">${esc(String(row.text || ''))}</span></li>`;
    }).join('');
    // The compiled record lists its evidence ceiling among its limitations;
    // only a card whose limitations do not carry it states it in one line.
    const listed = limitations.some(row => row.code === 'evidence_ceiling');
    const ceilingLine = ceiling === 'analysis_only' && !listed
      ? `<p class="gpi-trial-ceiling">${esc(label('limitation', 'evidence_ceiling', tr) || tr('Evidence ceiling: analysis only', '证据上限：仅为分析'))}</p>` : '';
    return `<details class="gpi-trial-section"><summary>${esc(tr(`Limitations (${limitations.length})`, `局限（${limitations.length}）`))}</summary>${ceilingLine}<ul class="gpi-trial-limitations">${items}</ul></details>`;
  }

  function approvalHtml(card, { tr, esc }) {
    const approval = card && card.approval;
    if (!approval || card.state !== 'approved') return '';
    const count = Number(approval.n_lines_confirmed);
    return `<p class="gpi-trial-approved">${esc(tr(
      `Approved with ${Number.isInteger(count) ? count : 0} lines confirmed.`,
      `已批准，确认了 ${Number.isInteger(count) ? count : 0} 行。`,
    ))}</p>`;
  }

  // The card's body under the confirmation's title and note.
  function bodyHtml(workflow, helpers) {
    const card = cardOf(workflow);
    if (!card) return '';
    const { tr, esc } = helpers;
    const editable = approvable(card);
    const previous = card.stale === true && (card.protocol || []).length
      ? `<p class="gpi-trial-stale">${esc(tr('Below is the previous version of the trial; it cannot be approved until the new statement compiles.', '下面是上一版试验；新的陈述编好之前不能批准。'))}</p>`
      : '';
    return `<div class="gpi-trial">${previous}${reasonHtml(card, helpers)}${blockingHtml(card, helpers)}${approvalHtml(card, helpers)}${protocolHtml(card, helpers, editable)}${confirmationsHtml(card, helpers, editable)}${limitationsHtml(card, helpers)}</div>`;
  }

  // The approval the card's click sends: the record it shows, every line ticked.
  function approvalRequest(workflow, session, projectId) {
    const card = cardOf(workflow);
    const binding = (session && session.binding) || {};
    const studyId = String(binding.study_context_id || '').trim();
    const revision = Number(binding.study_revision || 0);
    if (!approvable(card) || !allTicked(card) || !studyId || !revision) return null;
    return {
      project_id: String(projectId || ''),
      study_context_id: studyId,
      expected_revision: revision,
      compile_sha256: digestOf(card),
      n_lines_confirmed: lineCount(card),
    };
  }

  function refusalText(error, tr) {
    return copyLine(String((error && error.code) || ''), tr);
  }

  function changeDraft(workflow, tr) {
    const code = String((workflow && workflow.next_action_code) || '');
    if (!CODES.has(code)) return '';
    return code === 'target_trial_statement_needed'
      ? tr('The target trial I want to emulate: ', '我要模拟的目标试验：')
      : tr('I want to change the target trial: ', '我想修改目标试验：');
  }

  // One line of one record ticked or cleared; the page keeps the ticks across
  // repaints. Returns how many lines of that record are ticked.
  function noteTick(digest, index, checked) {
    if (!DIGEST.test(String(digest || '')) || !Number.isInteger(index) || index < 0) return 0;
    const chosen = ticked.get(digest) || new Set();
    if (checked) chosen.add(index); else chosen.delete(index);
    ticked.set(digest, chosen);
    return chosen.size;
  }

  // A ticked line enables the card's approval once every line is ticked.
  function onChange(event) {
    const box = event.target && event.target.closest ? event.target.closest('[data-gpi-trial-line]') : null;
    if (!box) return;
    const count = noteTick(String(box.dataset.gpiTrialDigest || ''), Number(box.dataset.gpiTrialLine), box.checked);
    const list = box.closest('[data-gpi-trial-lines]');
    const total = Number(list && list.dataset.gpiTrialLines);
    const card = box.closest('.gpi-confirmation');
    const approve = card && card.querySelector('[data-gpi-confirm-action]');
    if (approve) approve.disabled = !(Number.isInteger(total) && count === total);
  }
  if (typeof document !== 'undefined' && document.addEventListener) document.addEventListener('change', onChange);

  window.EasyICU.guidedPi.declare('targetTrial', Object.freeze({
    CODES,
    confirmation,
    bodyHtml,
    approvalRequest,
    refusalText,
    changeDraft,
    noteTick,
  }));
})();
