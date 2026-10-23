/* Owner: Guided Copilot decision records -- the line an answered gate leaves.

   A gate card leaves the conversation once it is answered. What stays is one
   past-tense line per decision, at the place the decision was made. The host
   reads each record from the owner that persisted it (approval rows, the
   data-source authorization, provider_recovery.json) and sends it as
   `session.decision_records`; this owner only words and places them, so a
   reload shows the same lines. A system action is its own actor: it is never
   worded as the researcher's click. */
(function () {
  'use strict';

  const USER_CODES = new Set(['plan_confirmed', 'plan_approved', 'plan_rejected', 'data_chosen']);
  const SYSTEM_CODES = new Set(['provider_resumed']);
  const AGENT_CODES = new Set(['post_baseline_landmark_selected']);

  function timeMs(value) {
    const parsed = Date.parse(String(value || ''));
    return Number.isFinite(parsed) ? parsed : 0;
  }

  // The local wall-clock time of a record, as the conversation shows times.
  function clock(ms) {
    const at = new Date(ms);
    const pad = value => String(value).padStart(2, '0');
    return `${pad(at.getHours())}:${pad(at.getMinutes())}`;
  }

  // The records a session carries, as timeline rows. A record whose actor
  // does not match its code is dropped rather than shown as someone else's.
  function rows(session) {
    const records = session && Array.isArray(session.decision_records) ? session.decision_records : [];
    return records.flatMap((record, index) => {
      if (!record || typeof record !== 'object') return [];
      const code = String(record.code || '');
      const actor = String(record.actor || '');
      const at = timeMs(record.at);
      const known = actor === 'user' ? USER_CODES.has(code)
        : actor === 'agent' ? AGENT_CODES.has(code) : actor === 'system' && SYSTEM_CODES.has(code);
      if (!known || !at) return [];
      return [{
        ...record,
        id: `decision-${code}-${index}-${at}`,
        role: 'decision_record',
        code, actor, timelineAt: at,
      }];
    });
  }

  function sourceLabel(record) {
    const consent = window.EasyICU.guidedPi.optional('dataConsent');
    const source = { label: record.source_label, reference_release: record.source_release };
    return consent && typeof consent.sourceLabel === 'function'
      ? consent.sourceLabel({ source })
      : String(record.source_label || '').trim();
  }

  function text(record, tr) {
    const time = clock(record.timelineAt || timeMs(record.at));
    if (record.code === 'plan_confirmed') {
      return tr(`✓ You confirmed the plan and started data preparation · ${time}`, `✓ 你确认了方案，开始准备数据 · ${time}`);
    }
    if (record.code === 'plan_approved') {
      return tr(`✓ You approved the plan and started the analysis · ${time}`, `✓ 你批准了计划，开始分析 · ${time}`);
    }
    if (record.code === 'plan_rejected') {
      return tr(`✓ You sent the plan back · ${time}`, `✓ 你退回了计划 · ${time}`);
    }
    if (record.code === 'data_chosen') {
      const label = sourceLabel(record);
      return tr(`✓ You chose the data: ${label}`, `✓ 你选择了数据：${label}`);
    }
    if (record.code === 'provider_resumed') {
      const count = `${Number(record.number) || 0}/${Number(record.limit) || 0}`;
      return tr(
        `System action: the model service recovered; continued automatically from the checkpoint (${count}) · ${time}`,
        `系统操作：模型服务恢复后，从检查点自动继续（第 ${count} 次）· ${time}`,
      );
    }
    if (record.code === 'post_baseline_landmark_selected') {
      return tr(
        `Agent decision: the exposure is measured during the first ICU day, so the analysis starts at 24 hours among patients alive then · ${time}`,
        `Agent 决策：暴露在入 ICU 后第一天内测量，因此分析从第 24 小时开始，仅纳入该时点仍存活的患者 · ${time}`,
      );
    }
    return '';
  }

  // Place each record before the first timed row at or after it, so a
  // decision reads above the work it started. Untimed rows keep their place;
  // a record later than every row closes the conversation.
  function place(timeline, records) {
    const list = Array.isArray(timeline) ? timeline.slice() : [];
    (Array.isArray(records) ? records : [])
      .slice()
      .sort((left, right) => left.timelineAt - right.timelineAt)
      .forEach(record => {
        const index = list.findIndex(row => {
          const at = Number(row && (row.timelineAt || row.startedAt) || 0);
          if (!at) return false;
          return row.role === 'decision_record' ? at > record.timelineAt : at >= record.timelineAt;
        });
        if (index >= 0) list.splice(index, 0, record); else list.push(record);
      });
    return list;
  }

  function render(record, ctx) {
    const line = text(record, ctx.tr);
    if (!line) return '';
    const system = record.actor !== 'user';
    return `<p class="gpi-decision-record${system ? ' is-system' : ''}" role="note" data-gpi-decision-record="${ctx.esc(record.code)}" data-gpi-decision-actor="${ctx.esc(record.actor)}">${ctx.esc(line)}</p>`;
  }

  window.EasyICU.guidedPi.declare('decisionRecords', { rows, place, render, text, clock });
})();
