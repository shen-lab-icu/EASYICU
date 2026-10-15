/* Owner: the plan review card's question-requirement block.

   The host projects the record of what the question asks of the plan
   (workflow.plan_review_summary.question_requirements, owned by
   pi_copilot/question_requirement_notes.py): the judgment of the plan under
   review, or the planning record when there is none, which may not hold for
   the plan.  This owner renders only what a reviewer must look at, and every
   line names its source: a requirement the plan does not answer or cannot
   carry out, a claim the plan made that EasyICU did not verify, and, on a
   route that lists no requirements, each concept the question names.  Quotes
   and notes come from the question or a model, so each is truncated here and
   the whole line is escaped. */
(function () {
  'use strict';

  const QUOTE_LIMIT = 240;
  const NOTE_LIMIT = 300;
  // The run files the block may link: the record it read.
  const RECORD_FILES = new Set(['question_requirements_review.json', 'question_requirements.json']);

  function clip(value, limit) {
    const text = String(value == null ? '' : value).replace(/\s+/g, ' ').trim();
    return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
  }

  function itemLine(item, tr) {
    const quote = clip(item.quote, QUOTE_LIMIT);
    if (item.disposition === 'not_covered') {
      return tr(`Not answered by this plan: “${quote}”`, `这份计划没有回答：「${quote}」`);
    }
    if (item.disposition === 'capability_gap') {
      return item.source === 'host_verified'
        ? tr(`This plan cannot carry out: “${quote}”`, `这份计划做不到：「${quote}」`)
        : tr(
          `Planning states this plan cannot carry out (not verified by EasyICU): “${quote}”`,
          `规划声明这份计划做不到（EasyICU 未核实）：「${quote}」`,
        );
    }
    const stated = tr(`Stated by the plan, not verified by EasyICU: “${quote}”`, `计划声明，EasyICU 未核实：「${quote}」`);
    const note = item.disposition === 'definition_only' ? clip(item.note, NOTE_LIMIT) : '';
    return note
      ? stated + tr(` (the plan says it only defines: ${note})`, `（计划说它只用于定义：${note}）`)
      : stated;
  }

  function unstatedLine(row, tr) {
    const evidence = clip(row.evidence, QUOTE_LIMIT);
    return row.read
      ? tr(
        `Planning did not list the question's requirements; the plan reads “${evidence}”, which the question names — check that the plan answers it`,
        `规划时没有逐项列出题面要求；计划读取了题面提到的「${evidence}」，请核对计划是否回答了它`,
      )
      : tr(
        `Planning did not list the question's requirements; no step reads “${evidence}”, which the question names — check whether the plan leaves it out`,
        `规划时没有逐项列出题面要求；题面提到的「${evidence}」没有被任何步骤读取，请核对计划是否遗漏了它`,
      );
  }

  function notesHtml(summary, { tr, esc, recordLink = () => '' }) {
    const notes = summary && summary.question_requirements;
    const lines = [];
    const readings = summary && summary.coordinate_readings;
    if (readings && readings.status === 'unavailable') lines.push(tr(
      'The source record for the plan’s automatically selected outcome and exposure could not be verified. Check these coordinates in the complete plan.',
      '计划自动选择暴露与结局的来源记录无法核验，请直接核对完整计划中的指标。',
    ));
    else if (readings && readings.status === 'shown') {
      (Array.isArray(readings.items) ? readings.items : []).slice(0, 2)
        .filter(row => row && row.basis === 'keyword' && ['exposure', 'outcome'].includes(row.role))
        .forEach(row => {
          const role = row.role === 'exposure' ? tr('Exposure', '暴露') : tr('Outcome', '结局');
          const name = clip(tr(row.label_en, row.label_zh) || row.label_en || row.label_zh || tr('Unlabelled measure', '未命名指标'), 80);
          const vocab = window.AGENT_READER_VOCAB;
          const category = vocab && typeof vocab.category === 'function' ? vocab.category(row.category) : '';
          const evidence = clip(row.evidence, QUOTE_LIMIT);
          lines.push(tr(
            `${role}: ${name}${category ? ` (${category})` : ''} — read by EasyICU from the question${evidence ? ` “${evidence}”` : ''}; not a separate choice you confirmed. Check it before approving the plan; request changes in the conversation if needed.`,
            `${role}：${name}${category ? `（${category}）` : ''}——EasyICU 从题面${evidence ? `“${evidence}”` : ''}读到，未经你单独确认。批准计划前请核对；需要修改时在对话里说明。`,
          ));
        });
    }
    if (notes && notes.status === 'unavailable') {
      lines.push(tr('The question-requirements record could not be read, so it is not shown', '题面要求记录无法读取，未显示'));
    } else if (notes && notes.status === 'shown') {
      (Array.isArray(notes.items) ? notes.items : [])
        .filter(item => item && typeof item === 'object')
        .forEach(item => lines.push(itemLine(item, tr)));
      (Array.isArray(notes.unstated) ? notes.unstated : [])
        .filter(row => row && typeof row === 'object')
        .forEach(row => lines.push(unstatedLine(row, tr)));
      if (lines.length && notes.judged_on_plan_under_review === false) {
        lines.push(tr(
          'These judgments refer to an earlier version of the plan and may not hold for it',
          '这些判定针对改动前的计划，对当前计划可能已不成立',
        ));
      }
      const covered = Math.max(0, Math.floor(Number(notes.covered_count) || 0));
      if (lines.length && covered) {
        lines.push(tr(
          `${covered} more ${covered === 1 ? 'is' : 'are'} answered by plan steps (verified by EasyICU)`,
          `另有 ${covered} 项已由计划步骤回答（EasyICU 已核实）`,
        ));
      }
    }
    if (!lines.length) return '';
    // recordLink(file) returns the confirmation owner's own resource button
    // (trusted HTML) for the record the block read; an unreadable record has
    // nothing to open.
    const linked = notes && notes.status === 'shown' && RECORD_FILES.has(notes.record) ? recordLink(notes.record) : '';
    const link = linked ? `<div class="gpi-question-requirements-link">${linked}</div>` : '';
    const heading = readings ? tr('Question and plan', '题目与计划') : tr('Question requirements', '题面要求');
    return `<div class="gpi-question-requirements"><strong>${esc(heading)}</strong><ul>${lines.map(line => `<li>${esc(line)}</li>`).join('')}</ul>${link}</div>`;
  }

  window.EasyICU.guidedPi.declare('questionRequirements', { notesHtml });
})();
