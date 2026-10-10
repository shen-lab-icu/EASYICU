/* Owner: the shared artifact reader's view of the question-requirement records. */
/* The planning owner (research_agent/planning/question_requirements.py) keeps
   two records of what the question asks of a plan: the planning record
   (question_requirements.json), judged on the plan the Planner compiled, and
   the judgment of the plan a review request offers
   (question_requirements_review.json), which an approval rests on.  This owner
   reads either one as tables: each requirement in the question's words, how
   the host judged the plan on it, why, and whether EasyICU verified it; on a
   route that lists no requirements, each concept the question names and what
   reads it.  Every disposition and verification is the record's own; nothing
   here judges again.  Quotes, notes and gap details come from the question or
   a model, so each is clipped here, and the shared table escapes every cell.
   The renderer (screens-agent-render.js) asks this owner first, at call time,
   so it loads before the renderer. */
(function () {
  'use strict';

  const PLANNING_SCHEMA = 'easyicu.question_requirements/1';
  const REVIEW_SCHEMA = 'easyicu.question_requirements_review/1';
  const FILES = new Set(['question_requirements.json', 'question_requirements_review.json']);
  // The owner's bounds: a quote, a note and a gap's detail; a named concept's
  // words (QuestionRequirement, ProgressiveCapabilityGap, NamedQuestionConcept).
  const QUOTE_LIMIT = 240;
  const NOTE_LIMIT = 300;
  const EVIDENCE_LIMIT = 120;
  const MAX_ROWS = 8;
  // The dispositions that refuse approval (QUESTION_REQUIREMENT_APPROVAL_STOPS).
  const STOPS = new Set(['not_covered', 'capability_gap']);
  // A gap the host decided for a family template carries the host's English
  // detail; its Chinese reading is supplied here, by its stable reason code.
  const TEMPLATE_GAPS_ZH = {
    question_benchmark_step_unavailable_in_family_template: '这份计划所用的模板没有在同一批行上把模型与已有评分或模型比较的步骤',
    question_subgroup_step_unavailable_in_family_template: '这份计划所用的模板没有分析指定亚组的步骤',
  };

  // The renderers' own i18n global decides the language.
  function tr(en, zh) {
    return typeof window.t === 'function' ? window.t(en, zh) : en;
  }
  function clip(value, limit) {
    const text = String(value == null ? '' : value).replace(/\s+/g, ' ').trim();
    return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
  }
  function names(values) {
    return (Array.isArray(values) ? values : []).slice(0, MAX_ROWS).map(value => clip(value, 128)).filter(Boolean).join(', ');
  }
  // A grouped column reads by its group name (screens-agent-exposure-levels.js).
  function conceptNames(values, levels) {
    return names((Array.isArray(values) ? values : []).map(value => (levels && levels.variableName(value)) || value));
  }
  function quoted(text) {
    return tr(`“${text}”`, `「${text}」`);
  }
  function rows(values) {
    return (Array.isArray(values) ? values : []).filter(row => row && typeof row === 'object').slice(0, MAX_ROWS);
  }

  function kindLabel(kind) {
    const labels = {
      benchmark: tr('Comparison with an existing score or model', '与已有评分或模型比较'),
      subgroup: tr('Named subgroup', '指定亚组'),
      estimand: tr('Named estimand or measure', '指定估计量或指标'),
      analysis: tr('Other named analysis', '其他指定分析'),
      definition: tr('Definition', '定义'),
    };
    return labels[kind] || clip(kind, 40);
  }

  function requirementCell(row) {
    return `${clip(row.id, 8)} · ${kindLabel(row.kind)}${tr(': ', '：')}${quoted(clip(row.quote, QUOTE_LIMIT))}`;
  }

  // On the judgment of the plan under review a stop refuses approval; the
  // planning record judged an earlier plan, so it only says what it found.
  function dispositionCell(row, review) {
    const stop = review && STOPS.has(row.disposition) ? tr(' · stops approval', ' · 不能批准') : '';
    const labels = {
      covered: tr('Answered', '已回答'),
      not_covered: tr('Not answered', '没有回答'),
      capability_gap: tr('This plan cannot do it', '这份计划做不到'),
      attested: tr('The plan states it answers it', '计划声明已回答'),
      definition_only: tr('Only defines another element', '只用于定义'),
    };
    return (labels[row.disposition] || clip(row.disposition, 40)) + stop;
  }

  // Why, as the planning owner decides each disposition (question_requirements._judge).
  function reasonCell(row, levels) {
    const reading = names(row.reading_step_ids);
    const gap = row.gap && typeof row.gap === 'object' ? row.gap : null;
    if (row.disposition === 'covered') {
      const owners = names(row.owner_step_ids);
      return owners ? tr(`Answered by step ${owners}`, `由步骤 ${owners} 回答`) : '';
    }
    if (row.disposition === 'not_covered') {
      const concepts = conceptNames(row.concepts, levels);
      const why = row.kind === 'benchmark'
        ? tr('No step compares the model with it on the same rows', '没有步骤在同一批行上把模型与它比较')
        : row.kind === 'subgroup'
          ? tr('No subgroup analysis step reads it', '没有亚组分析步骤读取它')
          : tr('Not every concept it names is read by an analysis step', '它的概念并非都有分析步骤读取');
      return why
        + (concepts ? tr(` (${concepts})`, `（${concepts}）`) : '')
        + (reading ? tr(`; analysis steps that read it: ${reading}`, `；读取它的分析步骤：${reading}`) : '');
    }
    if (row.disposition === 'capability_gap') {
      const detail = gap ? clip(gap.detail, NOTE_LIMIT) : '';
      if (row.host_judged === true && TEMPLATE_GAPS_ZH[row.reason_code]) {
        return tr(detail, TEMPLATE_GAPS_ZH[row.reason_code]);
      }
      return detail ? tr(`Planning says: ${detail}`, `规划的说明：${detail}`) : '';
    }
    if (row.disposition === 'attested') {
      return reading
        ? tr(
          `Analysis steps ${reading} read its concepts; reading a concept is not doing the analysis the question names`,
          `分析步骤 ${reading} 读取了它的概念；读取概念不等于做了题面要求的分析`,
        )
        : tr('It names no concept', '它没有指明概念');
    }
    if (row.disposition === 'definition_only') {
      const note = clip(row.note, NOTE_LIMIT);
      return note ? tr(`The plan says it only defines: ${note}`, `计划说它只用于定义：${note}`) : '';
    }
    return clip(row.reason_code, 80);
  }

  // Who stands behind the disposition: the planning owner says
  // (JudgedRequirement.verified_by_host), and for a gap how it was checked.
  function verificationCell(row) {
    if (row.verified_by_host === true) return tr('Verified by EasyICU', 'EasyICU 已核实');
    if (row.disposition !== 'capability_gap') return tr('Stated by the plan, not verified by EasyICU', '计划声明，EasyICU 未核实');
    if (row.gap_verification === 'unverifiable') return tr('Stated by planning; EasyICU cannot check it', '规划声明，EasyICU 无法核对');
    if (row.gap_verification === 'unverified') return tr('Stated by planning; EasyICU’s check did not confirm it', '规划声明，EasyICU 核对未证实');
    return tr('Stated by planning, not verified by EasyICU', '规划声明，EasyICU 未核实');
  }

  function unstatedRow(row, levels) {
    const readers = [
      names(row.reading_step_ids),
      row.cohort_criterion === true ? tr('a cohort criterion', '人群条件') : '',
    ].filter(Boolean).join(', ');
    return [
      quoted(clip(row.evidence, EVIDENCE_LIMIT)),
      conceptNames(row.concepts, levels),
      readers || tr('No step reads it; check whether the plan leaves it out', '没有步骤读取，请核对计划是否遗漏了它'),
    ];
  }

  function summaryPill(judged, unstated, esc) {
    const stops = judged.filter(row => STOPS.has(row.disposition)).length;
    const toCheck = judged.filter(row => !STOPS.has(row.disposition) && row.verified_by_host !== true).length + unstated.length;
    if (stops) return `<span class="pill warn" style="height:22px;">${esc(tr(`${stops} not answered or not possible`, `${stops} 项没有回答或做不到`))}</span>`;
    if (toCheck > 0) return `<span class="pill info" style="height:22px;">${esc(tr(`${toCheck} to check`, `${toCheck} 项待核对`))}</span>`;
    return `<span class="pill ok" style="height:22px;">${esc(tr('Nothing to resolve', '没有待处理的要求'))}</span>`;
  }

  function frame(title, pill, sections, esc) {
    return `<div class="ag-artifact-readable ag-question-requirements-reader">
      <div class="ag-artifact-readable-head"><div><div class="eyebrow">${esc(tr('Question requirements', '题面要求'))}</div><div class="ag-artifact-readable-title">${esc(title)}</div></div>${pill}</div>
      ${sections.join('')}
    </div>`;
  }

  /* The readable view of one record, or '' when the artifact is neither
     record.  A record file whose content is not one of the owner's schemas
     says it cannot be read instead of showing an empty judgment. */
  function view(name, payload, { artifactTable, esc, context }) {
    const record = payload && typeof payload === 'object' ? payload : {};
    const schema = String(record.schema_version || '');
    const known = schema === PLANNING_SCHEMA || schema === REVIEW_SCHEMA;
    if (!known && !FILES.has(String(name || '').toLowerCase())) return '';
    if (!known) {
      return frame(
        tr('This record cannot be read as question requirements, so no judgment is shown. The raw JSON is kept for audit.', '这份记录无法按题面要求读取，未显示任何判定；原始 JSON 保留用于审计。'),
        `<span class="pill warn" style="height:22px;">${esc(tr('unreadable', '无法读取'))}</span>`,
        [],
        esc,
      );
    }
    const review = schema === REVIEW_SCHEMA;
    const outline = record.route === 'outline';
    const levels = window.AGENT_EXPOSURE_LEVELS && window.AGENT_EXPOSURE_LEVELS.reader(context, tr);
    const judged = rows(record.judged);
    const unstated = rows(record.unstated);
    const sections = [];
    if (judged.length || !outline) {
      sections.push(artifactTable(
        tr('Each requirement and how the plan answers it', '每项要求与判定'),
        [tr('Requirement', '要求'), tr('Disposition', '判定'), tr('Why', '原因'), tr('Verified', '核实')],
        judged.map(row => [requirementCell(row), dispositionCell(row, review), reasonCell(row, levels), verificationCell(row)]),
        tr('The question asks for nothing beyond the study design.', '题面没有在研究设计之外提出要求。'),
        { formattedCells: true },
      ));
    }
    if (unstated.length || outline) {
      sections.push(artifactTable(
        tr('Concepts the question names (planning listed no requirements; check each one)', '题面提到的概念（规划没有逐项列出要求，请逐个核对）'),
        [tr('In the question', '题面原文'), tr('Columns it can denote', '可指的列'), tr('Read by', '读取它的')],
        unstated.map(row => unstatedRow(row, levels)),
        tr('Every concept the question names is already in the study design.', '题面提到的概念都已在研究设计中。'),
        { formattedCells: true },
      ));
    }
    const title = review
      ? tr('Judged on the plan offered for review; approval rests on this judgment. The raw JSON is kept for audit.', '在提交审阅的计划上的判定，批准依据这一份；原始 JSON 保留用于审计。')
      : tr('Judged on the plan as planned. The plan is shaped further before review, and the plan offered for review is judged in its own record. The raw JSON is kept for audit.', '在规划所得计划上的判定。计划在审阅前还会加工，提交审阅的计划另有一份判定；原始 JSON 保留用于审计。');
    return frame(title, summaryPill(judged, unstated, esc), sections, esc);
  }

  window.AGENT_QUESTION_REQUIREMENTS = Object.freeze({ view });
})();
