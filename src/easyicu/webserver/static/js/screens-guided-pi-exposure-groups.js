/* Owner: the plan review card's exposure-grouping block.

   The host projects the exposure groupings of the plan under review
   (workflow.plan_review_summary.exposure_groups, owned by
   pi_copilot/exposure_group_notes.py): for each, the grouped concept, its
   levels in code order with the study's own label for each, an English rule
   per level that the host wrote from the grouping record, and the two levels
   compared; or that the record cannot be read, which is said in the words
   every reader of the groups uses (screens-agent-exposure-levels.js).
   This owner names each level by its label and marks the reference and the
   compared level; the rules open below.  Levels are separated alike whether
   the scale is ordered or not, so none reads as a gradient.  A grouping whose
   record states no labels (codes_only) is shown by its codes, never given a
   meaning here.  Every value is escaped. */
(function () {
  'use strict';

  function levelName(level, grouping) {
    return grouping.status === 'labelled' && level.label ? String(level.label) : String(level.code);
  }

  function marked(level, grouping, tr) {
    const name = levelName(level, grouping);
    if (level.code === grouping.reference) return name + tr(' (reference)', '（参照）');
    if (level.code === grouping.contrast) return name + tr(' (compared)', '（比较）');
    return name;
  }

  function groupingHtml(grouping, { tr, esc }) {
    const levels = (Array.isArray(grouping.levels) ? grouping.levels : []).filter(level => level && typeof level === 'object');
    if (!levels.length) return '';
    const concept = tr(grouping.concept_label_en || '', grouping.concept_label_zh || '')
      || String(grouping.concept || grouping.variable || '');
    const names = levels.map(level => marked(level, grouping, tr)).join(' / ');
    const codesOnly = grouping.status === 'codes_only' ? tr(' (the record states no labels)', '（记录没有标签）') : '';
    const rules = levels.filter(level => level.rule)
      .map(level => `<li><span>${esc(levelName(level, grouping) + tr(': ', '：'))}</span><span lang="en">${esc(level.rule)}</span></li>`)
      .join('');
    const rulesHtml = rules ? `<details><summary>${esc(tr('Grouping rules', '分组规则'))}</summary><ul>${rules}</ul></details>` : '';
    return `<li><span>${esc(tr(`${concept}: `, `${concept}：`))}${esc(names)}${esc(codesOnly)}</span>${rulesHtml}</li>`;
  }

  function notesHtml(summary, helpers) {
    const record = summary && summary.exposure_groups && typeof summary.exposure_groups === 'object'
      ? summary.exposure_groups : null;
    const { tr, esc } = helpers;
    const title = `<strong>${esc(tr('Exposure groups', '暴露分组'))}</strong>`;
    const levels = window.AGENT_EXPOSURE_LEVELS;
    if (record && record.status === 'unavailable' && levels) {
      return `<div class="gpi-exposure-groups">${title}<p>${esc(levels.unreadableText(tr))}</p></div>`;
    }
    const groupings = (record && record.status === 'shown' && Array.isArray(record.groupings) ? record.groupings : [])
      .filter(row => row && typeof row === 'object');
    const items = groupings.map(row => groupingHtml(row, helpers)).filter(Boolean);
    if (!items.length) return '';
    return `<div class="gpi-exposure-groups">${title}<ul>${items.join('')}</ul></div>`;
  }

  window.EasyICU.guidedPi.declare('exposureGroups', { notesHtml, levelName });
})();
