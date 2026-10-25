/* Owner: host-owned study-family and variance readings in the confirmation.
   Display only. This owner neither re-reads the question nor writes study
   fields, resolves conflicts or grants approval. */
(function () {
  'use strict';

  const FAMILIES = {
    descriptive_epidemiology: ['descriptive research', '描述性研究'],
    association_study: ['association research', '关联研究'],
    prediction_model: ['prediction research', '预测研究'],
    survival_analysis: ['survival research', '生存研究'],
    causal_inference: ['causal research', '因果研究'],
    trajectory_clustering: ['trajectory research', '轨迹研究'],
  };
  const clip = value => String(value || '').slice(0, 300);
  const knownFamily = family => Object.prototype.hasOwnProperty.call(FAMILIES, family);
  function familyName(family, tr) {
    const pair = knownFamily(family) ? FAMILIES[family] : null;
    return pair ? tr(...pair) : tr('another research type', '另一种研究类型');
  }
  function notesHtml(workflow, { tr, esc }) {
    const receipt = workflow && workflow.study_setup_receipt;
    const configuration = receipt && receipt.configuration || {};
    const reading = configuration.study_family_reading;
    const variance = configuration.design_variance_basis;
    const lines = [];
    const details = [];
    if (reading && reading.source === 'question_wording' && knownFamily(reading.family)) {
      const name = familyName(reading.family, tr);
      const quotes = (Array.isArray(reading.elements) ? reading.elements : [])
        .slice(0, 4).filter(row => row && row.evidence).map(row => `“${clip(row.evidence)}”`).join(' / ');
      lines.push(tr(`Study type: ${name}.`, `研究类型：${name}。`));
      if (quotes) details.push(tr(`Read from the question: ${quotes}.`, `题目中的依据：${quotes}。`));
      const conflict = reading.conflict && reading.conflict.design_family;
      if (conflict) lines.push(tr(
        `The question and saved design differ: the saved design is ${familyName(conflict, tr)}. Review this difference; the design has not been changed automatically.`,
        `题目与当前设计不一致：已保存设计是${familyName(conflict, tr)}。请核对这一差异，系统没有自动修改设计。`,
      ));
    }
    if (variance && variance.conflict === 'counts_only_not_requested') lines.push(tr(
      'The current design reports counts without intervals, but no researcher request for counts only was recorded.',
      '当前设计只报告计数、不计算区间，但没有记录到研究者要求只计数。',
    ));
    else if (variance && variance.basis === 'user_words' && variance.evidence) lines.push(tr(
      `Counts without intervals were requested by you: “${clip(variance.evidence)}”.`,
      `你要求只报告计数、不计算区间，依据：“${clip(variance.evidence)}”。`,
    ));
    else if (variance && variance.replaced === 'none_counts_only') {
      if (variance.basis === 'source_patient_grouping') {
        lines.push(tr('Confidence intervals will account for repeated stays from the same patient.', '同一患者的多次入住将一并考虑，用于估计置信区间。'));
        details.push(tr('Patient-clustered uncertainty uses verified source grouping; it replaces counts-only inference that you did not request.', '按患者聚类估计不确定性，使用数据源已核验的患者分组；这替换了未经你要求的仅计数方案。'));
      }
      else if (variance.basis === 'source_without_patient_grouping') lines.push(tr(
        'EasyICU replaced unrequested counts-only inference with model-based uncertainty; the source has no verified patient grouping.',
        '数据源没有可核验的患者分组；EasyICU 改为模型方差估计，替换了未被请求的仅计数方案。',
      ));
    }
    if (!lines.length) return '';
    return `<div class="gpi-question-requirements gpi-study-readings"><strong>${esc(tr('Study type and uncertainty', '研究类型与置信区间'))}</strong><ul>${lines.map(line => `<li>${esc(line)}</li>`).join('')}</ul>${details.length ? `<details><summary>${esc(tr('Reading and method details', '查看判断依据与方法'))}</summary><ul>${details.map(line => `<li>${esc(line)}</li>`).join('')}</ul></details>` : ''}</div>`;
  }
  window.EasyICU.guidedPi.declare('studyReadings', { notesHtml });
})();
