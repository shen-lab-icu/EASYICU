/* Owner: display names of the study families the host registers. */
/* Keys are the host's registered family keys (planning/family_spec registry);
   aliases are other spellings an older record may carry. A key this owner
   does not know gets no invented name: callers show their own fallback. */
(function () {
  'use strict';

  const FAMILIES = [
    ['descriptive_epidemiology', 'Descriptive study', '描述性研究'],
    ['association_study', 'Association study', '关联研究'],
    ['ordinal_dose_response', 'Dose–response study', '剂量反应研究'],
    ['prediction_model', 'Prediction model', '预测模型'],
    ['dynamic_prediction', 'Dynamic prediction', '动态预测'],
    ['survival', 'Survival analysis', '生存分析', ['survival_analysis']],
    ['causal_inference', 'Causal inference', '因果推断'],
    ['treatment_response', 'Treatment response', '治疗反应'],
    ['trajectory_clustering', 'Trajectory clustering', '轨迹聚类'],
    ['multimodal', 'Multimodal study', '多模态研究'],
    ['validation', 'External validation', '外部验证'],
    ['cross_database_replication', 'Cross-database replication', '跨库重复'],
    ['reinforcement_learning', 'Reinforcement learning', '强化学习'],
    ['data_quality_audit', 'Data-quality audit', '数据质量审计'],
    ['measurement_bias_audit', 'Measurement-bias audit', '测量偏倚审计'],
    ['cohort_definition_sensitivity', 'Cohort-definition sensitivity', '队列定义敏感性'],
    ['score_policy_sensitivity', 'Score-rule sensitivity', '评分规则敏感性'],
  ];
  const BY_KEY = new Map();
  FAMILIES.forEach(([key, en, zh, aliases]) => [key, ...(aliases || [])].forEach(spelling => BY_KEY.set(spelling, { key, en, zh })));
  const ORDER = new Map(FAMILIES.map(([key], index) => [key, index]));

  window.EU_STUDY_FAMILY_NAMES = Object.freeze({
    /** "Survival analysis" / "生存分析"; '' for a family this owner does not know. */
    name(value) {
      const row = BY_KEY.get(String(value == null ? '' : value).trim());
      return row ? (window.EU_LANG === 'zh' ? row.zh : row.en) : '';
    },
    /** Display order; unknown families sort after known ones. */
    order(value) {
      const row = BY_KEY.get(String(value == null ? '' : value).trim());
      return row ? ORDER.get(row.key) : FAMILIES.length;
    },
  });
})();
