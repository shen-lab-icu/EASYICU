/* Owner: shared Project Monitor demo fixtures. */
(function () {
  window.EU_AGENT_DEMO_STUDIES = Object.freeze([
    {
      id: 'sepsis', name: ['Sepsis mortality prediction', '脓毒症死亡率预测'],
      mode: 'analysis', status: 'gate', stage: 3, // 0 plan,1 build,2 analyze,3 gate,4 draft
      cohort: 'sepsis_mortality_demo', source: ['demo · 10 stays · 6 modules', '演示 · 10 次住院 · 6 模块'],
      question: [
        'Which first-24h bedside features best predict in-hospital mortality among Sepsis-3 patients, and how does adding lactate change calibration?',
        '在 Sepsis-3 患者中,入院前 24 小时的哪些床旁特征最能预测院内死亡?加入乳酸如何改变校准度?',
      ],
      runs: [
        ['run 07', ['ROC + calibration', 'ROC + 校准'], 'complete', '2m 14s', ['today 14:22', '今天 14:22']],
        ['run 06', ['Table 1 + missingness', 'Table 1 + 缺失审计'], 'complete', '1m 02s', ['today 11:08', '今天 11:08']],
        ['run 05', ['Cohort summary only', '仅队列摘要'], 'complete', '0:36', ['yesterday', '昨天']],
        ['run 04', ['Full plan (review-ready draft)', '完整计划(草稿待核验)'], 'blocked', '2m 41s', ['2 days ago', '2 天前']],
      ],
      signed: false,
    },
    {
      id: 'crossdb', name: ['Cross-DB sepsis replication', '跨库脓毒症复现'],
      mode: 'analysis', status: 'ready', stage: 2,
      cohort: 'sepsis_crossdb', source: ['demo · 3 databases · 6 concepts', '演示 · 3 个数据库 · 6 概念'],
      question: [
        'Does the sepsis mortality signal replicate across MIMIC-IV, eICU and AUMCdb, and where do feature distributions diverge?',
        '脓毒症死亡信号能否在 MIMIC-IV、eICU 和 AUMCdb 间复现?特征分布在哪里出现分歧?',
      ],
      runs: [
        ['run 02', ['Distribution deltas', '分布差异'], 'complete', '1m 48s', ['today 09:30', '今天 09:30']],
        ['run 01', ['Availability matrix', '可用性矩阵'], 'complete', '0:52', ['yesterday', '昨天']],
      ],
      signed: false,
    },
    {
      id: 'lactate', name: ['Early lactate research idea', '早期乳酸研究想法'],
      mode: 'analysis', status: 'idle', stage: 0,
      cohort: 'sepsis_mortality_demo', source: ['research idea · not yet run', '研究想法 · 尚未运行'],
      question: [
        'Test whether early lactate trajectory adds prognostic information after the idea has been confirmed in Idea Mining.',
        '在 Idea 挖掘确认后,检验早期乳酸轨迹是否提供额外预后信息。',
      ],
      runs: [
        ['idea 02', ['Feasibility handoff', '可行性交接'], 'complete', '0:41', ['today 13:05', '今天 13:05']],
        ['idea 01', ['Source check', '来源核查'], 'complete', '0:19', ['today 12:40', '今天 12:40']],
      ],
      signed: false,
    },
    {
      id: 'aki', name: ['AKI in CKD patients', 'CKD 患者的急性肾损伤'],
      mode: 'analysis', status: 'idle', stage: 0,
      cohort: 'aki_ckd_demo', source: ['demo · not yet run', '演示 · 尚未运行'],
      question: [
        'Among CKD patients, which interventions in the first 48h are associated with progression to KDIGO stage 3 AKI?',
        '在 CKD 患者中,前 48 小时的哪些干预与进展至 KDIGO 3 期 AKI 相关?',
      ],
      runs: [],
      signed: false,
    },
  ]);
})();
