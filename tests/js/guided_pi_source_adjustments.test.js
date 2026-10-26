'use strict';

// Plan review words the host's database capability review; it never decides
// a substitution and says plainly when the plan has not applied one.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

function owner(language = 'zh') {
  const declared = {};
  const window = {
    EU_LANG: language,
    EU_CATALOG: { dict: { age: ['Age', '年龄', 'years'], lact: ['Lactate', '乳酸', 'mmol/L'] } },
    EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } },
  };
  const context = vm.createContext({ window });
  for (const file of ['database-names.js', 'html-escape.js', 'screens-guided-pi-source-adjustments.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return declared.sourceAdjustments;
}

const field = (table, column) => ({ source: 'data-sources.json', table, column });
const review = (decision, unapplied = []) => ({ decision, unapplied_substitutions: unapplied });

test('an exact study shows nothing', () => {
  assert.equal(owner().render(review({ status: 'exact', database: 'miiv', substitutions: [], missing: [] })), '');
  assert.equal(owner().render(null), '');
});

test('each substitution reads as planned → used, with its reason and source field', () => {
  const html = owner().render(review({
    status: 'substitution', database: 'hirid',
    substitutions: [
      { capability: 'analysis_unit', original: 'hospitalization', substitute: 'icu_stay',
        reason_code: 'hospitalization_identifier_unavailable', source_field: field('general', 'patientid') },
      { capability: 'outcome_meaning', original: 'hospital_death', substitute: 'icu_death',
        reason_code: 'mortality_meaning_differs', source_field: field('observations', 'value') },
      { capability: 'threshold:age', original: { concept: 'age', operator: '>=', value: 65 },
        substitute: { concept: 'age', operator: '>=', value: 60 },
        reason_code: 'numeric_threshold_aligned_to_band', source_field: field('admissions', 'agegroup') },
      { capability: 'something_new', original: 'a', substitute: 'b', reason_code: 'x', source_field: null },
    ],
    missing: [],
  }));
  assert.match(html, /<h4>HiRID 上的口径调整<\/h4>/);
  const rows = html.split('<li>').slice(1);
  assert.equal(rows.length, 3, 'an unknown capability is not invented into a sentence');
  assert.match(rows[0], /<s>按住院<\/s><span aria-hidden="true">→<\/span><strong>按 ICU 入住<\/strong>/);
  assert.match(rows[0], /HiRID 没有住院标识[\s\S]*<code>general\.patientid<\/code>/);
  assert.match(rows[1], /<s>院内死亡<\/s>[\s\S]*<strong>ICU 内死亡<\/strong>/);
  assert.match(rows[2], /<s>年龄 ≥ 65 岁<\/s>[\s\S]*<strong>年龄 ≥ 60 岁<\/strong>[\s\S]*年龄按区间记录/);
  assert.match(rows[2], /aria-label="原定：年龄 ≥ 65 岁；改为：年龄 ≥ 60 岁"/);
  assert.match(html, /<p class="">批准计划即同意按以上口径分析。<\/p>/);
});

test('a plan that has not applied the substitution cannot be approved', () => {
  const html = owner().render(review({
    status: 'substitution', database: 'sic',
    substitutions: [{ capability: 'outcome_meaning', original: 'hospital_death', substitute: 'icu_death',
      reason_code: 'mortality_meaning_differs', source_field: field('cases', 'OffsetOfDeath') }],
    missing: [],
  }, ['outcome_meaning']));
  assert.match(html, /<p class="is-blocked">计划还没有按这些口径修改，暂不能批准。请重新生成计划。<\/p>/);
  assert.doesNotMatch(html, /批准计划即同意/);
});

test('a refusal names what the database lacks, not the database', () => {
  const api = owner();
  const html = api.render(review({
    status: 'refusal', database: 'aumc', substitutions: [],
    missing: [
      { capability: 'hospital_los', code: 'hospital_los_unavailable', source: 'concept-dict.json' },
      { capability: 'concept:lact', code: 'required_concept_unmapped', source: 'concept-dict.json' },
      { capability: 'threshold:age', code: 'numeric_threshold_not_resolvable', source: 'admissions.agegroup' },
      { capability: 'hospital_los', code: 'hospital_los_unavailable', source: 'concept-dict.json' },
      { capability: 'x', code: 'unknown_future_code', source: '' },
    ],
  }));
  assert.match(html, /class="gpi-source-adjustments is-refusal"><h4>这项研究在 AmsterdamUMCdb 上做不了<\/h4>/);
  assert.deepEqual([...html.matchAll(/<li>([^<]*)<\/li>/g)].map(match => match[1]),
    ['没有住院时长', '没有乳酸的数据', '年龄按区间记录，无法按这个数值筛选']);
  assert.equal(api.databaseName('mimic_demo'), 'MIMIC-III Demo');
});

test('values are escaped and English reads naturally', () => {
  const html = owner('en').render(review({
    status: 'substitution', database: '<img src=x>',
    substitutions: [{ capability: 'event_time', original: 'recorded_death_time', substitute: 'date',
      reason_code: 'event_time_is_proxy_or_date', source_field: field('t<b>', 'c') }],
    missing: [],
  }));
  assert.doesNotMatch(html, /<img|<b>/);
  assert.match(html, /How &lt;img src=x&gt; changes this study/);
  assert.match(html, /<s>Recorded time of death<\/s>[\s\S]*<strong>Date of death only<\/strong>/);
  assert.match(html, /Approving the plan accepts these changes\./);
});

test('ICU discharge and last observation proxies are named before plan approval in both languages', () => {
  for (const [substitute, en, zh, table, column] of [
    ['icu_discharge_proxy', 'ICU discharge time used as a proxy for death time', '以 ICU 出科时间近似死亡时间', 'stays', 'exit_time'],
    ['last_observation_proxy', 'Last observation time used as a proxy for death time', '以最后观察时间近似死亡时间', 'measurements', 'time'],
  ]) {
    for (const [language, label] of [['en', en], ['zh', zh]]) {
      const html = owner(language).render(review({ status: 'substitution', database: 'synthetic', missing: [],
        substitutions: [{ capability: 'event_time', original: 'recorded_death_time', substitute,
          reason_code: 'event_time_is_proxy_or_date', source_field: field(table, column) }] }));
      assert.ok(html.includes(`<strong>${label}</strong>`));
      assert.ok(html.includes(`<code>${table}.${column}</code>`));
      assert.ok(html.includes(language === 'zh' ? '批准计划即同意' : 'Approving the plan accepts'));
      assert.ok(!html.includes(substitute));
    }
  }
});

test('mortality alternatives name their horizon and censoring before confirmation', () => {
  for (const language of ['en', 'zh']) {
    for (const days of [28, 90, 365]) {
      const html = owner(language).render(review({ status: 'substitution', database: 'synthetic', missing: [],
        substitutions: [
          { capability: 'outcome_meaning', original: 'hospital_death', substitute: `mort_${days}d`,
            source_field: field('derived_outcomes', `mort_${days}d`) },
          { capability: 'event_time', original: 'recorded_death_time', substitute: `followup_days_${days}d`,
            reason_code: 'fixed_horizon_event_and_censor_time', source_field: field('derived_outcomes', `followup_days_${days}d`) },
          { capability: 'censor_time', original: 'hospital_discharge', substitute: `followup_days_${days}d`,
            source_field: field('derived_outcomes', `followup_days_${days}d`) },
        ] }));
      assert.equal(html.split('<li>').length - 1, 3);
      assert.ok(html.includes(language === 'zh' ? `${days} 天内死亡` : `Death within ${days} days`));
      assert.ok(html.includes(language === 'zh' ? 'ICU 入科' : 'ICU admission'));
      assert.ok(html.includes(language === 'zh' ? '有随访记录' : 'documented follow-up'));
      assert.ok(html.includes(`<code>derived_outcomes.followup_days_${days}d</code>`));
      assert.ok(html.includes(language === 'zh' ? '批准计划即同意' : 'Approving the plan accepts'));
      assert.doesNotMatch(html, /no exact time of death|没有精确的死亡时间/);
    }
    const html = owner(language).render(review({ status: 'substitution', database: 'synthetic', missing: [],
      substitutions: [{ capability: 'censor_time', original: 'hospital_discharge', substitute: 'icu_discharge',
        source_field: field('stays', 'exit_time') }] }));
    assert.ok(html.includes(language === 'zh' ? '在 ICU 出科时结束随访' : 'Follow-up ends at ICU discharge'));
    assert.ok(html.includes('<code>stays.exit_time</code>'));
  }
});

test('a band-aligned cut-off says who it adds or drops', () => {
  const swap = (concept, from, to, database = 'hirid') => ({ status: 'substitution', database, missing: [],
    substitutions: [{ capability: `threshold:${concept}`, original: { concept, ...from }, substitute: { concept, ...to },
      reason_code: 'numeric_threshold_aligned_to_band', source_field: field('general', concept) }] });
  const cases = [
    ['zh', 'age', { operator: '>=', value: 18 }, { operator: '>=', value: 20 }, '18–19 岁的患者不纳入。'],
    ['en', 'age', { operator: '>=', value: 18 }, { operator: '>=', value: 20 }, 'Patients aged 18–19 are not included.'],
    ['zh', 'age', { operator: '>=', value: 17 }, { operator: '>=', value: 15 }, '15–16 岁的患者也会纳入。'],
    ['en', 'age', { operator: '>', value: 64 }, { operator: '>=', value: 65 }, ''],
    ['en', 'age', { operator: '<=', value: 66 }, { operator: '<', value: 65 }, 'Patients aged 65–66 are not included.'],
    ['en', 'age', { operator: '>=', value: 89 }, { operator: '>=', value: 90 }, 'Patients aged 89 are not included.'],
    ['zh', 'weight', { operator: '>=', value: 48 }, { operator: '>=', value: 50 }, '在 48 kg 至不满 50 kg 的患者不纳入。'],
    ['en', 'weight', { operator: '<', value: 102 }, { operator: '<', value: 100 }, 'from 100 to under 102 kg are not included.'],
    ['en', 'weight', { operator: '<=', value: 102 }, { operator: '<', value: 100 }, ''],
  ];
  for (const [language, concept, from, to, consequence] of cases) {
    const html = owner(language).render(review(swap(concept, from, to)));
    const reason = html.match(/<p>([^<]*)<code>/)[1];
    assert.ok(reason.includes(language === 'zh' ? '切点改为最近的区间边界' : 'nearest band edge'));
    if (consequence) assert.ok(reason.includes(consequence), `${language} ${concept} ${JSON.stringify(from)}: ${reason}`);
    else assert.doesNotMatch(reason, /included|纳入/, `${language} ${concept} ${JSON.stringify(from)}: ${reason}`);
  }
});

test('hospital death censored at hospital discharge says why follow-up continues past the ICU', () => {
  for (const [language, text] of [
    ['zh', '院内死亡包括出 ICU 后的死亡，所以随访到出院时结束'],
    ['en', 'In-hospital death includes deaths after ICU discharge, so follow-up ends at hospital discharge'],
  ]) {
    const html = owner(language).render(review({ status: 'substitution', database: 'eicu', missing: [],
      substitutions: [{ capability: 'censor_time', original: 'icu_discharge', substitute: 'hospital_discharge',
        reason_code: 'mortality_censoring_differs', source_field: field('patient', 'hospitaldischargeoffset') }] }));
    assert.ok(html.includes(text));
    assert.ok(html.includes(language === 'zh' ? '<s>ICU 出科</s>' : '<s>ICU discharge</s>'));
    assert.ok(html.includes(language === 'zh' ? '<strong>医院出院</strong>' : '<strong>Hospital discharge</strong>'));
    assert.doesNotMatch(html, /Follow-up ends at ICU discharge|在 ICU 出科时结束随访/);
  }
});
