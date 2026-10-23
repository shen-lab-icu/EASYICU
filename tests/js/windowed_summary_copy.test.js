'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const expected = {
  windowed_summary_statistic_unresolved: [
    'The question names a measurement but not which summary to report, such as its highest, lowest or mean value over a time window. State it, for example the highest lactate in the first 24 hours after ICU admission, then generate the plan again.',
    '问题提到了一项测量，但没有说明要报告哪种汇总值，例如某个时间窗内的最高值、最低值或平均值。请写明，例如入 ICU 后最初 24 小时内的最高乳酸，然后重新生成计划。',
  ],
  windowed_summary_statistic_unsupported: [
    'EasyICU can summarize a measurement over a time window only as its highest, lowest, mean, first or last value. Choose one of these, then generate the plan again.',
    'EasyICU 只能把时间窗内的测量值汇总为最高值、最低值、平均值、首次值或末次值。请从中选择一种，然后重新生成计划。',
  ],
  windowed_summary_window_required: [
    'A summary of a measurement needs a time window, for example the first 24 hours after ICU admission. State the window in the question, then generate the plan again.',
    '对测量值做汇总需要时间窗，例如入 ICU 后最初 24 小时。请在问题里写明时间窗，然后重新生成计划。',
  ],
  windowed_summary_window_unresolved: [
    'EasyICU could not read a single time window for this measurement from the question. State one window, for example the first 24 hours after ICU admission, then generate the plan again.',
    'EasyICU 无法从问题中读出这项测量的唯一时间窗。请写明一个时间窗，例如入 ICU 后最初 24 小时，然后重新生成计划。',
  ],
  windowed_summary_window_unsupported: [
    'EasyICU can summarize a measurement only over a window that starts at ICU admission, such as the first 24 hours. Change the window, then generate the plan again.',
    'EasyICU 目前只能对从入 ICU 时开始的时间窗做测量汇总，例如最初 24 小时。请修改时间窗，然后重新生成计划。',
  ],
  windowed_summary_concept_not_continuous: [
    'This variable is not a continuous measurement, so a highest, lowest or mean value over a time window does not apply. Ask about it another way, for example whether it occurred, then generate the plan again.',
    '这个变量不是连续测量值，不能取时间窗内的最高、最低或平均值。请换一种问法，例如是否发生，然后重新生成计划。',
  ],
  windowed_summary_materialization_window_mismatch: [
    'The prepared data covers a different time window from the one the question states, so the summary would not match the question. No analysis was run. Prepare the data again for the stated window.',
    '已准备的数据覆盖的时间窗与问题写明的不一致，汇总结果会与问题不符。尚未运行任何分析。请按问题写明的时间窗重新准备数据。',
  ],
};

for (const [index, language] of ['en', 'zh'].entries()) {
  const owners = {};
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const context = vm.createContext({ window: {
    EU_LANG: language,
    EU_HTML: { esc: value => String(value) },
    EasyICU: { guidedPi: {
      declare: (name, api) => { owners[name] = api; },
      optional: name => owners[name] || null,
    } },
  } });
  for (const path of process.argv.slice(2)) {
    vm.runInContext(fs.readFileSync(path, 'utf8'), context, { filename: path });
  }
  const errors = owners.errorText.create({ tr, staticPreview: () => false });
  for (const [code, sentences] of Object.entries(expected)) {
    assert.equal(errors.runFailureText(code), sentences[index], `${language}: ${code}`);
  }
}

console.log('Windowed summary copy: seven refusal codes pass verbatim in English and Chinese.');
