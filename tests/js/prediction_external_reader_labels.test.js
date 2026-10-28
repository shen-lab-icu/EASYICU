'use strict';
const assert = require('node:assert/strict');
global.window = { t: (_en, zh) => zh };
require('../../src/easyicu/webserver/static/js/screens-agent-reader-vocab.js');
for (const [product, expected] of [
  ['table:frozen_prediction_model', '冻结的已拟合预测模型'],
  ['dataset:external_validation_eicu', '独立外部验证队列'],
  ['table:external_validation', '外部验证性能'],
  ['table:external_calibration', '外部校准'],
  ['table:external_case_mix', '开发与验证病例构成'],
  ['table:external_decision_curve', '外部验证决策曲线'],
  ['log:external_validation_provenance', '模型冻结与外部验证来源记录'],
  ['report:external_validation_tripod', '外部验证 TRIPOD 清单'],
]) {
  assert.equal(window.AGENT_READER_VOCAB.tableTitle(product), expected, product);
}
console.log('8 external-validation reader labels passed');
