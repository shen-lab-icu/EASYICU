'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const expected = {
  agentic_coder_credentialed_source_forbidden: [
    'The desktop app does not run model-written code on credentialed data such as MIMIC-IV or eICU. No analysis was run.',
    '桌面版不会在 MIMIC-IV、eICU 等需授权的数据上运行模型临时写的代码。尚未运行任何分析。',
  ],
};
const help = [
  'Yes. The research conversation runs on a model the administrator set up. It receives your messages, the analysis plan and its code, variable names, error details, and summary statistics in which very small counts are hidden. Patient-level rows are never sent. For credentialed databases such as MIMIC-IV and eICU, the PhysioNet data use agreement expects a model that runs on this server or a service that keeps no data.',
  '会。研究对话由管理员配置的模型支持。发给模型的有：你的消息、分析计划及其代码、变量名称、错误详情，以及隐藏了极小计数的汇总统计。患者的逐行记录不会发给模型。对于 MIMIC-IV、eICU 等需授权的数据库，PhysioNet 数据使用协议要求模型运行在本服务器上，或其服务不保留数据。',
];

for (const [index, language] of ['en', 'zh'].entries()) {
  const owners = {};
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const context = vm.createContext({
    t: tr,
    icon: () => '',
    window: {
      EU_LANG: language,
      EU_HTML: { esc: value => String(value) },
      EU_CAPABILITIES: { deployment: { hosted: true } },
      EasyICU: { guidedPi: {
        declare: (name, api) => { owners[name] = api; },
        optional: name => owners[name] || null,
      } },
    },
  });
  for (const path of process.argv.slice(2)) {
    vm.runInContext(fs.readFileSync(path, 'utf8'), context, { filename: path });
  }
  const errors = owners.errorText.create({ tr, staticPreview: () => false });
  for (const [code, sentences] of Object.entries(expected)) {
    for (const rendered of [errors.errorText({ code }), errors.jobErrorText(code), errors.runFailureText(code)]) {
      assert.equal(rendered, sentences[index], `${language}: ${code}`);
      assert.ok(rendered.trim().length > 0);
      assert.notEqual(rendered, code);
    }
  }
  const screens = context.window.SCREENS;
  const body = screens.tutorial.render();
  assert.ok(body.includes(help[index]), `${language}: exact hosted disclosure`);
  assert.equal(body.includes('TODO(lead copy)'), false);
  assert.deepEqual(JSON.parse(JSON.stringify(screens.llmEgressDisclosure)), {
    noticeKey: 'physionet_llm_data_handling_declaration',
    sent: [
      'user_messages', 'analysis_plans_and_code', 'schema_and_concept_metadata',
      'aggregate_statistics_with_small_cell_suppression', 'structured_diagnostics',
      'project_artifacts_after_privacy_projection',
    ],
  });
  assert.ok(Object.isFrozen(screens.llmEgressDisclosure));
  assert.ok(Object.isFrozen(screens.llmEgressDisclosure.sent));
}

console.log('Credentialed egress copy: both refusal codes and hosted Help pass in English and Chinese.');
