'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// Optional owner paths let the same contract exercise the read-only base.
const owners = process.argv.length > 2 ? process.argv.slice(2) : [
  'html-escape.js', 'screens-guided-pi-modules.js', 'screens-guided-pi-hosted.js',
  'screens-guided-pi-starters.js', 'screens-guided-pi-effort-menu.js',
  'screens-guided-pi-help.js',
].map(name => path.join(__dirname, '../../src/easyicu/webserver/static/js', name));

function load(language, deployment) {
  const stored = new Map();
  const window = {
    EU_LANG: language,
    EU_CAPABILITIES: { deployment },
    EU_SETTINGS: { about: { version: 'test-version' } },
    localStorage: {
      getItem: key => stored.get(key) || null,
      setItem: (key, value) => stored.set(key, String(value)),
      removeItem: key => stored.delete(key),
    },
  };
  const tr = (en, zh) => window.EU_LANG === 'zh' ? zh : en;
  window.t = tr;
  const context = vm.createContext({ window });
  for (const owner of owners) vm.runInContext(fs.readFileSync(owner, 'utf8'), context, { filename: owner });
  return { window, tr, modules: window.EasyICU.guidedPi };
}

for (const language of ['zh', 'en']) {
  const zh = language === 'zh';
  for (const [name, deployment, hosted] of [
    ['desktop', { hosted: false }, false],
    ['hosted', { hosted: true }, true],
    ['older desktop', undefined, false],
    ['missing hosted flag', {}, false],
    ['non-boolean hosted flag', { hosted: 'true' }, false],
  ]) {
    test(`${language}: ${name} entry describes approval and the actual execution location`, () => {
      const { modules, tr } = load(language, deployment);
      const html = modules.require('starters').render({ tr });
      const location = hosted ? (zh ? '服务器上' : 'the server') : (zh ? '本机' : 'this computer');
      assert.ok(html.includes(zh
        ? `写下你的临床问题，选择数据，再核对分析计划。批准后在${location}运行分析。`
        : `Describe your clinical question, choose the data, then review the analysis plan. After approval, analysis runs on ${location}.`));
      assert.doesNotMatch(html, /两次确认|可以关闭|both|twice|two confirmations|close the (?:page|app|application)/i);
      if (!hosted) assert.doesNotMatch(html, /服务器|on the server/);
    });
  }

  test(`${language}: entry follows a deployment change without reloading the module`, () => {
    const { modules, tr, window } = load(language, { hosted: false });
    const render = () => modules.require('starters').render({ tr });
    assert.match(render(), zh ? /批准后在本机运行分析/ : /analysis runs on this computer/);
    window.EU_CAPABILITIES.deployment.hosted = true;
    assert.match(render(), zh ? /批准后在服务器上运行分析/ : /analysis runs on the server/);
  });

  test(`${language}: reasoning depth is readable in the chip, accessible name and menu`, () => {
    const { modules } = load(language);
    const menu = modules.require('effortMenu');
    const html = menu.render({ level: 'medium', iconHtml: () => '', disabled: false });
    assert.ok(html.includes(zh ? '<span>思考深度 · 中</span>' : '<span>Reasoning depth · Medium</span>'));
    assert.ok(html.includes(zh ? 'aria-label="思考深度：中"' : 'aria-label="Reasoning depth：Medium"'));
    assert.ok(html.includes(zh ? 'aria-label="本对话的思考深度"' : 'aria-label="Reasoning depth for this conversation"'));
    assert.doesNotMatch(html.replace(/<[^>]+>/g, ''), /努力|Effort|每轮更慢|slower and uses more/i);
    assert.ok(html.includes(zh ? '对话助手对疑难问题想得更深；回复更慢，用量更多。' : 'replies are slower and use more tokens.'));
  });

  for (const applied of ['minimal', 'off']) {
    test(`${language}: model-applied ${applied} stays distinct from the requested preference`, async () => {
      const { modules } = load(language);
      const menu = modules.require('effortMenu');
      let session = { session_id: 'synthetic-conversation', thinking_level: 'medium' };
      const calls = [];
      let rendered = 0;
      const callbacks = {
        session: () => session, busy: () => false, projectId: () => 'synthetic-project',
        api: () => ({ setPiCopilotThinkingLevel: async (id, body) => {
          calls.push([id, body.thinking_level, body.project_id]);
          return { thinking_level: applied, session: { ...session, thinking_level: applied } };
        } }),
        setSession: value => { session = value; }, render: () => { rendered += 1; },
        setError: error => assert.fail(error),
      };
      const choice = { dataset: { gpiEffortLevel: 'high' }, closest: () => null };
      assert.equal(menu.handleClick({ target: { closest: () => choice }, preventDefault() {} }, callbacks), true);
      await new Promise(resolve => setImmediate(resolve));
      assert.deepEqual(calls, [['synthetic-conversation', 'high', 'synthetic-project']]);
      assert.equal(session.thinking_level, applied);
      assert.equal(menu.preferred(), 'high');
      assert.equal(rendered, 1);
      const html = menu.render({ level: session.thinking_level, iconHtml: () => '', disabled: true });
      assert.ok(html.includes(`data-gpi-effort-level="${applied}" aria-pressed="true"`));
      assert.match(html, /aria-disabled="true"/);
      assert.equal((html.match(/data-gpi-effort-level=/g) || []).length, 4);
      if (applied === 'minimal') {
        assert.ok(html.includes(zh ? '模型自动采用此档位。' : 'The model selected this level automatically.'));
        assert.doesNotMatch(html, /收敛请求|clamped a request/);
      } else {
        assert.match(html, zh ? /不向模型指定推理档位/ : /No effort level is sent/);
      }
    });
  }

  test(`${language}: a failed depth change uses the same clinician-facing term`, async () => {
    const { modules } = load(language);
    const menu = modules.require('effortMenu');
    let error = '';
    const choice = { dataset: { gpiEffortLevel: 'high' }, closest: () => null };
    menu.handleClick({ target: { closest: () => choice }, preventDefault() {} }, {
      session: () => ({ session_id: 'synthetic-conversation', thinking_level: 'medium' }),
      busy: () => false, projectId: () => 'synthetic-project',
      api: () => ({ setPiCopilotThinkingLevel: async () => { throw new Error(); } }),
      setSession: () => assert.fail('a rejected change must not update the session'),
      setError: value => { error = value; }, render() {},
    });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(error, zh ? '思考深度未能修改。' : 'The reasoning depth could not be changed.');
  });

  test(`${language}: the existing Help entry still exposes the version and problem reference`, () => {
    const { modules, tr, window } = load(language);
    const help = modules.require('help').create({
      tr, esc: window.EU_HTML.esc,
      context: () => ({ projectId: 'synthetic-project', sessionId: 'synthetic-conversation' }),
    });
    assert.match(help.render(), /data-gpi-help-copy/);
    assert.match(help.referenceText(), /EasyICU test-version/);
    assert.match(help.referenceText(), /synthetic-project/);
  });
}
