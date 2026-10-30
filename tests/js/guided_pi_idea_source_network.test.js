'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../src/easyicu/webserver/static/js/screens-guided-pi-idea-source.js'), 'utf8');

function fixture() {
  const calls = [];
  let owner;
  let response = { source_adapter: { status: 'metadata_fetched' }, suggested_payload: { title: 'Article' } };
  const window = {
    EU_LANG: 'en',
    EU_API: { resolveIdeaSource: async body => { calls.push(body); return response; } },
    EasyICU: { guidedPi: { declare: (_name, value) => { owner = value; } } },
  };
  vm.runInNewContext(source, { window, JSON, String, Number, Boolean, Array });
  const input = { value: '' };
  const callbacks = { host: () => ({ querySelector: () => input }), render: () => {}, tr: en => en };
  const toggle = () => owner.handleClick({ target: { closest: selector => selector === '[data-gpi-idea-url-network]' } }, callbacks);
  const selectSession = (sessionId, projectId = 'project-1') => owner.controls({
    tr: en => en, esc: String, icon: () => '', projectId, sessionId,
  });
  selectSession('session-1');
  return { owner, calls, input, toggle, selectSession, response: value => { response = value; } };
}

test('article resolution requires explicit consent for the exact link on each attempt', async () => {
  const f = fixture();
  const first = 'Read https://example.org/first';
  await assert.rejects(f.owner.prepareForMessage(first, 'idea_mining_entry'), /plus menu/);
  assert.equal(f.calls.length, 0);
  f.input.value = first;
  f.toggle();
  const result = await f.owner.prepareForMessage(first, 'idea_mining_entry');
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].allow_network, true);
  assert.equal(f.calls[0].url, 'https://example.org/first');
  assert.equal(result.title, 'Article');
  f.owner.consume();
  await assert.rejects(f.owner.prepareForMessage(first, 'idea_mining_entry'), /plus menu/);
  f.toggle();
  await assert.rejects(f.owner.prepareForMessage('Read https://example.org/second', 'idea_mining_entry'), /plus menu/);
  assert.equal(f.calls.length, 1);
});

test('a disabled connector or failed fetch is not represented as an article source', async () => {
  const f = fixture();
  f.input.value = 'https://example.org/article';
  f.toggle();
  f.response({ source_adapter: { status: 'blocked_network_opt_in_required' } });
  await assert.rejects(f.owner.prepareForMessage(f.input.value, 'idea_mining_entry'), /could not be retrieved/);
  assert.equal(f.owner.hasSource(), false);
  await assert.rejects(f.owner.prepareForMessage(f.input.value, 'idea_mining_entry'), /plus menu/);
  assert.equal(f.calls.length, 1);
});

test('reset clears consent and switching the link cannot reuse a resolved source', async () => {
  const f = fixture();
  f.input.value = 'https://example.org/first';
  f.toggle();
  f.owner.reset();
  await assert.rejects(f.owner.prepareForMessage(f.input.value, 'idea_mining_entry'), /plus menu/);
  f.toggle();
  await f.owner.prepareForMessage(f.input.value, 'idea_mining_entry');
  await assert.rejects(f.owner.prepareForMessage('https://example.org/second', 'idea_mining_entry'), /plus menu/);
  assert.equal(f.owner.hasSource(), false);
});

test('switching project or session clears consent even for the same link', async () => {
  const f = fixture();
  f.input.value = 'https://example.org/article';
  f.toggle();
  f.selectSession('session-2');
  await assert.rejects(f.owner.prepareForMessage(f.input.value, 'idea_mining_entry'), /plus menu/);
  f.toggle();
  f.selectSession('session-2', 'project-2');
  await assert.rejects(f.owner.prepareForMessage(f.input.value, 'idea_mining_entry'), /plus menu/);
  assert.equal(f.calls.length, 0);
});

test('a late response from a previous session cannot attach its source to another', async () => {
  const f = fixture();
  let finish;
  const wait = new Promise(resolve => { finish = resolve; });
  f.response(wait);
  f.input.value = 'https://example.org/article';
  f.toggle();
  const pending = f.owner.prepareForMessage(f.input.value, 'idea_mining_entry');
  f.selectSession('session-2');
  finish({ source_adapter: { status: 'metadata_fetched' }, suggested_payload: { title: 'Article' } });
  assert.equal(await pending, null);
  assert.equal(f.owner.hasSource(), false);
});

test('the attached article distinguishes abstract, missing excerpt and unverified user quote', async () => {
  for (const [origin, text] of [
    ['pubmed_abstract_excerpt', 'PubMed abstract excerpt'],
    ['user_supplied_unverified', 'User-supplied excerpt; unverified'],
    ['', 'No source excerpt available'],
  ]) {
    const f = fixture();
    f.input.value = 'https://pubmed.ncbi.nlm.nih.gov/26903338/';
    f.response({ source_adapter: { status: 'metadata_fetched' }, suggested_payload: {
      title: 'Article', excerpt: origin ? 'A bounded excerpt' : '', excerpt_origin: origin,
      user_supplied_title: 'A user title',
    } });
    f.toggle();
    const source = await f.owner.prepareForMessage(f.input.value, 'idea_mining_entry');
    assert.equal(source.excerpt_origin, origin);
    assert.equal(source.user_supplied_title, 'A user title');
    assert.match(f.owner.status({ tr: en => en, esc: String, projectId: 'project-1', sessionId: 'session-1' }), new RegExp(text));
  }
});
