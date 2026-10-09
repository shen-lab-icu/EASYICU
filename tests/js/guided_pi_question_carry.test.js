/* A source confirmation carries only the researcher's own opening question
   into the study setup. The shell remembers the first message typed and sent
   as written, and the data-binding owner saves that text and nothing else: a
   starter card's or a model option's words never become the question. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const [modulesFile, bindingFile, shellFile] = process.argv.slice(2).map(file => path.resolve(file));
global.window = global;
require(modulesFile);
require(bindingFile);
const binding = global.EasyICU.guidedPi.require('dataBinding');

function harness(question, missing) {
  const writes = [];
  const session = { session_id: 's1', binding: { study_context_id: 'ctx' } };
  global.EU_STUDY_CONTEXT = {
    hydrate: async () => {}, active: () => ({ id: 'ctx' }), activate: async () => {},
    refreshActiveFromServer: async () => {}, update: patch => writes.push(patch), persist: async () => {},
  };
  const owner = binding.create({
    api: () => ({}), render() {}, projectId: () => 'p', loadWorkflow: async () => {}, dataConsent: {},
    errorText: String, rememberSession() {}, continueAfterDataSourceConfirmation() {},
    session: () => session, busy: () => false, setError() {}, setSession() {}, rebind: async () => {},
    workflow: () => ({ missing_setup_fields: missing }), researchQuestion: () => question,
  });
  return { owner, writes };
}

(async () => {
  const typed = 'In MIMIC-IV ICU stays, how common is Sepsis-3?';
  let run = harness(typed, ['question', 'data_source']);
  assert.equal(await run.owner.carryQuestionIntoSetup(), true);
  assert.deepEqual(run.writes, [{ question: typed }]);
  // No typed opening question: nothing is saved, and the conversation asks once.
  run = harness('', ['question']);
  assert.equal(await run.owner.carryQuestionIntoSetup(), false);
  assert.deepEqual(run.writes, []);
  // A setup that already has its question is left as it is.
  run = harness(typed, ['data_source']);
  assert.equal(await run.owner.carryQuestionIntoSetup(), false);
  assert.deepEqual(run.writes, []);

  // The shell's question is the first message typed and sent as written,
  // never read back from the message list.
  const shell = fs.readFileSync(shellFile, 'utf8');
  assert.match(shell, /const opening = !intent && decorated === text && !state\.messages\.some\(row => row && row\.role === 'user'\) \? text : '';/);
  assert.match(shell, /await sendText\(decorated, undefined, intent, true, '', opening\);/);
  assert.match(shell, /if \(visibleUserMessage && openingQuestion\) state\.openingQuestion = \{ sessionId: state\.session\.session_id, text: openingQuestion \};/);
  const reader = shell.slice(shell.indexOf('function conversationResearchQuestion()'), shell.indexOf('async function continueAfterDataSourceConfirmation()'));
  assert.match(reader, /state\.openingQuestion/);
  assert.doesNotMatch(reader, /state\.messages/);
  process.stdout.write('Only a typed opening question is carried into the setup.\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
