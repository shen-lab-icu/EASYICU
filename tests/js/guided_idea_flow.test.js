/* Executable contract for the Guided Idea Mining sub-flow.

   The guided screen is a 6000-line single IIFE, and everything below was
   closure-private: the only coverage was Python tests grepping the file for
   substrings, which cannot tell a behaviour change from a reformat and breaks
   on a move even when nothing changed.

   These assertions are written against the function text, so the same file
   runs them before and after the sub-flow is extracted into its own owner —
   that is what makes the extraction checkable rather than hopeful.

   Usage: node tests/js/guided_idea_flow.test.js <file that owns the flow> */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.resolve(process.argv[2]), 'utf8');

function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} must exist in ${path.basename(process.argv[2])}`);
  const bodyStart = source.indexOf('{', start);
  let depth = 0;
  for (let i = bodyStart; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error(`Could not extract ${name}`);
}

const NAMES = [
  'resetGuidedIdeaState',
  'guidedIdeaPayload',
  'guidedIdeaHasInput',
  'guidedIdeaSelected',
  'guidedIdeaStatusText',
  'guidedIdeaPreExperimentReady',
  'isGuidedIdeaIntent',
  'slotSnapshot',
  'restoreSlot',
];

/* restoreSlot re-fetches the run's artifacts; that is a network call, so the
   harness stubs it and asserts it was asked for instead. */
const artifactFetches = [];

const sandbox = {
  restoreArtifacts: (runId) => { artifactFetches.push(runId); },
  guidedIdea: null,
  guidedIdeaProvider: null,
  guidedLiteratureBrowser: null,
  createGuidedIdeaProviderState: () => ({ provider: 'openai', configOpen: false }),
  t: (en) => en,
  esc: (value) => String(value == null ? '' : value),
};
vm.createContext(sandbox);
vm.runInContext(
  `${NAMES.map(extract).join('\n')}\nthis.api = { ${NAMES.join(', ')} };`,
  sandbox,
);
const api = sandbox.api;

/* ---------------------------------------------------------------- state */

api.resetGuidedIdeaState();
const fresh = sandbox.guidedIdea;

assert(fresh, 'reset must install a state object');
assert.equal(fresh.sourceType, 'manual', 'a new idea starts from manual entry');
assert.equal(fresh.allowNetwork, false, 'network access is opt-in, never a default');
assert.equal(fresh.dataContextConfirmed, false, 'data context starts unconfirmed');
assert.equal(fresh.sourceEditorOpen, true);
for (const output of ['resolved', 'result', 'planDraft', 'prior', 'handoff', 'project']) {
  assert.equal(fresh[output], null, `${output} must start empty`);
}
for (const busy of ['resolving', 'mining', 'planning', 'priorArting', 'handoffing']) {
  assert.equal(fresh[busy], false, `${busy} must start idle`);
}
// Structural compare: the object is built inside the vm realm, so its
// prototype is not the host's and deepStrictEqual would reject an equal value.
assert.deepEqual(
  JSON.parse(JSON.stringify(sandbox.guidedLiteratureBrowser)),
  { open: false, loading: false, error: null, data: null, path: '' },
  'reset must also clear the literature browser',
);

/* -------------------------------------------------------------- payload */

api.resetGuidedIdeaState();
Object.assign(sandbox.guidedIdea, {
  topic: '  sepsis fluid strategy  ',
  doi: ' 10.1000/xyz ',
  literaturePdfCount: '7',
  allowNetwork: 1,
});
const payload = api.guidedIdeaPayload();

assert.equal(payload.topic, 'sepsis fluid strategy', 'free text is trimmed');
assert.equal(payload.doi, '10.1000/xyz');
assert.equal(payload.literature_pdf_count, 7, 'counts cross the wire as numbers');
assert.equal(payload.allow_network, true, 'the opt-in flag crosses as a boolean');
assert.equal(payload.source_type, 'manual');
// The wire contract is snake_case; the client state is camelCase.
assert('source_file_sha256' in payload && 'literature_folder' in payload);
assert.equal(payload.excerpt, '');

sandbox.guidedIdea = null;
assert.equal(
  api.guidedIdeaPayload().source_type,
  'manual',
  'a payload requested before reset must self-initialise, not throw',
);

/* ------------------------------------------------------------- hasInput */

api.resetGuidedIdeaState();
assert.equal(api.guidedIdeaHasInput(), false, 'an untouched form has no input');
for (const [field, value] of [
  ['topic', 'sepsis'],
  ['excerpt', 'a paragraph'],
  ['title', 'A paper'],
  ['url', 'https://example.org/x'],
  ['doi', '10.1000/xyz'],
  ['pmid', '12345678'],
  ['sourceFileSha256', 'abc'],
  ['literatureFolder', '/tmp/pdfs'],
]) {
  api.resetGuidedIdeaState();
  sandbox.guidedIdea[field] = value;
  assert.equal(api.guidedIdeaHasInput(), true, `${field} alone must count as input`);
}
// Whitespace is not input.
api.resetGuidedIdeaState();
sandbox.guidedIdea.topic = '   ';
assert.equal(api.guidedIdeaHasInput(), false);

/* ------------------------------------------------------------- selected */

api.resetGuidedIdeaState();
assert.equal(api.guidedIdeaSelected(), null, 'no result means no selected idea');

sandbox.guidedIdea.result = {
  idea_ledger: [{ idea_id: 'a' }, { idea_id: 'b' }],
  selected_idea_id: 'b',
};
assert.equal(api.guidedIdeaSelected().idea_id, 'b', 'the explicit selection wins');

sandbox.guidedIdea.result.selected_idea_id = 'missing';
assert.equal(
  api.guidedIdeaSelected().idea_id,
  'a',
  'an unknown selection falls back to the first idea, never to undefined',
);

sandbox.guidedIdea.result = { idea_ledger: [] };
assert.equal(api.guidedIdeaSelected(), null);

/* ----------------------------------------------------------- statusText */

/* The order of this chain is what the user reads. A running step must win
   over a finished one, and an error must win over both. */
api.resetGuidedIdeaState();
assert.match(api.guidedIdeaStatusText(), /Add a source clue or topic/);

sandbox.guidedIdea.resolved = { source_adapter: { status: 'ok' } };
assert.match(api.guidedIdeaStatusText(), /Source metadata resolved/);

sandbox.guidedIdea.resolved = {
  source_adapter: { status: 'failed', reason: 'no such doi' },
};
assert.match(api.guidedIdeaStatusText(), /Source resolver returned failed: no such doi/);

api.resetGuidedIdeaState();
sandbox.guidedIdea.result = { idea_ledger: [] };
assert.match(
  api.guidedIdeaStatusText(),
  /Confirm a local export\/cohort\/module context/,
  'an unconfirmed data context must be stated before feasibility',
);

sandbox.guidedIdea.dataContextConfirmed = true;
assert.match(api.guidedIdeaStatusText(), /Data context confirmed/);

sandbox.guidedIdea.handoff = { handoff_id: 'h1' };
assert.match(api.guidedIdeaStatusText(), /Handoff draft is frozen/);

sandbox.guidedIdea.project = { ok: true };
assert.match(
  api.guidedIdeaStatusText(),
  /metadata-only|Metadata-only/,
  'a created project seed must not read as an analysis run',
);

sandbox.guidedIdea.mining = true;
assert.match(
  api.guidedIdeaStatusText(),
  /Mining local idea ledger/,
  'a running step outranks every finished one',
);

/* The chain checks every busy flag before it checks `error`, so a step that
   is running now hides an error left over from the previous one. That is the
   behaviour the handlers rely on — each clears `error` as it starts — and it
   is pinned here so a reorder has to be deliberate. */
sandbox.guidedIdea.error = 'provider refused';
assert.match(
  api.guidedIdeaStatusText(),
  /Mining local idea ledger/,
  'a busy step still wins while it is running',
);

sandbox.guidedIdea.mining = false;
assert.equal(
  api.guidedIdeaStatusText(),
  'provider refused',
  'once nothing is running, the error outranks every finished state',
);

/* -------------------------------------------------- pre-experiment gate */

assert.equal(api.guidedIdeaPreExperimentReady(null), false);
assert.equal(api.guidedIdeaPreExperimentReady({}), false);
assert.equal(api.guidedIdeaPreExperimentReady({ pre_experiment: {} }), false);
assert.equal(
  api.guidedIdeaPreExperimentReady({ pre_experiment: { status: 'ready' } }),
  true,
);
for (const status of [
  'blocked',
  'missing_export',
  'not_configured',
  'no_active_export',
  'failed',
  'unavailable',
]) {
  assert.equal(
    api.guidedIdeaPreExperimentReady({ pre_experiment: { status } }),
    false,
    `${status} must not read as ready`,
  );
  assert.equal(
    api.guidedIdeaPreExperimentReady({ pre_experiment: { status: status.toUpperCase() } }),
    false,
    `${status} must not pass by changing case`,
  );
}

/* ---------------------------------------------------------------- intent */

for (const text of [
  'I have a study idea',
  'here is a paper',
  'mine this pdf',
  'literature review topic',
  '我有一个研究想法',
  '帮我挖掘一下文献',
  '这篇论文',
  '选题',
]) {
  assert.equal(api.isGuidedIdeaIntent(text), true, `should route to idea mining: ${text}`);
}
for (const text of ['', null, 'run the extraction', '导出队列']) {
  assert.equal(api.isGuidedIdeaIntent(text), false, `should not route to idea: ${text}`);
}

/* ------------------------------------------------------- session slots */

/* The guided shell used to build this section itself, reaching into the idea
   state 45 times, and restore it field by field. The field names and the exact
   fallbacks are the wire format of a session already saved to disk, so they are
   pinned here rather than left to whoever next edits the snapshot. */

api.resetGuidedIdeaState();
Object.assign(sandbox.guidedIdea, {
  sourceType: 'literature_folder',
  topic: 'fluids',
  allowNetwork: true,
  planEdits: 'edits',
  dataContextConfirmed: true,
  result: {
    run_id: 'run_1',
    idea_ledger: [{ idea_id: 'i1' }, { idea_id: 'i2' }],
    selected_idea_id: 'i2',
  },
  planDraft: { created_at: '2026-08-16T00:00:00Z' },
  handoff: { handoff_id: 'h9' },
  project: { project: { project_dir: '/tmp/p' } },
});
const slot = JSON.parse(JSON.stringify(api.slotSnapshot()));

assert.equal(slot.source_type, 'literature_folder');
assert.equal(slot.topic, 'fluids');
assert.equal(slot.allow_network, true);
assert.equal(slot.data_context_confirmed, true);
assert.equal(slot.run_id, 'run_1');
assert.equal(slot.selected_idea_id, 'i2', 'the chosen idea has to survive a reload');
assert.equal(slot.plan_created_at, '2026-08-16T00:00:00Z');
assert.equal(slot.handoff_id, 'h9');
assert.equal(slot.agent_project_dir, '/tmp/p');
// camelCase in the client, snake_case on the wire — the mapping is the contract.
assert(!('sourceType' in slot) && !('allowNetwork' in slot));

api.resetGuidedIdeaState();
assert.equal(api.slotSnapshot().topic, '', 'a fresh state still snapshots');
sandbox.guidedIdea = null;
assert.equal(api.slotSnapshot(), null, 'no idea, no idea slot');

/* A session that was saved mid-run must come back pointing at that run, with
   the source editor closed because the source is already resolved. */
api.restoreSlot(slot);
assert.equal(sandbox.guidedIdea.sourceType, 'literature_folder');
assert.equal(sandbox.guidedIdea.topic, 'fluids');
assert.equal(sandbox.guidedIdea.allowNetwork, true);
assert.equal(sandbox.guidedIdea.dataContextConfirmed, true);
assert.equal(sandbox.guidedIdea.result.run_id, 'run_1');
assert.equal(sandbox.guidedIdea.sourceEditorOpen, false);
assert.deepEqual(
  artifactFetches,
  ['run_1'],
  'restoring a mid-run session must re-fetch that run, and only that run',
);

const again = JSON.parse(JSON.stringify(api.slotSnapshot()));
for (const field of [
  'source_type', 'topic', 'excerpt', 'title', 'journal', 'year', 'doi', 'pmid',
  'url', 'allow_network', 'plan_edits', 'data_context_confirmed', 'run_id',
]) {
  assert.deepEqual(again[field], slot[field], `${field} must survive save -> restore`);
}

/* Restoring a session that never reached idea mining must not fabricate one. */
sandbox.guidedIdea = null;
api.restoreSlot(null);
assert.equal(sandbox.guidedIdea, null);
api.restoreSlot('not an object');
assert.equal(sandbox.guidedIdea, null);

artifactFetches.length = 0;
api.restoreSlot({ topic: 'no run yet' });
assert.deepEqual(artifactFetches, [], 'no run id, no fetch');
assert.equal(
  sandbox.guidedIdea.sourceEditorOpen,
  true,
  'without a run id the source editor stays open for editing',
);
assert.equal(sandbox.guidedIdea.result, null);

console.log('guided idea flow contract: ok');
