/* Executable contract for Patient Review single-feature lazy loading. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, reject, resolve };
}

const pending = [];
const calls = [];
global.EU_API = {
  loadPatientReviewFeature(body) {
    calls.push(body);
    const request = deferred();
    pending.push(request);
    return request.promise;
  },
};

require(path.resolve(process.argv[2]));
const owner = global.EU_PATIENT_REVIEW.features;
assert(owner);

const drill = {
  demo: false,
  source: { path_hash: 'source-a' },
  selected: { ref: 'entity-a', ordinal: 1 },
};
let repaints = 0;
const config = {
  drill: () => drill,
  sourcePath: () => '/registered/export',
  repaint: () => { repaints += 1; },
};
const flush = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

(async () => {
  owner.prime(drill);
  owner.load(config, 'hr');
  assert.deepEqual(calls[0], {
    source_path: '/registered/export',
    entity_ref: 'entity-a',
    entity_ordinal: 1,
    feature: 'hr',
  });
  assert.equal(owner.stateFor('hr', drill).loading, true);

  pending[0].resolve({
    status: 'numeric_trajectory',
    feature: { feature: 'hr', module: 'vitals' },
    signal: {
      feature: 'hr',
      values: [82, 88, 86],
      times: [0, 1, 4],
      point_count: 3,
    },
  });
  await flush();
  assert.equal(owner.stateFor('hr', drill).loaded, true);
  const augmented = owner.augmentLanes([], drill);
  assert.equal(augmented[0].lane, 'vitals');
  assert.deepEqual(augmented[0].signals[0].values, [82, 88, 86]);
  assert.equal(augmented[0].signals[0].lazy_loaded, true);

  drill.selected = { ref: 'entity-b', ordinal: 2 };
  assert.equal(owner.stateFor('hr', drill).loaded, false);
  owner.load(config, 'lact');
  assert.equal(calls[1].entity_ref, 'entity-b');

  const sourceB = {
    demo: false,
    source: { path_hash: 'source-b' },
    selected: { ref: 'entity-c', ordinal: 1 },
  };
  owner.prime(sourceB);
  pending[1].resolve({
    status: 'numeric_trajectory',
    feature: { feature: 'lact', module: 'chemistry' },
    signal: { feature: 'lact', values: [1.2, 1.4], times: [0, 3] },
  });
  await flush();
  assert.deepEqual(owner.augmentLanes([], sourceB), []);
  assert(repaints >= 2);

  drill.selected = { ref: 'entity-d', ordinal: 4 };
  const batch = owner.loadMany(config, ['hr', 'map'], 'vitals');
  assert.deepEqual(calls[2], {
    source_path: '/registered/export',
    entity_ref: 'entity-d',
    entity_ordinal: 4,
    feature: 'hr',
  });
  assert.deepEqual(calls[3], {
    source_path: '/registered/export',
    entity_ref: 'entity-d',
    entity_ordinal: 4,
    feature: 'map',
  });
  pending[2].resolve({
    status: 'numeric_trajectory',
    feature: { feature: 'hr', module: 'vitals' },
    signal: { feature: 'hr', values: [80, 91], times: [0, 4] },
  });
  pending[3].resolve({
    status: 'observed_numeric_static',
    feature: { feature: 'map', module: 'vitals' },
    observation: { current: 72, observation_count: 1 },
  });
  await batch;
  assert.equal(owner.stateFor('hr', drill).loaded, true);
  assert.equal(owner.stateFor('map', drill).loaded, true);

  process.stdout.write(JSON.stringify({
    bounded_module_batch_supported: true,
    cache_scoped_to_entity: true,
    projected_request: true,
    stale_response_rejected: true,
  }));
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
