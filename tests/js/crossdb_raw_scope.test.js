/* Executable contract for explicit quick/full raw Cross-DB feature scopes. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
require(path.resolve(process.argv[2]));

const owner = global.EU_CROSSDB_RAW;
assert(owner, 'Cross-DB raw scope owner must be published');

const base = {
  dataRoot: '/raw/icu',
  databases: ['miiv', 'eicu', 'miiv'],
  maxPatients: 200,
  sampleSize: 600,
};
const quick = owner.buildRequest({ ...base, featureScope: 'core' });
assert.equal(quick.feature_scope, 'curated_core');
assert.equal(quick.features.length, 12);
assert.deepEqual(quick.databases, ['miiv', 'eicu']);

const full = owner.buildRequest({ ...base, featureScope: 'all' });
assert.equal(full.feature_scope, 'all_catalog');
assert.equal(Object.hasOwn(full, 'features'), false, 'full catalog must be resolved by the backend catalog owner');
assert.equal(owner.apiFeatureScope('all'), 'all_catalog');
assert.equal(owner.apiFeatureScope('unknown'), 'curated_core');

process.stdout.write(JSON.stringify({
  explicit_full_catalog: true,
  quick_core_preserved: true,
  backend_catalog_owned: true,
}));
