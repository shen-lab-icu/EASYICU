/* Executable contract for bounded Patient entity and table browse owners. */
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

const entityPages = [];
const entityDetails = [];
const tablePages = [];
const calls = { entities: [], entity: [], table: [] };

global.EU_API = {
  loadPatientReviewEntities(body) {
    calls.entities.push(body);
    const pending = deferred();
    entityPages.push(pending);
    return pending.promise;
  },
  loadPatientReviewEntity(body) {
    calls.entity.push(body);
    const pending = deferred();
    entityDetails.push(pending);
    return pending.promise;
  },
  loadPatientReviewTablePreview(body) {
    calls.table.push(body);
    const pending = deferred();
    tablePages.push(pending);
    return pending.promise;
  },
};

require(path.resolve(process.argv[2]));
require(path.resolve(process.argv[3]));

const navigation = global.EU_PATIENT_REVIEW.navigation;
const tables = global.EU_PATIENT_REVIEW.tables;
const drill = {
  demo: false,
  source: { path_hash: 'source-a' },
  summary: { entities: 30 },
  selected: { ref: 'ent_1', ordinal: 1, label: 'Entity 1' },
  entities: [{ ref: 'ent_1', ordinal: 1, label: 'Entity 1' }],
  entity_navigation: {
    page: 1,
    page_size: 12,
    page_count: 3,
    total_entities: 30,
    row_start: 1,
    row_end: 12,
    has_previous: false,
    has_next: true,
    options: [{ ref: 'ent_1', ordinal: 1, label: 'Entity 1' }],
  },
  data_tables: {
    modules: [{ module: 'demographics' }, { module: 'labs' }],
    module_picker: { default_module: 'demographics' },
    table_previews: [{
      module: 'demographics',
      page: 1,
      page_size: 24,
      rows: [{ entity: 'ent_1', age: 61 }],
    }],
  },
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
  navigation.prime(drill);
  tables.prime(drill);

  tables.load(config, { module: 'labs', page: 2, pageSize: 24 });
  assert.deepEqual(calls.table[0], {
    source_path: '/registered/export',
    table_module: 'labs',
    table_page: 2,
    table_page_size: 24,
  });
  assert.equal(tables.snapshot(drill).loading, true);
  assert.equal(tables.activePreview(drill).module, 'demographics');
  tablePages[0].reject(new Error('bounded table read failed'));
  await flush();
  assert.equal(tables.snapshot(drill).loading, false);
  assert.match(tables.snapshot(drill).error, /bounded table read failed/);
  assert.equal(tables.activePreview(drill).module, 'demographics');
  assert.equal(drill.data_tables.table_previews[0].rows[0].age, 61);

  tables.load(config, { module: 'labs', page: 2, pageSize: 24 });
  tables.load(config, { module: 'labs', page: 3, pageSize: 24 });
  tablePages[2].resolve({ module_preview: {
    module: 'labs', page: 3, page_size: 24, rows: [{ entity: 'ent_25', lact: 2.5 }],
  } });
  await flush();
  tablePages[1].resolve({ module_preview: {
    module: 'labs', page: 2, page_size: 24, rows: [{ entity: 'ent_13', lact: 1.3 }],
  } });
  await flush();
  assert.equal(tables.snapshot(drill).page, 3);
  assert.equal(tables.activePreview(drill).rows[0].entity, 'ent_25');
  assert.equal(
    tables.moduleStatus(
      { module: 'labs', review_status: 'inventory_only' },
      tables.activePreview(drill),
      { t: en => en },
    ),
    'loaded',
  );
  assert.equal(
    tables.moduleStatus(
      { module: 'chemistry', review_status: 'inventory_only' },
      tables.activePreview(drill),
      { t: en => en },
    ),
    'available · load',
  );

  tables.load(config, { module: 'demographics', page: 1, pageSize: 24 });
  assert.equal(tables.activePreview(drill).module, 'demographics');
  tables.load(config, { module: 'labs', page: 3, pageSize: 24 });
  assert.equal(tables.activePreview(drill).rows[0].entity, 'ent_25');
  assert.equal(calls.table.length, 3);

  tables.load(config, { module: 'labs', page: 4, pageSize: 24 });
  const sourceB = {
    demo: false,
    source: { path_hash: 'source-b' },
    data_tables: {
      modules: [{ module: 'demographics' }],
      module_picker: { default_module: 'demographics' },
      table_previews: [{
        module: 'demographics', page: 1, page_size: 24, rows: [{ entity: 'source_b' }],
      }],
    },
  };
  tables.prime(sourceB);
  tablePages[3].resolve({ module_preview: {
    module: 'labs', page: 4, page_size: 24, rows: [{ entity: 'stale_source_a' }],
  } });
  await flush();
  assert.equal(tables.activePreview(sourceB).rows[0].entity, 'source_b');

  navigation.loadPage(config, { page: 2 });
  navigation.loadPage(config, { page: 3 });
  entityPages[1].resolve({ navigation: {
    page: 3,
    page_size: 12,
    page_count: 3,
    total_entities: 30,
    row_start: 25,
    row_end: 30,
    has_previous: true,
    has_next: false,
    options: [{ ref: 'ent_25', ordinal: 25, label: 'Entity 25' }],
  } });
  await flush();
  entityPages[0].resolve({ navigation: {
    page: 2,
    page_size: 12,
    page_count: 3,
    total_entities: 30,
    row_start: 13,
    row_end: 24,
    has_previous: true,
    has_next: true,
    options: [{ ref: 'ent_13', ordinal: 13, label: 'Entity 13' }],
  } });
  await flush();
  const navHtml = navigation.render({ drill, selected: drill.selected, helpers: {} });
  assert.match(navHtml, /25-30/);
  assert.match(navHtml, /Entity 25/);
  assert.doesNotMatch(navHtml, /Entity 13/);

  navigation.loadEntity(config, 'ent_13', 13);
  navigation.loadEntity(config, 'ent_25', 25);
  entityDetails[1].resolve({
    selected: { ref: 'ent_25', ordinal: 25, label: 'Entity 25' },
    entities: [{ ref: 'ent_25', ordinal: 25, label: 'Entity 25' }],
    time_lanes: [{ lane: 'vitals', status: 'ready' }],
    trajectory_review: { selected_ref: 'ent_25' },
    patient_overview: { selected_ref: 'ent_25' },
  });
  await flush();
  entityDetails[0].resolve({
    selected: { ref: 'ent_13', ordinal: 13, label: 'Entity 13' },
    time_lanes: [{ lane: 'vitals', status: 'stale' }],
  });
  await flush();
  assert.equal(drill.selected.ref, 'ent_25');
  assert.equal(drill.time_lanes[0].status, 'ready');

  const selectedBeforeFailure = drill.selected;
  const lanesBeforeFailure = drill.time_lanes;
  navigation.loadEntity(config, 'ent_27', 27);
  entityDetails[2].reject(new Error('bounded entity read failed'));
  await flush();
  assert.equal(drill.selected, selectedBeforeFailure);
  assert.equal(drill.time_lanes, lanesBeforeFailure);
  assert.match(
    navigation.render({ drill, selected: drill.selected, helpers: {} }),
    /bounded entity read failed/,
  );

  navigation.loadEntity(config, 'ent_26', 26);
  drill.source.path_hash = 'source-b';
  navigation.prime(drill);
  entityDetails[3].resolve({
    selected: { ref: 'ent_26', ordinal: 26, label: 'Entity 26' },
  });
  await flush();
  assert.equal(drill.selected.ref, 'ent_25');
  assert.ok(repaints >= 8);

  process.stdout.write(JSON.stringify({ ok: true, table_calls: calls.table.length }));
})().catch(error => {
  process.stderr.write(String(error && error.stack || error));
  process.exitCode = 1;
});
