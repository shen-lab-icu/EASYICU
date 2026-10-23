/* Owner: explicit module boundary for the Guided Copilot script family.

   Classic script tags still provide deterministic load order, but feature
   modules declare and resolve named APIs here instead of publishing dozens of
   unrelated Guided Copilot globals. Duplicate and missing ownership is
   rejected immediately so script-order defects cannot silently select a
   different implementation. */
(function () {
  'use strict';

  const root = window;
  const app = root.EasyICU || (root.EasyICU = {});
  if (app.guidedPi) throw new Error('Guided Copilot module namespace already exists');

  const declared = Object.create(null);
  function checkedName(name) {
    const key = String(name || '').trim();
    if (!/^[a-z][A-Za-z0-9]*$/.test(key)) {
      throw new Error('Guided Copilot module name must be a non-empty identifier');
    }
    return key;
  }
  function declare(name, api) {
    const key = checkedName(name);
    if (Object.prototype.hasOwnProperty.call(declared, key)) {
      throw new Error(`Guided Copilot module already declared: ${key}`);
    }
    if (!api || typeof api !== 'object') {
      throw new Error(`Guided Copilot module API must be an object: ${key}`);
    }
    declared[key] = Object.freeze(api);
    return declared[key];
  }
  function requireModule(name) {
    const key = checkedName(name);
    if (!Object.prototype.hasOwnProperty.call(declared, key)) {
      throw new Error(`Guided Copilot module is not declared: ${key}`);
    }
    return declared[key];
  }
  function optional(name) {
    const key = checkedName(name);
    return Object.prototype.hasOwnProperty.call(declared, key) ? declared[key] : null;
  }

  // Large job receipts encode repeated column names and constant facts once.
  // Consumers always see the same event objects as legacy replay/live rows.
  function jobProgress(job) {
    const table = job && job.progress_table;
    if (!table || table.schema_version !== 'easyicu.pi-job-progress/1') {
      return Array.isArray(job && job.progress) ? job.progress : [];
    }
    const columns = Array.isArray(table.columns) ? table.columns : [];
    const dictionaries = table.dictionaries || {};
    return (table.batches || []).flatMap(batch => batch.map(cells => {
      const row = { ...table.defaults, ...Object.fromEntries(columns.flatMap((key, index) =>
        cells[index] == null ? [] : [[key, dictionaries[key] ? dictionaries[key][cells[index]] : cells[index]]])) };
      if (!row.timestamp) row.timestamp = row.ended_at || row.started_at;
      return row;
    }));
  }

  app.guidedPi = Object.freeze({ declare, require: requireModule, optional, jobProgress });
})();
