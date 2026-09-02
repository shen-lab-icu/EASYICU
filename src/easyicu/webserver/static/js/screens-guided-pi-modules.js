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

  app.guidedPi = Object.freeze({ declare, require: requireModule, optional });
})();
