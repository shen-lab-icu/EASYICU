/* Guided project continuity owner.
   Persists only one opaque local project id so a browser refresh can reopen
   the user's current project through the normal server-owned project list. */
(function () {
  'use strict';

  const STORAGE_KEY = 'easyicu_guided_active_project:v1';
  const PROJECT_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$/;

  function clean(value) {
    const id = String(value || '').trim();
    return PROJECT_ID.test(id) ? id : '';
  }

  function remember(projectId) {
    const id = clean(projectId);
    if (!id) return false;
    try {
      localStorage.setItem(STORAGE_KEY, id);
      return true;
    } catch (error) {
      return false;
    }
  }

  function remembered() {
    try {
      return clean(localStorage.getItem(STORAGE_KEY));
    } catch (error) {
      return '';
    }
  }

  function forget(projectId) {
    const expected = clean(projectId);
    try {
      const current = remembered();
      if (!expected || current === expected) localStorage.removeItem(STORAGE_KEY);
    } catch (error) {}
  }

  window.EU_GUIDED_PROJECT_CONTINUITY = { remember, remembered, forget };
})();
