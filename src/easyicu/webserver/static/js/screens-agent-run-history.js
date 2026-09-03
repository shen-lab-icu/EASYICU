/* Project Monitor persisted-run projection owner.
   Keeps history shaping independent from the screen's mutable UI state. */
(function () {
  function rows(history, study) {
    if (!study || !history || history.studyId !== study.id || !history.data || !Array.isArray(history.data.runs)) return [];
    return history.data.runs;
  }

  function count(history, study, realMode) {
    if (!realMode) return Array.isArray(study && study.runs) ? study.runs.length : 0;
    if (!study || !history || history.studyId !== study.id || history.loading || history.error || !history.data) return null;
    return Number.isInteger(history.data.count) ? history.data.count : rows(history, study).length;
  }

  function run(history, study) {
    const row = rows(history, study)[0];
    if (!row || !row.project_dir) return null;
    return {
      run_id: row.run_id,
      run_label: row.run_label,
      study_id: row.study_id || study.id,
      mode: row.mode || study.mode,
      run_type: row.run_type || 'preflight',
      project_dir: row.project_dir,
      source: {},
      summary: {},
      gate: {
        status: row.gate_status || 'blocked',
        reportable: false,
        draft_unlocked: false,
        checks: [],
      },
      artifacts: [],
      persistedHistory: true,
    };
  }

  function projectFolder(history, study, displayPath, labels) {
    if (study && study.empty) return labels.empty;
    const persisted = rows(history, study)[0];
    if (persisted && persisted.project_dir) {
      return displayPath(String(persisted.project_dir).replace(/[\\/]run_[^\\/]+[\\/]?$/, ''));
    }
    return study && study.ideaSeed && study.ideaSeed.project_dir
      ? displayPath(study.ideaSeed.project_dir)
      : labels.pending;
  }

  window.EU_AGENT_RUN_HISTORY_VIEW = { rows, count, run, projectFolder };
})();
