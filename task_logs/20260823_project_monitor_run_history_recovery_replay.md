# Project Monitor run-history recovery replay

Date: 2026-08-23  
Module: web / Project Monitor  
Branch: `codex/final-ci-web-reconcile-20260823`  
Code commit: `7105f95` (`Restore Project Monitor run history authority`)

## Scope

The uncommitted Project Monitor run-history repair was preserved as a recovery
bundle, independently checked against the original orphaned Web work, and then
replayed onto the reconciled Web baseline `22fe978`.

The replay keeps one owner path for the behavior:

- the server merges the default run root with the Copilot pipeline workspace;
- a failed pipeline writes a bounded, privacy-scanned terminal projection;
- the Monitor loads persisted history before deciding that a study has no run;
- the `screens-agent.js` cache key and its static assertions move together.

## Recovery identity

- recommended patch: `EASYICU-recovery-patches/20260823_project_monitor_run_history/project_monitor_run_history_on_045f461.patch`;
- patch SHA-256: `9ce67ee98386f6a78a1b19fb5313ffe8c4576ce4fab1132e973ee510b1aa8a41`;
- changed production/test paths: 8;
- `screens-agent.js` after replay: the preserved orphan content, SHA-256 prefix `78e905d0cd27`;
- `index.html` retains the reconciled Guided project-owner wiring while adding `?v=20260823-run-history-authority1`.

## Verification

- focused Python matrix: `219 passed, 5 warnings`;
- JavaScript contracts: `24/24 passed`;
- Ruff: passed for all changed Python paths;
- `node --check src/easyicu/webserver/static/js/screens-agent.js`: passed;
- `git diff --check`: passed.

Browser QA used an isolated `EASYICU_HOME`, an owner-created StudyContext, and a
persisted fail-closed pipeline projection at 1440x900:

- the first real-mode render showed `Checking run history`, not `0` or `not run yet`;
- after the request completed, the rail and Runs tab both showed `1`;
- Run History showed `run_persisted_history_qa`, `blocked`, and 4 whitelisted artifacts;
- document `scrollWidth == clientWidth == 1425` with a 1440px viewport;
- console: 0 errors and 0 warnings.

The historical `study_b5ac6c7533a82657` retained runtime was no longer present
at its recorded isolated home during this replay. Therefore this browser pass
is a real server/browser contract check over an isolated persisted fixture, not
a repeat claim about that historical instance. The recovery bytes themselves
were independently compared with the orphan before replay.

## Remaining gate

The previous exact `045f461` full suite was `14,698 passed / 74 skipped / 0 failed`
before this replay. Exact `7105f95` still needs one full suite only if this HEAD
is selected as the merge/freeze checkpoint. The dirty local `main@1e5cda1` was
not cleaned, reset, stashed, or pulled during replay.
