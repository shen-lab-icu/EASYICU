# Pi project / conversation ownership boundary

- Date: 2026-08-08
- Branch: `feat/pi-copilot-shell`
- Task: remove the duplicate “project folder = conversation memory” model while keeping EasyICU research projects and Pi AgentSession history.

## Decision

EasyICU research projects own study configuration, runs, artifacts, and evidence. Pi AgentSession owns conversation history and transcript persistence. A project stores only the stable identifier that scopes which Pi sessions may be listed or reopened; EasyICU does not copy Pi transcript bodies into the legacy Guided project conversation store.

Legacy `guided_copilot_session.json` messages remain readable only for the explicitly selected local Guided fallback. They are not migrated into Pi and are not restored when the Pi shell is active.

## Implemented contracts

- Every new Pi session requires a bounded `project_id`.
- The project field is frozen after record creation.
- Pi session listing is filtered by exact project ID; pre-migration unassigned records remain readable internally but appear in no project list.
- Opening a Pi session through the Web API requires the current project ID and returns stable `pi_session_project_mismatch` on cross-project access.
- Browser remember/resume pointers are stored per project instead of under one global Pi session key.
- The Guided project picker passes only `{id, title}` to the Pi owner. It does not pass project paths or restore legacy message bodies in Pi mode.
- The left rail is now “Research projects / 研究项目” and explicitly states the storage boundary. Project dialogs no longer describe folders as Pi conversation memory.

## Verification

- `144 passed`: complete Pi owner/gateway/install/provider/routes/static contract set, Web static route suite, and route ownership snapshots.
- `3 passed`: typed Copilot study-intent route compatibility.
- `1 passed` plus direct Node receipt: Guided evidence gate remains fail-closed.
- Ruff passed on all changed Python owners and tests.
- Node syntax checks passed for `api.js`, `screens-guided-projects.js`, `screens-guided-pi.js`, and `screens-guided.js`.
- `git diff --check` passed.
- Desktop browser QA at 1440×900: project-only rail copy, project selection, setup gate, and clipping/overflow checked; body/app/Pi widths were `1440/1440`, `1440/1440`, and `886/886`; console had 0 errors and 0 warnings.
- Screenshot: `task_logs/screenshots/20260808_pi_project_memory_boundary.png`.

## Remaining external evidence

The real provider prompt/tool canary still requires the user to enter a current credential through `http://127.0.0.1:8765/#guided`. No credential was read from conversation text, browser storage, project files, logs, or test fixtures during this task.
