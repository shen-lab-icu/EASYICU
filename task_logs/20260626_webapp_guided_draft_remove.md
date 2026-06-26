# 2026-06-26 Guided Draft Remove Control

## Scope

The Guided Copilot rail showed local metadata-only draft records but had no way to remove one from the list. The user selected a stale draft row and asked how to delete a conversation/project.

## Decision

Add a safe remove action for Guided drafts only. The action unregisters the draft from the local Guided registry and deliberately does not delete the project folder on disk. Agent run folders remain non-deletable from this rail because they contain analysis artifacts and should be managed from Agent Projects or the filesystem with explicit user intent.

## Implementation

- Added `POST /api/guided/drafts/remove`.
- Added `guided_sessions.remove_guided_draft`.
- Added `EU_API.removeGuidedDraft`.
- Added a small remove button next to each local Guided draft row in `screens-guided.js`.
- Added route-owned styles in `guided.css`.
- Bumped Guided static cache tags to `20260626-guided-draft-remove`.
- Added backend regression coverage:
  - dangerous `delete_project_folder=true` is blocked,
  - normal remove unregisters the draft,
  - `disk_deleted=false`,
  - the project folder artifact remains present.
- Added static route assertions for the API, UI button, and owner CSS.

## Validation

- `python -m py_compile src/easyicu/webserver/app.py src/easyicu/webserver/guided_sessions.py`
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `pytest -q tests/test_webserver_workspace_summary.py -k 'guided_draft'` -> 2 passed
- `pytest -q tests/test_webserver_static_routes.py -k guided` -> 4 passed

## Browser QA

New server: `http://127.0.0.1:8786/?_v=guided-draft-remove#guided`

Flow:

1. Created a temporary QA draft through `/api/guided/drafts`.
2. Opened Guided Copilot.
3. Verified the draft row had a separate `Remove from Guided draft list` control.
4. Accepted the confirmation dialog.
5. Verified the draft disappeared from left-rail draft buttons and from `/api/guided/drafts/list`.
6. Verified the project file was preserved immediately after API removal.

Browser assertion result:

```json
{
  "dialog": "从研究引导草稿列表移除“QA delete draft 20260626”？本地项目文件夹不会被删除。",
  "removedMessage": true,
  "overflowX": 0
}
```

Backend registry after removal did not contain the QA draft id/title, and the matching local `guided_draft.json` remained on disk at validation time.

The temporary QA folder was removed manually after validation so it would not remain in the user's project list.
