# 2026-06-26 Guided rail removes seeded examples

## Scope

Remove the hard-coded `Seeded examples` section from the Guided Copilot left rail.
The Guided rail now defaults to real local context only:

- local metadata-only Guided drafts
- local Agent runs discovered from project folders

Seeded/tutorial examples should live in Get Started or Demo-specific surfaces, not
beside project memory.

## Changed files

- `src/easyicu/webserver/static/js/screens-guided.js`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_static_routes.py`

## Evidence

- Removed the `Seeded examples` hard-coded list and its `data-sess` click handler.
- Updated the static contract test so the Guided rail must not contain
  `Seeded example · not a local project`, `Seeded examples`, `data-sess`, or
  the old seeded-example bot response.
- Bumped Guided static cache tag to `20260626-guided-real-rail`.

## Validation

- `node --check src/easyicu/webserver/static/js/screens-guided.js`
- `python -m compileall -q src/easyicu/webserver`
- `pytest -q tests/test_webserver_static_routes.py`
- `git diff --check`
- Browser QA on a temporary FastAPI server confirmed:
  - no `Seeded examples` text in `#gdSessions`
  - zero `#gdSessions [data-sess]` buttons
  - local draft/run headings remain visible
  - `overflowX=0`

## Boundary

This does not remove Demo mode or tutorial examples globally. It only prevents
seeded examples from appearing in the Guided project-memory rail where users
expect real local folders and real run history.
