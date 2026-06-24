# Stage29 FastAPI Migration Final Release Boundary

Date: 2026-06-24

## Goal

Create the final FastAPI migration release boundary and PR-readiness package
after Streamlit removal. This stage does not restore or edit the deleted
Streamlit package and does not add new Web UI functionality.

## Artifacts

- Release boundary and checklist:
  `docs/fastapi_migration_release_boundary_stage29.md`
- Stage29 task log:
  `task_logs/20260624_webapp_stage29_fastapi_release_boundary.md`

## Boundary Summary

- Current branch: `ux/easyicu-web-copilot-agent-projects`
- Repository PR base: `origin/main` at `d9d30d9`
- Stage28 release-hardening commit before this log: `db898b6`
- The current branch contains 41 commits ahead of `origin/main`.
- WebApp-only PRs should include the FastAPI native, CSS decommission,
  Streamlit package decommission, and release packaging commits listed in the
  Stage29 boundary document.
- Three interleaved research-agent commits should be split out or explicitly
  included in a broader PR:
  - `e179a4c`
  - `11e2bb9`
  - `ffff381`

## Stage29 Verification

Clean build:

- Build venv: `/tmp/easyicu_stage29_build_venv`
- Dist: `/tmp/easyicu_stage29_dist`
- Artifacts: `easyicu-1.0.0-py3-none-any.whl`, `easyicu-1.0.0.tar.gz`

Archive checks:

- Wheel native static count: `28`
- Wheel required native assets present: `true`
- Wheel legacy `easyicu/webapp` count: `0`
- Wheel legacy entrypoint/reference: `false`
- Sdist native static count: `30`
- Sdist required native assets present: `true`
- Sdist legacy `src/easyicu/webapp` count: `0`

Clean install:

- Install venv: `/tmp/easyicu_stage29_install_venv`
- `easyicu-webapp --help`: native FastAPI CLI displayed
- Installed resources: `index.html`, `app.js`, and `app.css` present
- `easyicu.webapp` import spec: `None`

Installed server:

- Port: `127.0.0.1:8779`
- `/api/health`: HTTP 200
- `/api/catalog`: HTTP 200
- Provider env file disabled for the smoke.

Browser QA:

- Report:
  `output/playwright/native_fastapi_route_qa_20260624_202631/route_qa.json`
- Result: passed
- Console errors: `0`
- Horizontal overflow: `overflowX=0` on desktop and 393x852 mobile routes
- Unknown hash fallback: `#entry`

Tests and scans:

- Focused release tests:
  `pytest -q tests/test_release_hardening_p0.py tests/test_release_archive_contract.py tests/test_repository_contract.py tests/test_webserver_static_routes.py`
  -> `30 passed, 1 skipped`
- Post-comment cleanup focused tests:
  `pytest -q tests/test_release_hardening_p0.py tests/test_repository_contract.py tests/test_webserver_static_routes.py`
  -> `30 passed`
- Touched JS syntax check: passed
- Active legacy import/entrypoint scan: no `easyicu.webapp`,
  `easyicu-webapp-legacy`, `webapp-legacy`, or `--run-legacy-streamlit`
  outside archival `_internal` path-only notes.
- Provider dormant smoke:
  `ai_enabled=false`, `ready=false`, `client_constructed=false`,
  `network_calls=0`, `secrets_returned=false`

## Notes

- The old `127.0.0.1:8765` process and concurrent Claude/research-agent/provider
  process were detected during preflight and were not touched.
- Stage29 used temporary build/install directories under `/tmp` and the
  separate installed-server port `8779`.
- No provider secrets were read or printed.
- No Streamlit package or legacy CSS was restored.

## Verdict

Stage29 is done once this document, the task log, and the two native JS comment
cleanups are committed. The WebApp migration line can stop here; the next work
should return to the manuscript, benchmark, or discovery priorities unless a
release manager asks for CI rerun or version-tag work.
