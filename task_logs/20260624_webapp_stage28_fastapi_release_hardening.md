# Stage28 FastAPI Post-Streamlit Release Hardening

Date: 2026-06-24

## Goal

Verify the FastAPI native Web UI as the only default Web UI after Streamlit decommission, without restoring `src/easyicu/webapp` or changing legacy Streamlit code.

## Changes

- Added native FastAPI static assets to package data in `pyproject.toml`.
- Added `recursive-include src/easyicu/webserver/static *.html *.css *.js` to `MANIFEST.in`.
- Added repository/package contract coverage for required native assets:
  - `src/easyicu/webserver/static/index.html`
  - `src/easyicu/webserver/static/js/app.js`
  - `src/easyicu/webserver/static/css/app.css`
- Extended release archive contract checks so both sdist and wheel must include those assets.
- Adjusted the archive test skip gate from `build` to `build.__main__`; the active Anaconda env has an importable non-executable `build` package, so the test should skip unless `python -m build` is actually runnable.
- Made foreground `easyicu-webapp run` return shell status `130` on Ctrl-C instead of printing a traceback.

## Packaging Evidence

Baseline before the packaging fix:

- Built wheel/sdist in `/tmp/easyicu_stage28_release_venv`.
- Initial wheel/sdist had `static_count=0` for `easyicu.webserver.static`.
- No legacy `easyicu/webapp` package was present.

After the packaging fix:

- Build output: `/tmp/easyicu_stage28_build/`.
- Wheel: `/tmp/easyicu_stage28_build/easyicu-1.0.0-py3-none-any.whl`.
- Wheel static assets: `static_count=28`.
- Wheel required assets present: `index.html=true`, `app.js=true`, `app.css=true`.
- Wheel legacy checks: `legacy_count=0`, `has_legacy_entry=false`.
- Sdist static assets: `static_count=30`.
- Sdist required assets present: `index.html=true`, `app.js=true`, `app.css=true`.
- Sdist legacy checks: `legacy_count=0`.

## Clean Install Evidence

- Clean install venv: `/tmp/easyicu_stage28_clean_install`.
- Installed local wheel with the `webapp` extra from `/tmp/easyicu_stage28_build/easyicu-1.0.0-py3-none-any.whl`.
- `/tmp/easyicu_stage28_clean_install/bin/easyicu-webapp --help` displayed the native FastAPI CLI.
- Installed package resource checks:
  - `static_exists=true`
  - `index_exists=true`
  - `app_js_exists=true`
  - `app_css_exists=true`
  - `legacy_webapp_spec=None`
- Installed server started on `127.0.0.1:8778` with provider env file disabled.
- `GET /api/health` returned `200 {"status":"ok"}`.
- `GET /api/catalog` returned `200` with the expected native catalog keys.

## Browser QA

Command:

```bash
python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8778/ --out-dir output/playwright
```

Report:

- `output/playwright/native_fastapi_route_qa_20260624_200722/route_qa.json`

Result:

- Passed.
- Covered desktop and mobile route QA for `entry`, `extraction`, `patient`, `cohort`, `crossdb`, `agent`, `settings`, `dictionary`, `states`, `help`, and `guided`.
- JavaScript console errors: `0`.
- Horizontal overflow: `overflowX=0` for all checked routes/viewports.
- Unknown hash fallback rewrote to `#entry` and rendered `Welcome to EasyICU`.
- The known non-blocking offscreen/clipped counters remain QA-recorded for some complex pages, but the release route criteria passed and no old Streamlit path was restored.

## Validation

| Check | Result |
| --- | --- |
| `python -m compileall -q src/easyicu` | passed |
| `find src/easyicu/webserver/static/js -name '*.js' -print0 \| xargs -0 -n1 node --check` | passed |
| `ruff check src tests` | passed |
| Focused pytest: `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py tests/test_repository_contract.py tests/test_release_hardening_p0.py tests/test_release_archive_contract.py` | `87 passed, 1 skipped, 1 warning` |
| Full default pytest: `pytest -q` | `2248 passed, 41 skipped, 950 warnings in 327.90s` |
| `git diff --check` | passed |
| Provider dormant smoke with `EASYICU_DISABLE_PROVIDER_ENV_FILE=1` | `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false` |

## Concurrency Notes

- A concurrent Claude/research-agent/provider process and the old `127.0.0.1:8765` uvicorn were detected during preflight and were not touched.
- Stage28 used a separate installed-server port, `127.0.0.1:8778`.
- No provider secrets were read or printed.
- No `src/easyicu/webapp` package was restored.

## Verdict

Stage28 is done. The default FastAPI Web UI can be built, installed, launched, and browser-QAed after Streamlit package deletion. Release archives now carry the native FastAPI static assets and exclude the deleted legacy WebApp package.
