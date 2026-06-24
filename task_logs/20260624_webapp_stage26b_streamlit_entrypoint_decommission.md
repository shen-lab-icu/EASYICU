# Stage26B Streamlit entrypoint/docs decommission

Date: 2026-06-24

## Scope

FastAPI native is the maintained WebApp path. This stage removes Streamlit from
the default package entrypoint and repository launcher path without deleting
`src/easyicu/webapp`.

## Changes

- Added native FastAPI CLI entrypoint:
  - `src/easyicu/webserver/__main__.py`
  - default host/port: `127.0.0.1:8765`
  - supports `run`, `stop`, `status`, and `--background`.
- Updated package metadata:
  - `easyicu-webapp` now points to `easyicu.webserver.__main__:main`.
  - `easyicu-webapp-legacy` points to `easyicu.webapp.__main__:main`.
  - `webapp` extra now installs native FastAPI dependencies.
  - `webapp-legacy` carries the deprecated Streamlit dependency set.
- Updated one-click launcher:
  - `scripts/launch_easyicu.py` now installs/runs the native FastAPI WebApp.
  - default port changed from `8501` to `8765`.
  - health check changed from Streamlit `_stcore/health` to `/api/catalog`.
- Updated public docs:
  - `README.md` and `README_zh.md` point Path A to native FastAPI.
  - `docs/native_fastapi_webserver.md` documents `easyicu-webapp` as the default.
  - `src/easyicu/README.md`, `src/easyicu/webapp/README.md`, and
    `src/easyicu/webapp/LEGACY.md` mark Streamlit as legacy/deprecated.
  - `docs/streamlit_decommission_audit_stage25.md` records Stage26A/26B resolved
    blockers and remaining delete blockers.

## Verification

Commands run from `EASYICU/`:

```bash
python -m py_compile src/easyicu/webserver/__main__.py scripts/launch_easyicu.py
python -m compileall -q src/easyicu/webserver
python -m compileall -q src/easyicu
pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py
python -m easyicu.webserver --help
git diff --check
```

Results:

- `py_compile`: passed.
- `compileall src/easyicu/webserver`: passed.
- `compileall src/easyicu`: passed.
- FastAPI focused pytest: `60 passed, 1 warning`.
- Native CLI help: parsed and exited without starting a server.
- `git diff --check`: passed.

Provider dormant smoke:

```text
ai_enabled=False
ready=False
client_constructed=False
network_calls=0
secrets_returned=False
```

## Reference scan

Representative scan:

```bash
rg -n "easyicu-webapp|streamlit|easyicu\.webapp" pyproject.toml README* docs scripts src tests
```

Default entrypoint/docs now point to FastAPI native. Remaining Streamlit
mentions are expected in:

- `pyproject.toml` legacy extra/script: `webapp-legacy`, `easyicu-webapp-legacy`.
- `src/easyicu/webapp/**`: legacy package self-imports and Streamlit runtime.
- Legacy tests under `tests/` and `tests/webapp/`.
- Archive/internal docs and legacy CSS inventory/guard tooling.
- `scripts/full_export_modules.py`, which still imports the legacy concept
  catalog shim and remains a Stage26C/Stage27 blocker unless migrated or
  archived.

## Remaining blockers

Do not delete `src/easyicu/webapp` yet.

- Stage26C must split/skip/archive Streamlit UI tests from the active gate.
- `scripts/full_export_modules.py` still imports `easyicu.webapp.concept_catalog`.
- The legacy package still has internal self-imports and Streamlit runtime code.
- Stage27 package deletion should only happen after the above references are
  removed or explicitly archived.
