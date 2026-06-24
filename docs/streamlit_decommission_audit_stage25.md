# Streamlit WebApp decommission audit

Date: 2026-06-24

## Verdict

The legacy Streamlit WebApp has been removed from the active package boundary.
The maintained Web UI is the native FastAPI app launched by `easyicu-webapp`.

Recovery path for archive forensics:

- Use git history before Stage27.
- Or apply the local dirty-worktree recovery patch written before deletion:
  `output/stage27_streamlit_archive_patch/legacy_streamlit_dirty.patch`.

Do not treat the old Streamlit app as a runtime fallback.

## Stage evidence

- Stage24B removed the inactive route split CSS files.
- Stage26A moved shared helper logic to non-Streamlit modules:
  `easyicu.ai_optin`, `easyicu.concept_catalog`, and `easyicu.data_paths`.
- Stage26B moved the default `easyicu-webapp` launcher to the native FastAPI
  server.
- Stage26C removed legacy Streamlit tests from the default pytest path and
  migrated the local full-export utility to the shared concept catalog.
- Stage27 archived the dirty legacy worktree diff and removed:
  - `src/easyicu/webapp/`
  - legacy Streamlit UI tests
  - the legacy console entrypoint
  - the legacy optional dependency extra

## Active package boundary

The active web-facing package boundary is now:

- `src/easyicu/webserver/`
- `src/easyicu/webserver/static/`
- shared core modules under `src/easyicu/`
- FastAPI/core tests under `tests/`

Legacy Streamlit browser/visual parity is no longer a release gate.

## Stage27 validation commands

```bash
rg "easyicu\\.webapp" src tests scripts pyproject.toml README.md README_zh.md docs
python -m compileall -q src/easyicu
pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py tests/test_repository_contract.py
git diff --check
```

Provider status must remain dormant:

```text
ai_enabled=false
ready=false
client_constructed=false
network_calls=0
secrets_returned=false
```

## Remaining cleanup

- Internal historical docs and archive tools may still mention Streamlit as
  provenance. They are not runtime dependencies.
- If a future task needs an old UI helper for comparison, recover it from git
  history into a scratch branch rather than reintroducing it into the active
  package.
