# Native FastAPI Webserver

This is the local-first native frontend/server path for the EasyICU WebApp
migration. It is separate from the legacy Streamlit launcher and does not
change `easyicu-webapp`.

## Start Locally

From `EASYICU/`:

```bash
source .venv/bin/activate
python -m uvicorn easyicu.webserver.app:app --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

If port `8765` is already in use, pick another local port:

```bash
python -m uvicorn easyicu.webserver.app:app --host 127.0.0.1 --port 8766
```

## Local Export Registry

The native UI uses a local registry shared by Patient Review, Cohort
Statistics, Cross-DB Benchmark, Agent Projects, and Copilot:

```text
~/.easyicu/webserver_sources.json
```

The registry stores local export paths and bounded metadata only. Removing a
source from the UI unregisters metadata; it must not delete export folders.
Valid EasyICU module exports can also be auto-discovered from the configured
export directory.

Relevant API checks:

```bash
curl http://127.0.0.1:8765/api/workspaces/registry
curl -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/easyicu/export"}' \
  http://127.0.0.1:8765/api/workspace/summary
```

Cross-DB preview is fail-closed until two or more valid exports are selected:

```bash
curl -i -H 'Content-Type: application/json' \
  -d '{"paths":[]}' \
  http://127.0.0.1:8765/api/workspaces/crossdb-summary
```

Expected result: HTTP 400 with `need_two_exports`.

## Provider Defaults

External model providers stay dormant by default.

- `~/.easyicu/webserver_settings.json` defaults to `ai_enabled=false`.
- `GET /api/agent-runs/provider-status` reports readiness only.
- Provider status may read env names and a private env file status, but must not
  construct a client, make a network call, or return secret values.
- `~/.easyicu/provider.env` is optional and must remain private user state.

For an extra conservative local launch that ignores the private provider env
file while testing UI routes:

```bash
EASYICU_DISABLE_PROVIDER_ENV_FILE=1 \
python -m uvicorn easyicu.webserver.app:app --host 127.0.0.1 --port 8765
```

Do not run a real external provider call unless the operator explicitly enables
global AI opt-in and grants per-run consent.

## Fallback-Only Readiness

As of 2026-06-24, the native FastAPI frontend is the active migration target for
low-risk WebApp replacement work. The legacy Streamlit UI and its CSS are frozen
as fallback/reference material; do not continue resolving native frontend issues
by editing Streamlit route files or Streamlit override CSS.

Current FastAPI fallback-only blockers addressed:

- `#help` opens the existing Get Started/Help screen through a route alias.
- Unknown hashes fail safe to `#entry` and do not leave the previous screen in
  place.
- Mobile route QA at `393x852` confirms no horizontal overflow or visible
  clipping for Settings, Workspace States, Guided Copilot, and Data Dictionary.
- Route QA covers desktop and mobile for `entry`, `extraction`, `patient`,
  `cohort`, `crossdb`, `agent`, `settings`, `dictionary`, `states`, `help`, and
  `guided`; it also asserts `#help` and unknown-hash behavior.
- Extraction advanced filters are now wired to a metadata-only FastAPI backend:
  `/api/extraction/filter-options` and `/api/extraction/filter-preview` use the
  active registered export source, return source provenance plus bounded
  module/schema/coverage aggregates, and fail closed for unsupported cohort-row
  filters.
- Patient Review real mode now has a minimum true drilldown path:
  `/api/patient-review/drilldown` uses the active registered export, returns
  aggregate summary plus one pseudonymous bounded entity drilldown, caps signal
  arrays for browser review, and fails closed when there is no active registered
  source. The payload intentionally does not return direct clinical identifiers,
  `tableRows`, or patient row tables.
- Cohort Review real mode now has a minimum true aggregate path:
  `/api/cohort-review/summary` uses the active registered export, returns
  cohort size, mortality, age/sex, SOFA-2, ICU LOS, Sepsis-3/event presence, and
  module coverage/quality aggregates, and fails closed for unsupported filters
  or inferential statistics. The native `#cohort` screen renders those real
  aggregates across Group contrast, Coverage audit, Cohort profile, and SOFA
  panels without returning patient rows or direct clinical identifiers.
- Cross-DB real mode now has a minimum true aggregate path:
  `/api/crossdb-review/summary` uses two or more registered export sources,
  returns descriptive cross-database cohort-level metrics, module availability,
  source provenance by path hash, and an explicit compatibility gate. It fails
  closed when fewer than two sources are selected, when a requested source is not
  registered, when shared core modules are missing, or when row-level filters,
  matched cohorts, p-values/SMDs, paired reclassification, or other unsupported
  statistics are requested. The native `#crossdb` screen renders the real
  registered-source aggregate without demo copy, patient rows, direct clinical
  identifiers, `tableRows`, or time-series payloads.
- The numeric evidence audit gate is now wired into full Agent runs before any
  future reportable/draft-unlock path. Numeric claims in `manuscript_draft.json`
  must cite evidence artifacts and match concrete artifact values within a
  recorded tolerance. Mismatched values, ghost evidence, and numeric sentences
  without evidence ids fail closed. This gate does not unlock `reportable=true`
  or draft release.
- Stage20 archive readiness audit passed on 2026-06-24. See
  `docs/native_fastapi_archive_readiness_stage20.md`.

Local QA command:

```bash
python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8765/
python tools/qa_native_fastapi_extraction_filters.py
python tools/qa_native_fastapi_patient_drilldown.py
python tools/qa_native_fastapi_cohort_parity.py
python tools/qa_native_fastapi_crossdb_parity.py
```

The scripts write screenshots and JSON reports to `output/playwright/`.

Stage18 Cross-DB browser QA evidence:

- `tools/qa_native_fastapi_crossdb_parity.py` registers two fixture exports,
  validates `/api/crossdb-review/summary`, opens `#crossdb` in real mode at
  `393x852`, and asserts non-empty content, no JavaScript console errors, no
  horizontal page overflow, no demo copy, no raw identifier markers, real source
  provenance, a real cohort aggregate table, an availability matrix, and
  fail-closed unsupported-analysis scope.

Cleanup boundary:

- Cohort row-level filters, p-values/SMDs/statistical inference, matched
  cohorts, Cross-DB matched analyses, and paired SOFA-1/SOFA-2
  reclassification remain explicitly blocked or fail-closed; they are not
  reportable native features.
- The old Streamlit app can now be treated as frozen fallback/reference for the
  audited paths, and a separate legacy CSS cleanup/archive task may start next.
- This document still does not authorize deleting the whole Streamlit stack or
  historical CSS in the same change. Cleanup must be separately staged, with a
  fallback snapshot and focused QA after each cleanup slice.

This state is enough to treat Streamlit as fallback-only for the validated
native routes above. It is evidence to begin legacy CSS cleanup planning, not a
blanket deletion order.
