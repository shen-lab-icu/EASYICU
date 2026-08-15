# Patient table search and header deduplication

- Date: 2026-07-30
- Module: web
- Task ID: `PATIENT-TABLE-SEARCH-AND-HEADER-DEDUP`
- Branch: `codex/web-copilot-cockpit-lite-20260729`

## Outcome

- Added exact local patient/stay ID search to the Patient Review data-table tab.
- Search results remain pseudonymous. The raw identifier is cleared after submission and is never sent to the table-preview endpoint or returned in its payload.
- The table-preview endpoint now verifies the pseudonymous entity reference and ordinal before filtering the selected module.
- Patient scope persists across module changes and pagination; clearing the filter restores cohort-wide module pages.
- Renamed `Review features` / `Observed features` to `Catalog features` / `Features with data`, with a visible definition: at least one non-null value in the current export.
- Simplified the loaded strip to readiness plus Export. Dataset provenance stays in the source card, counts stay in the data-table summary, and Edit setup stays only in the topbar.

## Verification

- `107 passed, 1 deselected` across:
  - `tests/test_webserver_patient_browse.py`
  - `tests/test_webserver_patient_browse_frontend.py`
  - `tests/test_webserver_route_contracts.py`
  - `tests/test_webserver_static_routes.py`
- The deselected static test is the existing worktree-path assertion that expects the checkout basename to be exactly `EASYICU`; this isolated worktree is named `easyicu-copilot-cockpit-lite`.
- Node owner contract passed with `table_calls: 5`.
- Python and JavaScript syntax checks passed.
- CSS ownership, brace/comment balance, and unrelated-route absence checks passed.
- `git diff --check` passed.

## Browser QA

Official MIMIC-IV Demo v2.2:

- exact stay search resolved to one pseudonymous `Entity 1`;
- the raw identifier disappeared from the input and rendered page;
- blood-gas preview narrowed from `954` rows to `11` rows for the selected entity;
- switching to `vitals` retained the entity filter and showed `104` rows;
- clearing the filter restored the `12,020`-row cohort table;
- the loaded strip contained no dataset name or repeated counts;
- one visible Edit setup action remained in the page topbar;
- no document or main-content horizontal overflow;
- zero console errors.

Screenshot:

`output/ui-qa/20260730_patient_table_search_header_dedup/patient-table-search.png`
