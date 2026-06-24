# Native FastAPI Stage20 Archive Readiness Audit

Date: 2026-06-24

Scope: audit the native FastAPI WebApp path for fallback-only/archive readiness.
This audit did not modify `src/easyicu/webapp/**`, did not run Streamlit, did
not commit changes, and did not call an external provider.

One audit finding was remediated during the run: route QA initially found
mobile `#settings` horizontal overflow of 22 px from long mono environment
values in the native Settings page. The fix is a local native CSS constraint in
`src/easyicu/webserver/static/css/pages.css`. The full route matrix was rerun
after the fix and passed.

## Decision

Stage20 passes for the fallback-only threshold.

- The native FastAPI path covers the validated WebApp routes and parity
  blockers needed to downgrade legacy Streamlit from active development to
  frozen fallback/reference.
- A separate legacy CSS cleanup/archive plan may start next.
- This is not permission to delete the whole Streamlit stack in the same change.
  Keep cleanup separate, evidence-backed, and reversible until the fallback
  snapshot and launch paths are explicitly handled.

## Route And Browser Matrix

Evidence: `output/playwright/native_fastapi_route_qa_20260624_101916/route_qa.json`.

The route QA covered 23 browser checks:

- Desktop and mobile `393x852`: `entry`, `extraction`, `patient`, `cohort`,
  `crossdb`, `agent`, `settings`, `dictionary`, `states`, `help`, `guided`.
- Runtime unknown hash fail-safe: `#__qa_unknown_hash__` falls back to `#entry`.
- Result: max `overflowX=0`, JavaScript console errors `0`, non-empty main
  content for every route.

`#help` opens the Help/Get Started alias rather than leaving an old screen or a
blank screen.

## API And Parity Evidence Matrix

| Area | Evidence | Result |
|---|---|---|
| Extraction filters | `output/playwright/native_fastapi_extraction_filters_20260624_102113/extraction_filter_qa.json` | Real active registered fixture source; `/api/extraction/filter-options` and `/api/extraction/filter-preview`; browser real provenance shown; unsupported filters shown blocked; no seeded demo copy; `overflowX=0`; console errors `0`. |
| Patient Review | `output/playwright/native_fastapi_patient_drilldown_20260624_102120/patient_drilldown_qa.json` | Active registered fixture source; bounded pseudonymous drilldown; 3 entities, 5 modules, 4 selected signals; no demo copy; no raw markers in browser; `overflowX=0`; console errors `0`. |
| Cohort Review | `output/playwright/native_fastapi_cohort_parity_20260624_102127/cohort_parity_qa.json` | Active registered fixture source; cohort size 3, mortality 33.3%, median SOFA-2 6.5, LOS/age/sepsis aggregates, module quality; blocked inferential/matched/pairwise features; no demo copy; `overflowX=0`; console errors `0`. |
| Cross-DB Review | `output/playwright/native_fastapi_crossdb_parity_20260624_102134/crossdb_parity_qa.json` | Two registered fixture exports; descriptive range rows, module availability, compatibility gate; `matched_cohort=false`, `inferential_statistics_allowed=false`, `claim_level=preview_not_reportable`; no demo copy; `overflowX=0`; console errors `0`. |
| Agent run/history/review/signoff | `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` | 60 passed. Covers active registry-backed run, bounded artifacts, review, local signoff, history, artifact viewer/download, stale/tamper review checks, and locked reportability. |
| Numeric evidence gate | Same pytest run | Positive numeric binding and negative mismatch/ghost/missing-evidence cases pass; full mock run stays `reportable=false` and `draft_unlocked=false`; tampered numeric claim blocks with `numeric_evidence_gate_failed`. |
| Provider status | Runtime dormant probe | `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`, env file status `disabled`. |

## Fail-Closed Matrix

| Gate | Evidence | Status |
|---|---|---|
| Cross-DB fewer than two registered exports | `test_crossdb_review_summary_fails_closed_until_two_registered_sources` | HTTP 400 fail-closed; no synthetic second source. |
| Unsupported extraction filters | `test_extraction_filter_preview_rejects_unsupported_filters` | HTTP 400 fail-closed. |
| Unsupported Cohort filters/statistics | `test_cohort_review_summary_rejects_unsupported_filters_and_statistics` | HTTP 400 fail-closed. |
| Unsupported Cross-DB filters/statistics/matched cohort | `test_crossdb_review_summary_rejects_unsupported_filters_and_statistics` and Cross-DB QA | HTTP 400 fail-closed; UI labels unsupported analyses as blocked. |
| Provider AI disabled | Provider dormant probe and provider gate tests | Readiness false before client construction; no network. |
| Numeric mismatch or missing evidence | `test_numeric_evidence_audit_fails_mismatch_ghost_and_missing_evidence` and `test_full_agent_numeric_evidence_gate_blocks_mismatched_mock_claim` | Gate blocks with mismatch/ghost/missing evidence reasons. |

## Privacy Matrix

The backend and browser QA assert no row-level payload markers in the native
parity paths:

- `stay_id`
- `subject_id`
- `hadm_id`
- `tableRows`
- `"series"`

The native APIs return source labels, path hashes, aggregate metrics, bounded
pseudonymous entities, module coverage, and gate status. They do not return
patient row tables or direct clinical identifiers for the audited paths.

## Commands Run

```bash
python -m compileall -q src/easyicu/webserver
find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check
pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py
python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8771/
python tools/qa_native_fastapi_extraction_filters.py --port 8772
python tools/qa_native_fastapi_patient_drilldown.py --port 8773
python tools/qa_native_fastapi_cohort_parity.py --port 8774
python tools/qa_native_fastapi_crossdb_parity.py --port 8775
git diff --check
```

Results: compile passed, JS check passed, pytest 60 passed, route QA passed,
all four fixture browser/API QA scripts passed, and `git diff --check` passed.

## 8765 Process Note

`127.0.0.1:8765` was still occupied by an older uvicorn process started many
hours earlier. The audit did not kill it. Route QA used a controlled isolated
server on `127.0.0.1:8771`; fixture QA scripts used temporary servers on
`8772`-`8775` and shut them down.

To view the latest native FastAPI UI manually, restart `8765` intentionally or
launch a new local port with the current working tree.

## Remaining Work Boundary

There are no Stage20 blockers to treating Streamlit as frozen fallback/reference
for the audited paths.

The following remain intentionally blocked native capabilities, but they are
not blockers to fallback-only archive readiness:

- Cohort row-level filters and inferential statistics.
- Cross-DB matched cohorts, p-values/SMDs, and paired reclassification.
- Any automatic `reportable=true` or draft unlock; current gates still keep
  reportable output locked.
- Real external provider use without explicit operator opt-in and credentials.

## Cleanup-Ready Conditions

Before deleting or deeply pruning legacy Streamlit CSS, run a separate cleanup
goal with these boundaries:

1. Freeze or tag the current Streamlit fallback/reference state.
2. Keep native FastAPI fixes separate from legacy CSS cleanup commits.
3. Do not delete user data, exports, provider env files, agent runs, or project
   artifacts.
4. Keep or document the fallback launch path until the operator accepts its
   removal.
5. Re-run native route QA and focused backend tests after each cleanup slice.
6. Do not unlock reportable drafts or external provider behavior as part of CSS
   cleanup.
