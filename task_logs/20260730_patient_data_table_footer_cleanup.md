# PATIENT-DATA-TABLE-FOOTER-CLEANUP

Date: 2026-07-30
Branch: `codex/web-copilot-cockpit-lite-20260729`
Scope: Patient Review → Data Tables only

## Outcome

The Patient Data Tables view now ends after the selected module's bounded,
server-paged table. Four redundant footer areas were removed:

1. review-workspace summary and provenance notices;
2. the repeated cohort aggregate table;
3. the collapsed module-scope audit;
4. the exact-value matrix entry and bounded-preview reminder.

The module selector, pseudonymous bounded table, page-size control, previous/next
pagination, column-count status, and the global next-step CTA remain. The exact
value matrix is still available from Time Series, where it belongs; Data Quality
continues to own module/feature quality audit.

## Ownership

- Host composition: `src/easyicu/webserver/static/js/screens-viz.js`
- Table interaction owner: `src/easyicu/webserver/static/js/screens-viz-patient-tables.js`
- Table CSS owner: `src/easyicu/webserver/static/css/patient-tables.css`
- Cache wiring: `src/easyicu/webserver/static/index.html`

Removed owner-only JS exports and 51 lines of owner-only CSS instead of hiding
the UI with another override.

## Verification

- JS syntax: `screens-viz.js`, `screens-viz-patient-tables.js`
- Executable browse-owner contract: pass (`table_calls=4`)
- Focused Python gate: `79 passed, 1 deselected`
- CSS owner presence/absence plus brace/comment scan: pass
- `git diff --check`: pass
- Browser: MIMIC-IV Demo, Data Tables
  - switched from blood gas to `vitals`;
  - advanced to page 2 (`25-48 / 12,020`, `第 2 / 501 页`);
  - all four removed areas absent;
  - document/main horizontal overflow: false;
  - wide table scroll remains inside its owner container;
  - console errors: 0.

Screenshot:
`output/ui-qa/20260730_patient_data_table_footer_cleanup/patient-data-table-cleanup.png`
