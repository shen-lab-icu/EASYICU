# Patient search prefix and catalog language QA

- Date: 2026-07-30
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Task: `PATIENT-SEARCH-PREFIX-AND-CATALOG-LANGUAGE`

## Outcome

- Patient navigation, table preview, and multi-patient comparison now use the
  same local identifier-search contract:
  - a full patient/stay ID is resolved as an exact match first;
  - at least three leading characters can return up to 12 pseudonymous
    candidates;
  - raw identifiers and the submitted query are not returned to the browser.
- Replaced the ambiguous catalog phrase “导出中有观测” with a reader-facing
  definition: a feature is counted when the full export contains at least one
  non-null value. This does not imply that every patient has the feature or
  that the feature is a drawable time series.
- Removed the global “current-patient trajectories loaded” total. The fourth
  catalog summary now says “按需” and explains that opening a module generates
  the current-patient views. Per-module availability, missingness, and
  trajectory counts remain visible where they are interpretable.

## Verification

- Python contract suite: 107 passed, 1 deselected.
- Patient series owner contract: passed.
- Patient browse owner contract: passed.
- Patient route CSS ownership, foreign-selector absence, brace balance, and
  comment balance contract: passed as part of the frontend suite.
- Python compile, JavaScript syntax checks, and `git diff --check`: passed.
- Browser QA against the official MIMIC-IV demo:
  - a three-character prefix returned one bounded anonymous candidate
    (`Entity 1`);
  - the raw search input was cleared after the request;
  - the legacy “21 条当前患者轨迹已加载” copy was absent;
  - the export-wide non-null definition and on-demand behavior were present;
  - no document, main-content, or feature-catalog horizontal overflow at
    1280 × 720.

## Evidence

- Screenshot:
  `output/ui-qa/20260730_patient_search_prefix_catalog_language/patient-search-prefix-catalog.png`
