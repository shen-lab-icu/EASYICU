# Patient local identifier search

- Date: 2026-07-30
- Task: `PATIENT-LOCAL-IDENTIFIER-SEARCH`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: Patient Review entity navigation

## User-facing problem

Real cohorts cannot be reviewed efficiently by paging through anonymous Entity chips. Users need to locate a known patient or ICU stay directly, while the browser must not expose or persist direct clinical identifiers.

## Implemented contract

- Added an exact patient-or-stay identifier search to real and official-demo Patient Review navigation.
- Search scans only the registered local export's identifier columns.
- A unique match opens the pseudonymous entity and automatically moves the fallback pager to the matching group.
- A patient identifier that maps to multiple ICU stays returns a bounded set of pseudonymous Entity choices.
- No raw identifier is returned in the API payload, rendered in result labels, added to the URL, or persisted by the client.
- Synthetic fallback mode does not show the real-data identifier search.
- Previous/next/random bounded navigation remains available as a secondary fallback.
- Frontend behavior remains in the Patient navigation JS/CSS owners; no shared catch-all override was added.

## Browser evidence

Official MIMIC-IV Demo at 1521×1354:

- Exact local stay-ID search selected `Entity 17`.
- The navigator moved from page 1 to the matching `13–24 / 140` page.
- The input cleared after the verified match.
- Raw identifier presence in body, URL, and local storage: false.
- Document and main-content horizontal overflow: 0.

Screenshots:

- `output/ui-qa/20260730_patient_entity_identifier_search/01-real-search-matched-pseudonymous.jpg`
- `output/ui-qa/20260730_patient_entity_identifier_search/02-real-search-auto-positioned.jpg`

## Verification

- Ruff on changed Python owners/tests: passed.
- Node syntax checks: passed.
- Executable Patient browse owner contract: passed.
- Patient browse backend/frontend tests: `14 passed`.
- Focused native Patient static route contract: passed.
- CSS owner presence/absence and brace/comment scans: passed.
- `git diff --check`: passed.
