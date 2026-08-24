# Copilot extraction defaults and stable-scroll repair

- Date: 2026-08-24
- Branch: `codex/easyicu-unified-product-20260823`
- Base: `2f424735397f7549aaf47b730bd8a90638a44826`
- Owner files: `screens-extraction.js` and `screens-extraction-embedded.js`

## User-visible changes

1. The embedded preview now captures and restores its own `scrollTop` around
   every owner repaint, so cohort, module, concept, and export choices do not
   send the user back to the top.
2. The extraction defaults are now the unrestricted ICU denominator:
   - all ICU stays
   - all ages
   - no minimum ICU length of stay
   - full cohort (`max_patients=0`)
   - readmissions retained
3. The recommended execution contract uses the same unrestricted denominator.
4. Minimum module coverage, quality-status filtering, filter preview, and
   "use matched modules" were removed from the frontend. Registered-source
   provenance and module row counts remain read-only context.

The 30-day observation horizon remains the existing technical feature-reading
ceiling; it is not a patient inclusion/exclusion filter.

## Verification

- Live reproduction before reload: embedded preview `scrollTop` changed from
  724 to 0 after clicking the already-selected All ICU option.
- Added executable `extraction_embedded_scroll.test.js`; a repaint now preserves
  a synthetic long-panel position of 640 px.
- Python focused matrix: `126 passed, 5 warnings`.
- Canonical JavaScript contracts: `28/28` passed.
- Node syntax, Ruff, and `git diff --check` passed.
- Browser loaded the new cache tokens, reported 0 console errors, 1814/1814 px
  body width, and a right preview ending exactly at the 1354 px viewport edge.
- No extraction was started and no patient rows or local folder contents were read.
