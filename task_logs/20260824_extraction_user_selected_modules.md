# Data Extraction user-selected module defaults and clear-state repair

Date: 2026-08-24
Branch: `codex/easyicu-unified-product-20260823`
Baseline: `8a8dce2`
Scope: native Data Extraction owner plus the Copilot right-preview adapter; local `main` was not modified.

## Problem

- All 19 feature modules were initialized as selected.
- `exSelectedConcepts = {}` meant "use every catalog concept" even after the parent module was cleared, so expanded child feature controls stayed checked after **Clear all**.
- The Copilot compact header copied the separate **recommended extraction** card and could show six modules or start the recommended path even when the user's custom selection was empty.

## Repair

- Initialize every module unselected. **Select all** and **Core 6** remain explicit shortcuts.
- Treat an absent per-module concept list as all concepts only while that parent module is selected; clearing all parents now also renders all expanded children unselected.
- Expose a small owner `setupSummary` contract and make the embedded compact header and start button use the current custom selection, not the recommended configuration.
- Give the embedded owner its own HTML-escaping helper instead of reaching into the private closure of `screens-extraction.js`.

## Verification

- Focused Python: `6 passed` (`test_webserver_static_routes` selection/Copilot/output contracts plus extraction job continuity).
- JavaScript contracts: `extraction_embedded_handoff` and `extraction_embedded_scroll` passed.
- `node --check` passed for both extraction JS owners; `git diff --check` passed.
- Live in-app browser at `http://127.0.0.1:8897/#guided`:
  - initial state: `0 modules · 0 concepts`, demographics `0/6`, compact start disabled;
  - selecting demographics: `1 module · 6 concepts`, six visible child controls selected;
  - clicking **Clear all**: `0 modules · 0 concepts`, demographics `0/6`, zero selected child controls, clear/start disabled;
  - compact header followed `0 → 1/6 → 0`, no recommended-run button remained;
  - no new console warning/error after the repaired interaction and no horizontal page overflow.

## Boundary

This validates selection state and UI synchronization against a recognized local SICdb source. It does not claim a new MIMIC extraction, clinical cohort result, or full release/CI checkpoint.
