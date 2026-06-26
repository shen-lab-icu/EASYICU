# 2026-06-27 Patient Review Demo Clinical Shape Fix

## Issue

The native Patient Review demo workspace was generated from the EasyICU catalog with generic deterministic values. That made the UI preview look unlike real EasyICU exports:

- `charttime` rendered as calendar strings such as `2026-01-01 00:00`.
- SOFA/SOFA-2 component scores rendered as decimals.
- Small-range clinical values such as pH inherited the same generic drift used by labs/vitals.

The old Streamlit mock generator in git history (`src/easyicu/webapp/mock_data.py` before Streamlit decommission) used numeric hour offsets and clinically bounded feature ranges. This fix migrates that semantic shape into the FastAPI native demo layer without restoring the deleted Streamlit package.

## Changes

- Updated `src/easyicu/webserver/static/js/screens-viz-demo.js`.
  - Added explicit demo value helpers for boolean, integer score, and time-indexed fields.
  - `demoCharttimeAt()` now returns numeric hours: `0, 1, 2, 3, 4, ...`.
  - SOFA/SOFA-2 organ components are bounded integers `0-4`.
  - SOFA/SOFA-2 totals, qSOFA, SIRS, GCS, MEWS, and NEWS are integer-bounded.
  - pH, temperature, and small dose/rate values use narrower feature-specific drift.
- Updated `src/easyicu/webserver/static/js/screens-viz.js`.
  - Patient demo table preview now calls `demoCharttimeAt()` and `demoTableValue()` instead of hard-coded dates and generic `toFixed(2)` values.
- Updated `src/easyicu/webserver/static/index.html`.
  - Bumped `screens-viz-demo.js` cache key to `20260627-demo-clinical-shape`.
- Added `tests/test_webserver_patient_demo_data.py`.
  - Locks the clinical table-shape contract and prevents calendar `charttime` from returning.

## Verification

- `node --check` for all native JS files: passed.
- `python -m compileall -q src/easyicu/webserver`: passed.
- Focused pytest: `159 passed, 1 warning`.
- Browser smoke on temporary `127.0.0.1:8802`: passed.
  - `charttime`: `[0, 1, 2, 3, 4]`
  - `sofa2_resp`: integer values within `0-4`
  - `sofa2`: integer values
  - `ph`: `7.35-7.41`
  - boolean medication flag preview returns booleans

## Scope

This is demo/fixture shape only. Real Patient Review remains backed by `/api/patient-review/drilldown` and active registered exports.
