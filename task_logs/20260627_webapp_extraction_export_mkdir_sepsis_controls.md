# 2026-06-27 WebApp extraction export mkdir + code-aligned Sepsis controls

## Scope

- Route: native FastAPI WebApp `#extraction`.
- User reports:
  - Export destination path has browse but no way to create a new local folder.
  - Sepsis-3 timing block cannot be a static explanation. It must reflect the actual EasyICU Sepsis diagnosis code parameters, and those choices must affect extraction, not only manifest text.

## Changes

- Added local-only directory creation backend:
  - `src/easyicu/webserver/dataio.py`: `create_dir(raw_path)` creates local directories with `parents=True`, fails on file-path collisions, and never deletes/renames.
  - `src/easyicu/webserver/app.py`: `POST /api/fs/mkdir`.
  - `src/easyicu/webserver/static/js/api.js`: `window.EU_API.createDir`.
- Updated extraction UI:
  - `src/easyicu/webserver/static/js/screens-extraction.js`: export destination now has `New folder` / `新建目录`; the folder picker can create a destination folder.
  - `src/easyicu/webserver/static/css/extraction.css`: owner styles for the export destination button, picker creation row, and Sepsis panel.
  - `src/easyicu/webserver/static/js/screens-extraction-sepsis.js`: new owner module for Sepsis parameter state, rendering, contract generation, and click binding.
  - `src/easyicu/webserver/static/index.html`: wires `screens-extraction-sepsis.js` before `screens-extraction.js` and bumps extraction cache versions to `20260627-extraction-sepsis-runtime`.
- Aligned Sepsis controls to the implementation:
  - `susp_inf()` parameters exposed/persisted: `si_mode`, `abx_win`, `samp_win`, `abx_count_win`, `abx_min_count`, `positive_cultures`.
  - `sep3()` / `sep3_sofa2()` parameters exposed/persisted: `si_window`, `si_lwr`, `si_upr`, `delta_fun`, `sofa_thresh`, `keep_components`.
  - UI copy states that `ΔSOFA >= 2` is the default Sepsis-3 criterion; `ΔSOFA >= 3` and non-default modes are sensitivity/strategy choices supported by the code.
- Connected UI selections to runtime:
  - `src/easyicu/webserver/dataio.py`: normalizes `cohort.sepsis_definition` into `runtime_kwargs`, preserves legacy field aliases, and passes those kwargs to Sepsis-derived cohort prefilters and Sepsis modules only.
  - `src/easyicu/concept/callbacks.py`: `_callback_susp_inf`, `_callback_sep3`, and `_callback_sep3_sofa2` now consume the relevant kwargs and forward them to the score detectors.
  - `tests/test_webserver_static_routes.py`: locks split owner wiring, runtime kwargs, README manifest output, and callback/runner wiring.

## Verification

- `node --check src/easyicu/webserver/static/js/screens-extraction.js`
- `node --check src/easyicu/webserver/static/js/screens-extraction-sepsis.js`
- `python -m py_compile src/easyicu/webserver/dataio.py src/easyicu/concept/callbacks.py`
- `python -m pytest tests/test_webserver_static_routes.py -q`
  - Result: 43 passed.
- `python -m pytest tests/test_sepsis.py -q`
  - Result: 6 passed.
- `git diff --check -- src/easyicu/webserver/static/js/screens-extraction.js src/easyicu/webserver/static/js/screens-extraction-sepsis.js src/easyicu/webserver/static/css/extraction.css src/easyicu/webserver/static/index.html src/easyicu/webserver/dataio.py src/easyicu/concept/callbacks.py tests/test_webserver_static_routes.py`
  - Result: passed.
- Owner scan:
  - Sepsis controls are in `screens-extraction-sepsis.js` and `extraction.css`.
  - No Sepsis selectors or `EUExtractionSepsis` wiring found in `redesign.css`, `tweaks.css`, `pages.css`, broad `app.js`, or `api.js`.
- Browser QA on `http://127.0.0.1:8765/#extraction`:
  - Restarted local FastAPI WebApp on `127.0.0.1:8765`.
  - In Chinese mode, the Sepsis panel shows code-aligned controls.
  - Clicking `Δ >= 3` updates the summary chip to `ΔSOFA >= 3`.
  - Clicking `抗菌药 + 采样` updates the SI mode summary.
  - Clicking `SOFA-1 敏感性` updates the profile summary.
  - Browser console errors: 0.

## Next

- Further reduce `screens-extraction.js` below the soft budget by splitting another internal area, likely entry/home or folder picking, in a separate owner-scoped cleanup.
