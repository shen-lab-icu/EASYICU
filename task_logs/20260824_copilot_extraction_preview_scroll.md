# Copilot extraction preview scroll repair

- Date: 2026-08-24
- Branch: `codex/easyicu-unified-product-20260823`
- Base before repair: `2f8f443994b41d967b8f86ed3a86443845e546e8`
- Owner: Guided Copilot right-side native-workspace preview

## Cause

The preview body clipped overflow, while its anonymous native-workspace mount did
not participate in the constrained flex-height chain. The extraction embed was
therefore sized to its 2,084 px content instead of the 1,268 px preview viewport;
its own `overflow:auto` never became a scrollport and the preview body discarded
the remaining content.

Browser measurements before the repair:

- preview body: client height 1,268 px; scroll height 2,084 px; overflow hidden
- native mount: client height 1,268 px; scroll height 2,084 px; overflow visible
- extraction embed: height/client/scroll height 2,084 px; overflow auto; bottom 2,171 px

## Repair

- Constrained `[data-gpi-native-workspace-mount]` to the preview body's flex height.
- Made `.gpi-extraction-embed` fill that constrained mount with `min-height:0`.
- Kept the rule in `guided-pi-preview.css`; standalone `extraction.css` remains
  free of Copilot-preview selectors.
- Bumped the preview stylesheet cache token.
- Added owner-presence and non-owner-absence regressions.

## Verification

- Focused red test failed before the CSS change and passed after it.
- `126 passed, 5 warnings`:
  - `tests/test_pi_copilot_extraction_workspace.py`
  - `tests/test_webserver_static_routes.py`
  - `tests/test_pi_copilot_static.py`
- Browser after reload:
  - preview body, native mount, and extraction embed all end at the 1,354 px viewport boundary
  - preview/body horizontal widths are 1,814/1,814 px
  - browser console errors: 0
- No patient rows or local folder contents were read.
