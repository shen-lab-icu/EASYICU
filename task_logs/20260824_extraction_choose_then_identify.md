# Extraction choose-then-identify flow

- Date: 2026-08-24
- Branch: `codex/easyicu-unified-product-20260823`
- Base: `9775e30b25047d55fc4a163a51e196a2423a542c`
- Owner: `src/easyicu/webserver/static/js/screens-extraction.js`

## Problem

The primary action claimed that EasyICU would identify the folder, but when the
field contained the mounted-volume root `/Volumes`, it scanned that root
directly and returned "folder not recognized". The user then had to choose or
paste a more specific path, contradicting the primary-action copy.

## Repair

- Renamed the action to "Choose folder and identify".
- Empty paths, `/`, and `/Volumes` now open the local folder picker instead of
  being submitted as ICU data folders.
- After a folder is selected, EasyICU immediately runs the existing read-only
  structure scan; there is no second analyze click and no required manual path
  entry.
- The secondary Browse action and the error-state "Choose another folder"
  action use the same choose-then-identify path.
- The picker remains explicit because EasyICU does not recursively search the
  user's whole disk or mounted volumes.

## Verification

- Before repair, clicking Identify with `/Volumes` produced the live
  `unrecognized_folder` state.
- After repair, the primary button opened the folder picker at the user's home;
  the Volumes shortcut opened `/Volumes`.
- Selecting the harmless `/Users/haibo/test-results` directory and pressing
  "Use this folder" immediately closed the picker and ran the scan without a
  second click; the expected non-ICU result was shown.
- The picker was reopened and left at `/Volumes` for user continuation.
- Focused Python matrix: `126 passed, 5 warnings`.
- Canonical JavaScript contracts: `28/28`.
- Node syntax, Ruff, diff check, browser console (0 errors) passed.
- No extraction/conversion job was started and no patient rows were read.
