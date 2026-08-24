# Extraction output open actions and Copilot handoff receipt

Date: 2026-08-24

Branch: `codex/easyicu-unified-product-20260823` (isolated worktree)

Owner boundary: extraction output UI + jobs lifecycle route; Guided Copilot owns the visible conversation receipt.

## User-visible defect

- A completed extraction displayed an absolute local output path and output files as inert text.
- “同步回 Copilot / 在研究引导中继续” persisted a handoff but gave no visible evidence that the Copilot conversation had received it.
- The result summary undercounted supporting artifacts because the column-metadata sidecar was not projected in the file list.

## Implemented contract

- The completed output directory is a button. Clicking it opens the exact job-owned directory in the operating-system file manager.
- Each declared output artifact is a button. The server opens the file with its default application. On macOS, a file with no registered application (for example Parquet on a default installation) falls back to Finder reveal/select.
- The browser never submits an absolute path. `POST /api/jobs/{job_id}/open-output` resolves the directory from the completed extraction job result and accepts only a direct-child filename declared by that result. Unknown jobs, non-extraction/running jobs, traversal, undeclared files, missing files, and symlink escapes fail closed.
- Sync persists the extraction handoff in `StudyContext`, rebinds the active Guided Copilot session, and inserts a visibly labelled local workflow receipt into the conversation. The receipt reports database, cohort, modules/export format, file counts, row count, and local output directory.
- The receipt explicitly says it is EasyICU workflow state rather than a model reply. The host path is displayed locally and is not represented as model-authored text.

## Real local integration evidence

The implementation was exercised against the completed output:

`/Users/haibo/easyicu_export/easyicu_export_20260824_103732_miiv_parquet`

- Directory open returned `target=folder`, `method=finder`.
- `_manifest.json` returned `target=file`, `method=application` and opened with the registered application.
- `vitals.parquet` had no registered default application, so direct open failed and the macOS fallback returned `target=file`, `method=finder`, revealing the file in Finder.
- The result contains one Parquet data file with 7,891,741 rows plus five supporting files. The matched cohort receipt reports 94,458 ICU stays.

The existing user preview on port 8897 was deliberately not restarted during verification because its completed job is in process memory. Restarting would discard that result card even though the exported files remain on disk.

## Verification

- Focused Python matrix: `154 passed, 5 warnings`.
- Executable JavaScript contracts: `30/30` passed, including the embedded handoff/rebind/visible-receipt path.
- Ruff: passed for all edited Python and test files.
- `node --check`: passed for all edited JavaScript and the new executable contract.
- `git diff --check`: passed.
- Browser smoke on a temporary isolated instance: Guided and Extraction routes loaded, no console warnings/errors, and no horizontal overflow at 1280 px. Demo results remained non-clickable and truthfully stated that no files had been written.

## Scope boundary

This patch does not send local files or host paths to the model, does not expose arbitrary filesystem access, and does not change the extraction engine. It only opens artifacts already declared by a completed local extraction job and makes the existing handoff state visible in Guided Copilot.
