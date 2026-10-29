---
name: web-prototype
description: Build or revise a small, local, self-contained web artifact in the current EasyICU project workspace.
---

# Web prototype workflow

Use this workflow when the user asks for a webpage, calculator, dashboard, form,
interactive explanation, or another browser-viewable artifact.

1. List the current project files before choosing names.
2. Prefer one self-contained HTML file with inline CSS and JavaScript so the
   governed preview can render it without network access.
3. Label simulated values and unvalidated formulas explicitly. Never invent a
   clinically validated model, treatment recommendation, effect estimate, or
   patient-specific claim.
4. Use the write tool only to create a new artifact. Do not paste a full
   substitute code block into chat when the requested file can be created.
   To change an existing file, read it first and use the exact-edit tool with
   the returned `sha256` as `expected_sha256`, so unseen content or another
   session's changes cannot be silently overwritten.
5. For numerical charts or filters, include one declarative UI-check contract
   in `<script type="application/json" data-easyicu-runtime-qa>`. Use schema
   `easyicu.web-runtime-qa/1` and a `checks` array, with unique short `id` values.
   Declare intended checks rather than executable test code:
   - `text_number`: `selector`, `expected`, optional numeric `tolerance`.
   - `ratio`: `numerator`, `denominator`, `output` selectors, optional `scale`
     (default 100) and `tolerance` (default 0.051, for one-decimal percentages).
   - `bar_geometry`: `selector`, `value_selector`, numeric `domain_max` and
     pixel `extent`; optional `axis` (height or width), `min_extent` and
     `tolerance_px`. The observed bar size is compared with the declared scale.
   - `select_cases`: a select `selector` and up to six `cases`, each with a
     `value` and leaf `checks`. Declare expected filtered counts and dependent
     chart/table checks. The probe restores the original selection afterwards.
   - `download_control`: its `selector`. This remains pending until the Host
     accepts the current page's CSV export; an enabled button is insufficient.
   Selectors must identify one element. Keep the whole contract below 16 KiB
   and at most 64 checks, including cases. Example for a nonclinical count:
   `{"schema":"easyicu.web-runtime-qa/1","checks":[{"id":"visible-count","kind":"text_number","selector":"[data-record-count]","expected":12}]}`.
6. Read the saved file back, run the bounded static check, then request the web
   preview tool. A static pass and a prepared preview do not verify rendered
   charts, filter behavior, CSV transport, or actual file saving. The isolated
   preview runs only the bounded declared UI checks; it grants no science or
   download authority. Missing contracts produce partial observations.
7. Read `easyicu_get_project_runtime_qa` with `file` and the exact current
   `checked_sha256`. A missing receipt is `partial`/`browser_runtime_pending`;
   do not substitute a static result or your own claim for browser evidence.
   On `failed`, use the check ids and codes to fix the current file, preserving
   the declared data and failure history, then check and preview the new hash.
   On `partial`, state the uncovered interactions or pending CSV export. Do
   not loop or wait for a receipt that requires an unopened browser or a user
   download permission. `complete` means only the declared UI checks passed.
8. Summarize the exact version, observed coverage and remaining issues. Say
   “CSV ready to save” only when `csv_export_ready` is true; the runtime receipt
   always leaves actual browser/OS saving and scientific validation unverified.
   If a tool was blocked, say that no file was changed.

Keep all files relative to the isolated EasyICU project workspace. Do not ask
for or embed patient rows, identifiers, credentials, private source paths, or
external tracking scripts. Workspace file contents read by Pi may be sent to the
configured Pi model service, so never place PHI, patient rows, credentials, or
private clinical data in this workspace.
