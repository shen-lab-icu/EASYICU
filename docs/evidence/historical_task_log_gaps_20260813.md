# Historical task-log gap register — 2026-08-13

The repository-hygiene pass compared every `EASYICU/task_logs/*.md` pointer in
the six module `CURRENT.md` files against both the current working copy and all
reachable Git history.  Referenced logs present on historical branches were
restored to the current branch.  The five names below were never committed and
could not be found in the workspace archive or cleanup quarantine.

| Missing historical log | What remains | Evidence status |
|---|---|---|
| `20260729_guided_interaction_audit.md` | A historical summary remains in `项目进度/web/HISTORY.md`. | Summary only; not independent evidence. |
| `20260729_patient_review_echarts_information_density.md` | A historical summary remains in `项目进度/web/HISTORY.md`. | Summary only; not independent evidence. |
| `20260729_patient_review_feature_audit.md` | A historical summary remains in `项目进度/web/HISTORY.md`. | Summary only; not independent evidence. |
| `20260730_cohort_survival_config_audit.md` | The dated audit conclusion remains in `项目进度/web/CURRENT.md`. | Original task log unavailable; re-run before relying on the claim. |
| `20260730_crossdb_quality_provenance_audit.md` | The dated audit conclusion remains in `项目进度/web/CURRENT.md`. | Original task log unavailable; re-run before relying on the claim. |

These gaps must not be silently replaced with reconstructed prose presented as
original evidence.  If a future task needs one of the claims, repeat the
focused browser/data audit and create a new dated task log with current code,
inputs, and receipts.
