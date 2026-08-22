# E3 Dev9 12/12 deterministic association closure

Date: 2026-08-22

## Outcome

- Task: `e3_kdigo_gradient` (Dev9 development diagnostic only).
- Final execution: `12/12`; missing steps `[]`; failed steps `[]`.
- Final run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_1320e2c_e3_fresh_checkpoint010_20260822/e3_kdigo_gradient/aware/run_20260822T135715_464a68`.
- Exact clean Git HEAD: `1320e2c273f2424851a8a8d48b38e39345064a8c`.
- Exact runner image: `easyicu-research-agent:1320e2c`, digest `sha256:c743cde247271f84ec15e888e2975a07bcf39044cdcdfae097ba9696e0c538d0`.
- Final execution identity recorded `git_dirty=false`; repair ledger is empty.
- Provider use in the successful checkpoint execution: writer only, 6 calls, 79,969 reported tokens, estimated cost `$1.00197`. Planner and Coder calls were zero.
- Runtime: 120.74 seconds.

This is not a paper-authorized result. The run is `development_diagnostic`, final gate is `analysis_only`, `paper_authorized=false`, and the post-hoc Figure 2 scoring attempt is invalid because a separately frozen formal execution identity and complete Canonical9 authority are intentionally absent. The result must not be described as manuscript-ready, publication-ready, or a formal benchmark score.

## Fresh-plan boundary

E3 planning began from a fresh Planner run, but the Planner exceeded the development efficiency threshold after checkpoint 007. The process was stopped and resumed from that immutable checkpoint; three additional Planner calls produced checkpoint 010. The final execution reused that fresh checkpoint rather than restarting Planner:

- checkpoint 010: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_92d4bd7_e3_fresh_resume007_20260822/e3_kdigo_gradient/aware/run_20260822T132617_f9e59e/progressive_planner_checkpoint_010.json`
- SHA-256: `3494f1bad85ab92d6de9005eb5e04904b922a553f8f97afda67dc686f967a523`

Therefore the accepted evidence is a segmented fresh-plan plus execution-only checkpoint replay, not one uninterrupted fresh run.

## Generic owner-contract repairs

No E3, KDIGO, AKI, Sepsis, or benchmark-item-specific branch was added. Repairs were assigned to general owners:

- `8352c1e`: distinguish ordinal-linear exposure from categorical contrasts.
- `d9abd27`: recognize host-owned descriptive output.
- `4c2b04b`: add the typed ordered-stratified/ordinal-trend adapter.
- `a0990d8`, `f578b33`: preserve Planner checkpoint schema and product graph.
- `b70745a`, `366b4a8`: compile known Table One/exposure missingness.
- `00c04d9`, `1b4c5f2`: preserve typed ordinal inputs and close executor receipts.
- `92d4bd7`: add deterministic four-table association publication-figure execution.
- `786b6bf`: accept reviewed categorical ordinal parent specifications for ordinal-trend execution.
- `02fdb3b`, `63bcaef`: replay the registered primary code on locked complete-case variants and issue replay-local typed cohort bindings.
- `1320e2c`: admit either per-spec `robustness_matrix` or aggregate `robustness_summary` as typed figure evidence; render summary ranges without inventing point estimates.

Focused verification included 77 robustness-owner tests and 18 composite-figure/selection tests. A real E3 four-table smoke generated exact source CSV copies, PNG/SVG/PDF/TIFF, and a FigureContract; visual inspection confirmed the range renderer did not manufacture point estimates.

## Efficiency observations

- The checkpoint-execution suffix no longer depends on Planner/Coder for ordinal trend, complete-case replay, robustness display, or the primary four-table figure.
- One intermediate replay exposed `robustness_summary` versus `robustness_matrix`; it was stopped immediately after the first Coder/repair attempt, then fixed in the figure owner. Those calls are not part of the successful final run cost.
- The next Dev9 item should follow the same discipline: one fresh bounded plan, checkpoint reuse after a validated boundary, owner-local repair, focused tests, and no Qualification12/Held-out27 activity.

## Next action

Start M1 Dev9 only. Keep expert scoring deferred until the later formal-evaluation preparation stage. Do not run full exact-head CI after this single development item; reserve it for the Dev9 freeze or another explicit release checkpoint.
