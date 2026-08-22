# M1 Dev9 11/11 source-aware association closure

Date: 2026-08-22

## Outcome

- Task: `m1_hepatobiliary_missingness` (Dev9 development diagnostic only).
- Final execution: `11/11`; missing steps `[]`; failed steps `[]`.
- Final run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_9391248_m1_checkpoint009_20260822/m1_hepatobiliary_missingness/aware/run_20260822T142701_b453ea`.
- Exact clean Git HEAD: `93912482788062d8957faae21f8b1d43f0bb88d3`.
- Exact runner image: `easyicu-research-agent:9391248`, digest `sha256:c503fa4e7cd0e46787ec56ea42a90bf8d30316b3b0ec7255e0311fdb7814f495`.
- Final execution identity recorded `git_dirty=false`; repair ledger is empty.
- Provider use in the successful checkpoint execution: Writer only, 7 completed calls, 78,493 reported tokens, estimated cost `$1.03615`. Planner and Coder calls were zero. One additional Writer transport receipt was rejected before execution by `ProviderHardStopExceeded`; it has no usage and is not counted as a successful or charged call.
- Runtime: 140.08 seconds.

The run is not paper-authorized. It remains `development_diagnostic`, `diagnostic_only`, maturity `analysis_only`, and `paper_authorized=false`. The post-hoc Figure 2 scoring attempt is invalid because the formal frozen execution identity and full Canonical9 task authority are intentionally absent. This result must not be described as manuscript-ready, publication-ready, or a formal benchmark score.

## Fresh-plan boundary

M1 planning began from a fresh Planner run but was deliberately segmented at the development efficiency boundary:

- Fresh segment: 6 completed Planner calls, 61,777 tokens, checkpoint 004.
- Resume segment: 5 additional Planner calls, 50,386 tokens, checkpoint 009.
- Final checkpoint 009: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_e725824_m1_fresh_resume004_20260822/m1_hepatobiliary_missingness/aware/run_20260822T141156_a0ea6a/progressive_planner_checkpoint_009.json`.
- Checkpoint SHA-256: `17d0bd1a0d6240fbf0823fe5f93c49f681a8328474b2ffd73a760e33447df1b5`.

The accepted evidence is therefore a segmented fresh plan followed by execution-only checkpoint replay, not one uninterrupted fresh run.

## Generic owner-contract closure

No M1, bilirubin, liver, or benchmark-item-specific branch was added. The failure surface was assigned to the shared association-figure owner:

- `e0aa4eb` added the source-aware deterministic association renderer for the exact typed four-table profile: adjusted association, absolute-risk context, robustness summary, and measurement-process audit.
- The renderer preserves every source row, checks count/denominator and confidence-interval reconciliation, displays measurement availability among eligible observations, does not manufacture robustness point estimates, and emits PNG/SVG/PDF/TIFF plus FigureContract.
- The first exact-image execution completed 11/11 but exposed a cross-owner semantic mismatch: the step figure used non-canonical panel role labels, so the publication-figure skill declined promotion and regenerated an older two-panel association figure. This produced the false maturity finding `PRIMARY_FIGURE_ABSOLUTE_RISK_CONTEXT_MISSING` despite the step figure containing an absolute-risk panel.
- `9391248` aligned the deterministic renderer with canonical role values: `descriptive_result`, `primary_estimand`, `robustness`, and `data_quality`. The final execution promoted the same registered four-panel bundle to `publication_figures/` and the false absolute-risk finding disappeared.

Focused verification covered 59 association/publication-figure tests plus Ruff and `git diff --check`. The final run-level FigureContract SHA-256 is `81ae738dcaf9a43a797e78adb16b9aea6b26bd45bcc4c734f857a1ff929c14f6`; the final PNG SHA-256 is `4451c617b7c02a5a3acb8e02b5c96bf948b8e8dcf8df088e1920335144bfc503`.

## Input authority

- Input root: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_e725824_m1_input_20260822`.
- JSONL SHA-256: `d13f2c3b3776c03f5dcd0b5c4ca04cf5a8d7267d094cf213e4e0ba76679b71dd`.
- Development binding receipt SHA-256: `804114570fb9829e7438fd9fca910626286568b343989cc42595248a94f369d7`.
- Cohort parquet SHA-256: `7bb04799b1a30437d98b42f7592c90056f0da634e6b420c2a51433228ec0e4e0`.
- Cohort: 94,458 rows and 64 columns; materialization authority is `development_subset_not_paper_authority`.

## Next action

Start M2 Dev9 only. Keep Qualification12, Held-out27, expert scoring, broad Web work, and full exact-head CI deferred. Reuse the same discipline: fresh bounded plan, immutable checkpoint reuse at the efficiency boundary, owner-local generic repair, focused tests, and one final exact-image checkpoint execution.
