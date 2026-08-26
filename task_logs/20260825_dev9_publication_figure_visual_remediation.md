# Dev9 deterministic publication-figure visual remediation

Date: 2026-08-25 EDT  
Renderer commit: `279bb589f9fbe37f35eaba6b415f60e7fb821767`  
Branch: `codex/literature-to-design-contract`

## Outcome

The eight result-bearing Dev9 figures were regenerated from the frozen Dev9
typed source tables with zero Provider calls. H2 remains intentionally without
a result figure because its feasibility owner failed closed before an
estimable comparison existed.

The remediation is owner-scoped and case-neutral:

- association figures now use reader-facing contrasts and asymmetric panel
  hierarchy;
- continuous-exposure figures promote the fitted curve and show sensitivity
  estimate ranges instead of a large execution-percentage panel;
- prediction figures make calibration the primary panel and disclose
  development/validation counts plus zero patient overlap;
- phenotyping figures promote the profile heatmap and retain candidate-only,
  analysis-only boundaries;
- landmark survival adds a source-derived number-at-risk table and preserves
  the PH-driven RMST promotion;
- trajectory selection removes decorative grids and states the fail-closed
  upper-boundary decision outside the data region;
- shared ICU fallback labels expand common internal abbreviations without
  overriding Planner-owned labels.

## Exact artifacts

Output root:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_visual_remediation_279bb58_20260825/`

Preview contact sheet:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_visual_remediation_279bb58_20260825/dev9_visual_remediation_contact_sheet.png`

Each of E1, E2, E3, M1, M2, M3, H1 and H3 has PNG, SVG, PDF and a
FigureContract; all except the established H1 survival owner also export TIFF.
The root contains 8 contracts, 8 PNG, 8 SVG, 8 PDF and 7 TIFF files.

## Verification

- Focused contract/renderer tests: `46 passed`.
- Earlier adjacent figure/claim tests in the same remediation: `89 passed`
  with one contract-order failure repaired and the focused regression rerun.
- `git diff --check`: passed before commit.
- All eight final render logs contain no traceback/error.
- E1/E2/E3/M1/M2/M3/H3 source rows and values match the prior frozen
  publication source data (`atol=1e-9`; differences are CSV float text
  round-trips only). H1 copied source CSV bytes match exactly.
- Original-resolution visual review covered all eight figures plus the final
  contact sheet; no clipped panel, overflowing title or raw long KDIGO
  contrast remained.
- Provider/Planner/Coder calls: `0`; model tokens/cost: `0`.

## Authority boundary

This is a deterministic visual and reporting-contract remediation. It does not
change any estimate, cohort, time zero, missingness rule, model, task score or
paper authority. Dev9 remains `analysis_only`, `paper 0/9`; these figures are
development-quality evidence and must not enter the formal Held-out27 result
denominator or manuscript numeric claims.
