# H3 signed trajectory analysis validation

Date: 2026-08-24 EDT  
Branch: `codex/dev9-quality-remediation`  
Exact HEAD: `5451192ccf83938185d38d483dd50cb14648f951`

## Outcome

The locked H3 analysis completed all four signed deterministic owners under the
exact-head image and is now `analysis_validated=true`. The scientific result is
an explicit non-solution, not a recovered phenotype: minimum BIC occurred at the
upper prespecified boundary (`k=6`), so the runtime retained
`H3_NO_INTERIOR_BIC_OPTIMUM`, executed zero stability refits, bound no outcome,
and emitted no authorized phenotype labels.

This is a development `analysis_only` result. It is not manuscript-ready,
paper-authorized, externally replicated, or evidence that a stable biological
phenotype exists.

## Exact evidence

- Image: `easyicu-research-agent:5451192`
- Image ID: `sha256:bc4a1df8cc6831ff44d462932c7b580208432ebdb38b39c2d4db8f02703b5d81`
- Locked plan SHA-256: `486a12676f24f65cb87b10889e33ddcca6f06403339dfb030bcdb59134384ef7`
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h3_5451192_analysis_replay_20260824/h3_trajectory_clustering/aware/run_20260824T124822_ae50b3`
- Required/completed: `4/4`; missing `[]`; failed `[]`
- Capability: `trajectory_signed_phenotyping_v1`
- Provider calls/tokens/cost: `0 / 0 / $0.00`
- `run_status.json` SHA-256: `9e129a9464eb6cacab93e16e3ef2433e92f495b3245d2a143614c111805e3ae3`
- Diagnostic PNG SHA-256: `0e78c3e34acc1559969752f0cffd526315f175c1256c17205667303ad3c98ba3`
- FigureContract SHA-256: `2912d2ff26a535f4f4bec5e3a52c7fb73ea03823986f3c318c8bd299d86dbd64`

## Owner-scoped repairs

1. `932456e` added a case-neutral signed trajectory scientific validator. It
   verifies the representation, closed candidate grid, replayed BIC decision,
   cross-step artifact digests, stability authorization, absence of outcome
   binding, and a coherent failed-closed non-solution.
2. The first exact-head replay stopped before execution because generic article
   shaping plus the four-step cap temporarily removed the stability suffix.
   It made zero Provider calls and produced no scientific result.
3. `5451192` changed the trajectory authority owner to recognize the minimal
   digest-bound representation/candidate prefix and rebuild the complete four
   owners after generic shaping. It did not change the locked plan, candidate
   grid, BIC values, threshold, or H3 result.

Related focused/adjacent verification: `252 passed, 2 skipped`; Ruff and
`git diff --check` passed.

## Visual and claim-boundary review

The diagnostic figure contains a BIC-by-K panel and an observed coordinate
availability heatmap. Original-resolution review found no clipping or overflow.
The boundary point is explicitly marked as unauthorized; the figure contains no
phenotype names, outcome comparisons, or implied external reproducibility.

## Dev9 handoff

Do not start Qualification12 or Held-out27 yet. Complete the gold-free Dev9
comparator shadow review, assign every actionable gap to one generic owner, run
the necessary focused replays, and only then freeze one exact HEAD/image and run
one full exact-head CI. Current known review targets include temporal/adjustment
authority in E1/E3/M1, duplicate or narrow robustness axes in E1/E2/E3/M1,
H1 non-PH handling, and the current M3 article/display suffix. The prior H1
mixed-panel adjustment-label finding was an evaluator lineage defect and is
fixed generically in `a602388`; it was not an error in the rendered figure.
