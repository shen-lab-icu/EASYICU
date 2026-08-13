# Figure 2–5 nature-figure layout mockups

Date: 2026-07-24  
Task: `FIGURE-LAYOUT-MOCK-20260724`  
Module: 论文图件  
Status: composition review; Figure 3 is source-backed but provisional

## Scope

The user requested first-pass Figure 2–5 templates using the `nature-figure`
workflow. Final result-bearing rendering used Python/matplotlib only and exported
editable SVG, one-page PDF and 300 dpi PNG. One image2 composition storyboard was
generated for the Figure 3 redraw; it contains no scientific values and was used
only to compare hierarchy and panel balance. No benchmark, database extraction or
discovery pipeline was run.

## Figure 2 correction

The first prototype incorrectly presented nine items as nine individual
questions. The user clarified that Figure 2 evaluates nine common ICU
scientific-question/method families, with multiple repeated runs contributing to
dimension-level scoring.

The revised layout follows `docs/figure2_taskbank_9x3_protocol.md`:

- nine method families, not nine single questions;
- A development, B frozen validation and C sealed confirmation are distinct
  scientific tasks;
- stochastic repeats are shown separately as r1–r3 within B/C;
- five dimensions remain separate; no heterogeneous composite headline score;
- terminal reportability and objective-error interception remain explicit.

The rejected process-box strip was removed after user review. Figure 2 now opens
directly with the repeated score atlas. The red top placeholder badge was removed
from all figures; a discreet bottom statement still marks every figure as
synthetic and non-manuscript evidence.

After a second user review identified crowded lower-panel typography, the
redundant dimension-level outcome-mixture panel was removed rather than made
smaller. The final review layout uses:

- one full-width hero matrix;
- one dedicated legend strip;
- two wide supporting panels for repeat agreement and pre-specified error
  detection.

This keeps the panel logic quantitative while removing a re-aggregation of the
same score-atlas evidence.

In the next user-directed revision, the two-line overall Figure 2 heading
(title plus protocol subtitle) was removed entirely. The panel grid was moved
upward so the deleted heading did not leave an empty top band; panel titles and
the bottom synthetic-placeholder disclaimer remain.

## Figure 3 correction

The initial Figure 3 mockup used three panels. Article-level review showed that
the limiting-component scatterplot re-plotted information already carried by the
coverage heatmap and complete-score bars, while implying a near-definition-driven
relationship. The user approved adjustment to the locked two-panel article
structure.

The revised Figure 3 therefore:

- removes panel c completely;
- aligns the six database rows exactly across panel a and panel b;
- uses the current Table 1 source snapshot for component coverage and complete
  six-component score availability;
- excludes the SICdb CNS structural-no-source cell from the heatmap colour scale
  and renders it with hatching;
- distinguishes HiRID liver measurement sparsity from SICdb CNS structural
  absence using direct labels, colour and texture;
- carries a provisional footer because the final six-database bounds/mapping
  spot-check and source freeze are still pending.

The required image2 composition reference is
`outputs/easyicu-figure-layout-mockups-20260724/figures/Figure3_two_panel_storyboard_image2.png`;
its prompt/provenance note is stored beside the package root. It is not a result
source.

## Artifacts

- Output package:
  `outputs/easyicu-figure-layout-mockups-20260724/`
- Contact sheet:
  `outputs/easyicu-figure-layout-mockups-20260724/figures/Figure2-5_layout_overview.png`
- Generator:
  `outputs/easyicu-figure-layout-mockups-20260724/build_layout_mockups.py`
- Figure contracts:
  `outputs/easyicu-figure-layout-mockups-20260724/figure_contracts.md`
- QA report:
  `outputs/easyicu-figure-layout-mockups-20260724/qa_report.md`
- Panel data:
  `outputs/easyicu-figure-layout-mockups-20260724/source_data/`

## QA

- Generator compiles and runs without warnings.
- All four figures have SVG/PDF/PNG exports.
- Canvas width is exactly 7.2 in / 518.4 pt (approximately 183 mm).
- PNG width is exactly 2,160 px at 300 dpi.
- SVG text remains editable (66–95 `<text>` elements per figure).
- Figure 3 CSV rows have `simulated=False` and an explicit pending-audit
  authority status; all remaining mockup CSV rows have `simulated=True`.
- Top `SIMULATED · LAYOUT ONLY` badges are absent.
- Bottom authority disclaimers remain present.
- All four PNGs were inspected at original resolution. The revised Figure 2
  2,160 × 1,815 px export has no overlapping or clipped titles, tick labels,
  family labels, legend entries or footer text.
- The revised Figure 3 2,160 × 1,245 px export has no overlapping or clipped
  panel titles, cell values, axis labels, legend entries or footer text.

## Authority boundary

This package is not yet cleared for manuscript use. Figure 2 paper-facing results
remain 0/9 under the current benchmark authority. Figure 3 uses current
source-backed values but remains provisional until the bounds/mapping spot-check
and source freeze. Figure 4 remains simulated. Figure 5 must come from the full
`tools/run_discovery_to_manuscript.py` pipeline. No simulated value or visual
trend may be copied into the manuscript.
