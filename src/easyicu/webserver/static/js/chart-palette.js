/* ============================================================
   chart-palette.js — the categorical chart palette owner.
   Owns the series colours, the per-index stroke patterns, and
   nothing else. Depends on no other module so the SVG fallback
   renderers — which run precisely when the ECharts shell failed
   to load — read the same list the charts do.

   Six palettes used to live in five route files across three
   colour systems (hex, oklch, and a fourth ad-hoc list in the
   cohort route). Fixing an adjacency in one of them left the
   other five shipping it, including the fallback paths.
   ============================================================ */
(function () {
  'use strict';

  const FALLBACK_ACCENT = '#0f766e';

  /* Ordered so the most common case reads first: a two-group contrast (KM
     curves, SOFA strata, a two-database comparison) gets teal vs amber —
     opposite temperature, ~145 degrees of hue apart — rather than two colours
     from the same half of the wheel.

     Under deuteranopia the surviving channels are blue-yellow and lightness,
     so no two entries share both. The pairs that previously converged were
     #2563eb next to #7c3aed and, in the cohort route, #2563eb next to
     #8b5cf6: ~35-40 degrees of hue at nearly equal lightness, indistinguish-
     able at a 2.6px stroke.

     Index 5 is a near-neutral slate: at six series there is no chromatic hue
     left that cannot converge with one of the five, so it separates by
     lightness and chroma instead. Six is the ceiling because six databases
     are supported; nothing renders more categories than that. */
  const SERIES = [
    null, /* resolved from --accent, which the appearance tokens can retheme */
    '#b45309',
    '#2563eb',
    '#be123c',
    '#a21caf',
    '#334155',
  ];

  /* Colour is never the only channel: each index also gets its own stroke, so
     a line chart stays readable in greyscale, in print, and under any colour
     vision deficiency. Index 0 stays solid — the primary series should not
     look provisional. The list is the same length as SERIES so a full cycle
     never repeats a (colour, stroke) pair. */
  const DASHES = [null, [7, 4], [2, 3], [11, 4, 2, 4], [5, 3, 1, 3], [14, 5]];

  function accent() {
    try {
      const value = getComputedStyle(document.documentElement)
        .getPropertyValue('--accent')
        .trim();
      return value || FALLBACK_ACCENT;
    } catch (_) {
      return FALLBACK_ACCENT;
    }
  }

  /* Returns concrete colour strings, never `var(--accent)`. ECharts parses
     colours itself rather than handing them to CSS, so a custom property in an
     option renders as its neutral fallback (measured: `#dbdee4`). SVG
     presentation attributes do resolve var(), but the option path decides the
     representation and one representation is the point. Reads --accent once,
     so a caller colouring many rows should hoist this out of its loop. */
  function series() {
    const resolved = accent();
    return SERIES.map(colour => (colour == null ? resolved : colour));
  }

  function slot(index, list) {
    const position = Math.floor(Number(index));
    if (!Number.isFinite(position) || position < 0) return list[0];
    return list[position % list.length];
  }

  function color(index) {
    return slot(index, series());
  }

  function dashPattern(index) {
    return slot(index, DASHES);
  }

  function lineStyle(index, options) {
    const settings = options && typeof options === 'object' ? options : {};
    const dash = dashPattern(index);
    const style = { width: settings.width == null ? 2.6 : settings.width };
    if (dash) style.type = dash;
    return style;
  }

  window.EU_PALETTE = {
    SIZE: SERIES.length,
    color,
    dashPattern,
    lineStyle,
    series,
  };
})();
