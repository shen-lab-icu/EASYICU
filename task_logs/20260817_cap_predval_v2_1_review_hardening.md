# CAP-PREDVAL-V2.1 review hardening

Date: 2026-08-17
Task: `CAP-PREDVAL-V2.1`
Branch: `codex/cap-predval-v2-1-20260817`
Worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-cap-predval-v1-20260817`
Base HEAD: `dd7a369`

## Decision

Keep the prediction-validation capability on its isolated experimental lane and
close the five concrete gaps found in the V2 review.  This patch does not
promote the capability, register it with Planner/runtime selection, write
EvidenceStore authority, or change the active Figure 2 lane.

## Reproduced review gaps

Tests were added before implementation.  Against the V2 code they produced
13 failures and 25 passes:

- pandas silently mangled duplicate raw CSV headers instead of refusing them;
- boolean and numeric-string outcomes/probabilities were implicitly coerced;
- non-finite PPV/NPV candidates escaped schema validation and reached canonical
  hashing;
- spec, result, and receipt digest helpers trusted unvalidated Pydantic
  `model_copy` instances;
- the frozen calibration oracle was effectively an identity calibration
  example and could not strongly detect swapped or simplified coefficients.

## Minimal closure

- The exact digest-bound bytes are decoded once for a strict raw-header check.
  Empty, whitespace-bearing, or duplicate headers are refused, the explicit
  pandas parse settings are named by
  `easyicu.prediction_validation_csv_strict/1`, and parsed columns must exactly
  match the raw header.
- The numeric owner requires numeric, non-boolean dtypes before conversion;
  malformed direct DataFrames cannot acquire meaning through implicit bool or
  string coercion.  Numeric CSV tokens remain valid because the bound parser
  infers their numeric dtype before evaluation.
- Threshold optional metrics reject NaN and positive/negative infinity at the
  typed result boundary.
- All three canonical digest helpers dump and revalidate already-typed model
  instances before hashing, closing `model_copy(update=...)` bypasses.
- The R fixture now has deliberately non-identity calibration
  (`intercept=-0.95488534075126819`, `slope=0.36261329836791001`) and its exact
  source digest was resealed.  The base-R implementation itself is unchanged.

## Verification

- New owner/provenance suite: 38 passed, including the live base-R oracle.
- Owner, provenance, dynamic-prediction adjacency, method-kernel reachability,
  capability inventory governance, method-suite registry, existing R oracles,
  methods-package boundary, canonical contract re-exports, and module-graph
  suite: 140 passed, 1 skipped.
- The skip is the existing declared-unreachable-kernel parameter case.
- Capability inventory audit: OK.
- Research-agent module graph: 539 modules, 2,116 edges, 0 cyclic SCCs.
- Targeted Ruff lint and format checks: passed.
- Exact fixture digests:
  - source CSV:
    `a78eea969afc387153e3b9bb4ee0ec0d74dcf0dbe0673fa23b520a62f5c806bf`;
  - base-R script:
    `026f3775530662802d1c931ff60f37bb1eafb5c24b980abaaf900556eded9eb3`;
  - frozen oracle JSON:
    `673b535a7d66c26b89fce7e7b1d6eac824c0987bb53761130879f635224c47ec`.
- No full exact-head CI was run because this remains an isolated experimental
  slice, E1 is not yet 11/11, and this is not a freeze/merge/release checkpoint.

## Remaining gates

V2.1 is ready for independent human re-review, not production promotion.  A
governed product input route plus explicit code/runtime authority remains
required before Planner/runtime selection, EvidenceStore integration, or any
paper-facing claim can be considered.  Large-file/streaming ingestion, DeLong
confidence intervals, decision-curve analysis, model fitting/selection, and
dynamic-prediction production wiring remain separate owners.
