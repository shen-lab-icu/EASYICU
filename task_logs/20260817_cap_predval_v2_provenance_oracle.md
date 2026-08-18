# CAP-PREDVAL-V2 provenance receipt and independent oracle

Date: 2026-08-17
Task: `CAP-PREDVAL-V2`
Branch: `codex/cap-predval-v2-20260817`
Worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-cap-predval-v1-20260817`
Base HEAD: `381216f`

## Decision

Extend the isolated prediction-validation incubator by closing two V1 review
gaps without changing the active Figure 2 runtime: bind evaluation to the exact
CSV bytes that were authorized, and compare the Python owner against a separate
base-R implementation.

The slice remains experimental. It is not registered with Planner or runtime
selection, does not fit or choose a model, writes no EvidenceStore authority,
and cannot authorize a manuscript result.

## Exact-byte provenance receipt

- `run_prediction_validation_csv` accepts one regular `.csv` artifact plus an
  expected lowercase SHA-256 digest.
- The runner reads the bytes once, verifies their digest, and parses those same
  bytes as UTF-8 with `pandas.read_csv`; it does not hash one read and analyze a
  second read.
- The frozen receipt binds the portable artifact name, exact byte digest and
  size, parser identity/version, parsed row count and columns, contract digest,
  result digest, and complete deterministic result.
- `paper_authorization` is structurally fixed to `false`.
- `prediction_validation_receipt_findings` schema-validates and fully
  recomputes a candidate. Source absence, invalid digest, digest drift, parse
  failure, receipt-schema tampering, and valid-but-changed receipt metadata are
  attributable through stable owner reason codes/findings.

## Independent implementation oracle

- Source fixture SHA-256:
  `9487a6abdc64699795d6406de77049149fcd40c5a76e390b9f17713e917492b4`.
- Base-R script SHA-256:
  `026f3775530662802d1c931ff60f37bb1eafb5c24b980abaaf900556eded9eb3`.
- The R implementation independently computes rank-sum AUROC, mean-squared
  Brier score, joint calibration intercept/slope with `stats::glm`, and
  explicit confusion counts at the declared thresholds.
- The frozen oracle is always checked against both fixture digests and the
  Python output. When `Rscript` is installed, the same test also executes the R
  script and compares its live JSON output with the frozen oracle.
- This is an independent implementation oracle, not an independent human code
  review or a production authorization.

## Verification

- V1 baseline before editing: 28 passed.
- Owner, provenance, dynamic-prediction adjacency, method-kernel reachability,
  capability inventory governance, method-suite registry, existing R oracles,
  methods-package boundary, and canonical contract re-exports: 117 passed,
  1 skipped.
- The live base-R oracle passed with R 4.6.0 on this machine; the skip is one
  unrelated declared-unreachable-kernel parameter case.
- Capability inventory audit: OK.
- Research-agent module graph: 539 modules, 2,116 edges, 0 cyclic SCCs.
- Targeted Ruff, formatting, and `git diff --check`: passed.
- No full exact-head CI was run because this is an isolated experimental slice,
  E1 is not yet 11/11, and this is not a freeze/merge/release checkpoint.

## Remaining gates

Before promotion, obtain an independent human review and define a governed
product input route with explicit code/runtime authority in addition to the
data, declaration, and result bindings added here. Planner/runtime selection
and EvidenceStore integration remain intentionally absent until that decision.
Large-file/streaming ingestion is also not claimed by this direct experimental
CSV route.

DeLong confidence intervals, decision-curve analysis, model fitting/selection,
and dynamic-prediction production wiring remain separate owners and should not
be folded into this slice opportunistically.
