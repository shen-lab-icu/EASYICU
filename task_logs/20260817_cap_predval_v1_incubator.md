# CAP-PREDVAL-V1 prediction-validation incubator

Date: 2026-08-17
Task: `CAP-PREDVAL-V1`
Branch: `codex/cap-predval-v1-20260817`
Worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-cap-predval-v1-20260817`
Base HEAD: `cc4f9226c74bc321c3b2fb2540061273f3774ee6`

## Decision

Add one parallel, experimental capability slice without changing the active
Figure 2 E1-H3 runtime. The slice validates already-produced binary risk
probabilities; it does not fit or select a model and does not choose a cohort,
outcome, split, threshold, or analysis unit.

It remains outside Planner, the production runner registry, EvidenceStore, and
paper-facing authority. Promotion requires an independently reviewed oracle,
data/artifact provenance binding, a governed product route, and an explicit
production decision.

## Owner boundary

- `contracts/prediction_validation.py` owns the immutable declaration, result
  schema, canonical contract digest, stable refusal codes, and structured
  result findings.
- `methods/prediction_validation.py` owns deterministic AUROC, Brier score,
  joint logistic calibration intercept/slope, denominator-bearing quantile
  calibration bins, and predeclared threshold operating characteristics.
- `execution/runners/prediction_validation_executor.py` is the experimental
  direct-call adapter. Its result validator recomputes the complete result and
  rejects any schema-invalid or changed candidate.
- `docs/research_agent_capability_inventory.md` explicitly records the runner
  as `experimental`; no production reachability is claimed.

## Fail-closed behavior

The owner rejects missing declared columns, empty input, missing identities,
duplicate analysis units, empty split labels, subject overlap across splits,
an absent evaluation split, repeated subject rows in a subject-level analysis,
invalid outcomes, invalid probabilities, and a single-class evaluation split.

Calibration coefficients are never fabricated. Constant probabilities,
perfect separation, and numerical non-convergence return explicit
non-estimable status with null intercept and slope. Threshold metrics retain
confusion counts and use null PPV or NPV when its denominator is zero.

## Verification

- New synthetic oracle and negative/fail-closed suite: 20 passed.
- Prediction-validation, existing dynamic-prediction adjacency, kernel
  reachability, and capability inventory governance: 63 passed, 1 skipped.
- Method-suite registry, methods-package boundary, and canonical contract
  re-export checks: 38 passed.
- Capability inventory audit: OK.
- Research-agent module graph: 539 modules, 2,115 edges, 0 cyclic SCCs.
- Targeted Ruff and formatting checks: passed.
- `git diff --check`: passed.
- No full exact-head CI was run because this is an isolated experimental slice,
  E1 is not yet 11/11, and this is not a freeze/merge/release checkpoint.

## Next gate

Review this diff independently before any integration. If accepted, the next
capability slice should add provenance binding plus an external oracle for the
same contract. DeLong confidence intervals and decision-curve analysis remain
separate owners and must not be folded into this slice opportunistically.
