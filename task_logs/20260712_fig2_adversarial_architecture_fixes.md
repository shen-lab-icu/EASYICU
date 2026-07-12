# FIG2-CANONICAL9-GATE — adversarial architecture fixes

- Date: 2026-07-12
- Module: `benchmark实验`
- Branch: `fix/easyicu-concept-bounds-enforcement`
- Scope: repair general research-agent architecture after the review of `0f1dab2..HEAD`; no benchmark-specific runner or prompt rule was added.

## Outcome

The confirmed FigureSource and trajectory-clustering findings were fixed, then the same trust boundaries were audited more broadly. Two adversarial passes found and closed additional fail-open paths in current-record authority, repair credit, robustness locks/replay, primary-model evidence, typed routing, study-design role matching, O22/O23/O24 evidence selection, and FigureSource semantic binding.

The architecture now keeps scientific method/cohort/exposure/outcome selection with the agent. Deterministic runners are restricted to closed auxiliary products or evidence-bound rendering, and cannot credit or replace a failed primary scientific step.

## Local commits

- `dddc4be fix(agent): bind reports to current evidence authority`
- `3209dff fix(agent): anchor robustness replay and claims`
- `0b26e59 refactor(agent): keep scientific routing agent-owned`
- `cef023c fix(agent): fail closed on unverifiable model and figure data`

These commits do not include the concurrent Web/Copilot worktree changes and were not pushed by this session.

## Main fixes

- Latest successful step records and producer-scoped, digest-verified evidence now govern resume, reporting, repair credit, publication artifacts, O22, O23, and O24. Stale partial manifests, inactive producers, symlinks, path escape, digest drift, and first-write aliases fail closed.
- Rendering repairs require an orchestrator-owned binding to an explicit figure-only target and current source evidence. A renderer cannot satisfy a model/table step or replace a mixed scientific contract.
- Robustness variants require the evidence-anchored plan-time lock. Panel rows must be lock members and must match their digest-bound summaries. Exact cohort replay rejects model `n` outside replayed membership and records cohort/membership digests.
- Execution routing uses exact method heads, typed artifact kinds, closed products, and direct-parent evidence. Typed tables whose names contain `figure`, cluster-robust association prose, negated clustering/survival language, and secondary/supporting outputs cannot seize primary ownership.
- FigureSource joins must verify real value columns. Explicit semantic declarations bind their named parent value and cannot borrow a sibling same-family vector. Primary-model contracts require current planner-owned model rosters and verified fit/penalty evidence.
- Automatic structural figure repair is limited to the two evidence-closed, science-neutral renderer families currently proven safe; other legacy renderers remain directly testable but are not automatically authorized.

## Verification

Completed green batches after the fixes included:

- `267 passed`: FigureSource, F1/F2/meta, primary-model contract, and validator suites.
- `79 passed`: deterministic robustness, robustness panel, estimator adapter, and anti-pipeline robustness.
- `133 passed, 2 deselected`: runtime authority, resume, stale-summary, repair credit, artifact authority, preflight ownership, and direct multiple-testing tests. The two deselected tests are the known full mock-pipeline variants that rebuild per-step Matplotlib caches; their direct O22 authority paths were covered in this batch.
- `101 passed`: routing registries, ordinal/cohort routing, method suite, deterministic ownership, study design, and focused pipeline contracts.
- `70 passed`: association/cohort-flow/ordered figure rescue plus report artifact/repair authority.
- Final post-patch gate: `105 passed` for O23/O24 current evidence, primary-estimand preservation, FigureSource, trajectory clustering, and meta-generalization.
- `ruff check`, `python -m compileall`, and `git diff --check` all passed for changed research-agent files.

The required review command (`test_validators_figure_source_trace.py`, `test_trajectory_clustering_routing.py`, `test_meta_benchmark_spec.py`) remained green; the final combined gate includes those suites.

## Benchmark status and next action

This was an architecture repair and validation pass, not a new paper result. Figure 2 therefore remains `6/9`, and Fresh E3 remains `9/12`. No manuscript number was hand-computed and no real Step 06/07 result was promoted.

The next action is unchanged: when the local OpenAI-compatible endpoint can complete requests again, run the full EasyICU `aware` workflow from `06_secondary_adjusted_association`, stop after the step for contract review, then proceed to Step 07 and its figure only if Step 06 is reportable. Do not run the historical `naive` arm.
