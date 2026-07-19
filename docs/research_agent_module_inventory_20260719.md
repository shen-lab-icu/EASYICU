# Research-agent Module Inventory and Retirement Audit

Baseline: `refactor/agent-control-plane@c2df928`; current architecture is the
pre-v1 breaking-cleanup line after the responsibility-package and authority
relocation bundles.  Archived diagnostic runs will be rerun from fresh inputs;
they no longer constrain import-path compatibility.

This inventory explains why `src/easyicu/research_agent/` still appears flat
after responsibility packages were introduced, and distinguishes safe cleanup
from compatibility or scientific-authority boundaries.

## Quantitative result

Before the retirement patch, the 161 top-level Python files comprised:

| Class | Files | Lines | Meaning |
|---|---:|---:|---|
| Exact module-object compatibility aliases | 65 | 694 | Old public/archive import paths pointing to canonical package modules |
| Non-exact archive compatibility shims | 3 | 65 | `figure_contract.py`, `temporal_features.py`, `step_summary.py` |
| Real top-level implementations | 93 | 98,530 | Canonical/public/frozen/dormant implementations requiring individual review |

The cleanup has now removed the old facade layer rather than retaining hundreds
of tiny forwarding files.  The current tree has **37 top-level Python files
including `__init__.py`**, with 226 modules in 27 responsibility packages and
758 static import edges.  The graph remains at **zero cyclic
modules / zero SCCs**.  The former 23-module control-plane SCC,
validator/replication pair, and final pipeline/execute/publication-figure cycle
are gone.

The execution-runtime bundle moved `runner.py`, `code_hygiene.py`,
`run_coordination.py`, `step_execution.py`, and `step_worker_state.py` into
`execution/` and retired their old root paths.  Runtime boundary types now live
in dependency-neutral `contracts/runtime.py`; declared-product, ordered-group
and robustness execution contracts share the same package, while fail-closed
finding semantics live under `gates/semantics.py`.  Authority code therefore
does not reverse-import the execution layer.

The trajectory, publication-figure, cohort and acquisition bundles now own
their domain implementations under `trajectory/`, `figures/`, `cohort/` and
`acquisition/`. Their former top-level import paths are deleted. Generic cohort
dataframe primitives now live in `cohort/primitives.py`, so the materializer no
longer imports private helpers from a case plugin. The acquisition API requires
the caller's outcome declaration and carries no default death/censoring science.

Analysis blueprints now live with planning, runtime method capabilities with
execution, and temporal/numeric context construction with `research_context/`.
Their former root paths are retired; current code and tests import only the
responsibility-owned modules.

This is the AST-visible, module-top-level static import graph. Sealed-renderer
digest selection uses a controlled registry-mediated dynamic import of current
implementation modules; that runtime surface is not an import-time SCC.

The remaining visible file count is now implementation work, not compatibility
surface.  The two ~11k-line pipeline modules and the remaining top-level domain
modules show that the architecture is not yet fully organized.

## Retirement decision method

A file is deletable only if all six authority channels are empty:

1. static production/tool/script imports and calls;
2. dynamic `getattr`/`importlib`/string dispatch;
3. capability registry or runner dispatch ownership;
4. root/public API and packaging entry points;
5. tests that represent supported compatibility rather than a private unit;
6. sealed or archived local run scripts.

Absence from local archives is not proof that no external user imports a public
path, so public or documented paths require a deprecation/major-version policy
even when current production dispatch is empty.

## Retired in this audit

- `easyicu.research_agent.projection`: a source-data projection scaffold with
  no production, tool, dynamic, registry, root-export, or sealed-run consumer.
  Its only consumer was its own unit test. The implementation and test were
  deleted together.
- `plan_utils._question_primary_predictor_is_vasopressor_or_unknown`: zero
  callers; its unused `pipeline.py` import was removed.

These removals do not change the active figure provenance gate or Agent-owned
scientific choices.

## Retired compatibility surface

The pre-v1 alias files and the three non-exact archive shims
(`figure_contract.py`, `temporal_features.py`, and `step_summary.py`) have been
deleted.  Current code and tests import responsibility packages directly.
`legacy_code_migrations.py` and its exact publication-helper rewrite were also
deleted; current invalid helper introspection still fails closed and is routed
through the normal typed repair path.  The pre-cleanup state remains available
from the archive tag rather than from runtime shims.

## Retired deterministic primary runners

The following unowned implementations were deleted from `execution/runners/`:

- `deterministic_causal.py`
- `deterministic_clustering.py`
- `deterministic_cohort_flow.py`
- `deterministic_ordinal.py`
- `deterministic_sensitivity.py`
- `deterministic_survival.py`

They owned no production step: both primary-runner sets were empty and the
capability registry declared `llm_coded` with `primary_runner=None`.  Their
implementation-only tests and old import paths were retired with them.  This
removes 3,486 lines without changing any live analysis route.

The live auxiliary deterministic paths are descriptive, missingness,
robustness, and trajectory-stability execution. They compute or render an
Agent-locked specification and must not choose exposure, outcome, cohort,
method, or estimand.

## Live optional components (not dead)

- `memory.py` and `experience.py`: opt-in pipeline capability; canonical
  submission profiles disable them, but non-canonical consumers remain.
- `graph.py`: opt-in graph execution path.
- `cli.py` and `replication_cli.py`: installed console entry points.
- `acquisition/foundation.py`: consumed by `run_discovery_to_manuscript.py`.
- `evaluation_scorecard.py`, `validity_signals.py`, and `icu_agent_bench.py`:
  live benchmark/evaluator dependencies.

## Remaining design problems

1. **Paper semantics in installed core.** `evaluation_scorecard.py`,
   `validity_signals.py`, and `icu_agent_bench.py` still participate in the
   current Figure 2 evaluator/scorer authority.  The additive evaluator v3 now
   binds the relocated authority modules while preserving the frozen v1/v2
   protocol bytes; later scorer-core moves require another explicit authority
   version rather than an invisible path change.
2. **Case/showcase leaves.** `case_contexts.py` and
   `easyicu_case_builder.py` encode lactate/MAP/vasopressor showcase material.
   Move these toward examples/replication only after consumer and archive
   compatibility is explicit.
3. **Display-contract leakage.** `pipeline.py` still contains publication and
   sensitivity labels/semantic aliases with sepsis, lactate, or KDIGO language.
   This is a real shared-engine hygiene issue, not just a file-layout issue; a
   fix changes visible figure contracts and therefore needs dedicated review.
4. **Mock/learning case heuristics.** `llm_mocks.py`, `memory.py`, and
   `experience.py` contain KDIGO/SOFA/sepsis examples or heuristics. Canonical
   profiles do not consume them, but they should eventually be isolated as
   mock/example or learning-policy data instead of growing in shared logic.
5. **Large orchestration surfaces.** `pipeline.py` and `pipeline_execute.py`
   remain roughly 11k lines each, and the package root exports hundreds of
   symbols. Responsibility packages are real progress, but they do not by
   themselves complete orchestration/API reduction.

## Next retirement gates

1. Keep the import graph acyclic: execute/publication consumers receive fresh,
   immutable host-service snapshots and may not reverse-import `pipeline`.
2. Keep deleted primary runners absent and keep the registry's primary science
   owner set empty.
3. Re-authorize scorer-bound moves additively; never rewrite frozen v1/v2
   evaluator bytes.
4. Audit display labels and mock/learning examples as protocol changes, not
   mechanical file moves.
5. Run release archive, wheel old/new import smoke, module graph, meta/capability,
   resume/evidence authority, and full segmented regression before freezing the
   experiment engine.
