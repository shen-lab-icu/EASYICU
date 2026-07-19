# Research-agent Module Inventory and Retirement Audit

Baseline: `refactor/agent-control-plane@c2df928`; current architecture measured
at `c411ed0`/`00c962d` after the retirement and dependency-inversion patches.

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

The first safe retirement removed one never-wired implementation and its test,
plus one unused function/import. The current tree therefore has 160 top-level
files and 92 real top-level implementations (approximately 97.9k LOC). After
the planning-contract, provider-protocol, replication-metric, and execute-host
service cuts, the module graph reports 295 modules, 20 packages, 822 edges, and
**zero cyclic modules / zero SCCs**. The former 23-module control-plane SCC,
validator/replication pair, and final pipeline/execute/publication-figure cycle
are gone.

This is the AST-visible, module-top-level static import graph. Sealed-renderer
digest and archived-candidate compatibility still use a controlled registry-
mediated dynamic import of legacy implementation modules; that runtime
compatibility surface is not an import-time SCC and is not classified as dead.

The visible file count is therefore partly intentional compatibility surface,
but the remaining 92 real top-level implementations and the two ~11k-line
pipeline modules show that the architecture is not yet fully organized.

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

## Compatibility aliases that must remain

The 65 exact aliases are deliberately locked by module-identity and graph tests.
They cover moved gate/execution/authority, runner, discovery, reporting, repair,
planning, context, evaluation, and review modules. They let archived scripts,
third-party imports, and monkeypatches through either old or canonical paths
reach the same module object.

New production code must use the responsibility package; the alias is not the
canonical implementation. Removing aliases requires an explicit public API and
archive-retirement policy, not a cosmetic cleanup commit.

The three non-exact archive shims (`figure_contract.py`,
`temporal_features.py`, and `step_summary.py`) also remain. Likewise,
`legacy_code_migrations.py` is live through `repairs/source.py` and is not dead
code.

## Parked deterministic primary runners

The following canonical implementations live under `execution/runners/`:

- `deterministic_causal.py`
- `deterministic_clustering.py`
- `deterministic_cohort_flow.py`
- `deterministic_ordinal.py`
- `deterministic_sensitivity.py`
- `deterministic_survival.py`

They currently own no production step: both primary-runner sets are empty and
the capability registry declares `llm_coded` with `primary_runner=None`.
However, direct-import compatibility tests/public paths remain. They are marked
**parked/deprecated**, not deleted. Their headers must no longer imply that they
currently own primary science.

The live auxiliary deterministic paths are descriptive, missingness,
robustness, and trajectory-stability execution. They compute or render an
Agent-locked specification and must not choose exposure, outcome, cohort,
method, or estimand.

## Live optional components (not dead)

- `memory.py` and `experience.py`: opt-in pipeline capability; canonical
  submission profiles disable them, but non-canonical consumers remain.
- `graph.py`: opt-in graph execution path.
- `cli.py` and `replication_cli.py`: installed console entry points.
- `data_foundation.py`: consumed by `run_discovery_to_manuscript.py`.
- `evaluation_scorecard.py`, `validity_signals.py`, and `icu_agent_bench.py`:
  live benchmark/evaluator dependencies.

## Remaining design problems

1. **Paper semantics in installed core.** `evaluation_scorecard.py`,
   `validity_signals.py`, and `icu_agent_bench.py` still participate in the
   current Figure 2 evaluator/scorer authority. They cannot be moved as an
   ordinary refactor: use additive evaluator v3, preserve v1/v2 bytes and
   digests, then leave compatibility shims.
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
2. Add explicit deprecation metadata/tests for parked primary runners; remove
   them only under an archive/public API retirement version.
3. Design additive evaluator v3 before moving scorer-bound files.
4. Audit display labels and mock/learning examples as protocol changes, not
   mechanical file moves.
5. Run release archive, wheel old/new import smoke, module graph, meta/capability,
   resume/evidence authority, and full segmented regression before freezing the
   experiment engine.
