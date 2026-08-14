# 2026-08-14 — agents/core.py decomposition batch (P3 structure debt, batch 1)

> Update 2026-08-14 (later batch): the three pre-existing failures below were
> fixed in a follow-up commit — see "Post-batch fixes" at the end.

## Scope

One owner, one batch: the 4,933-line `research_agent/agents/core.py` monolith
(11 agent classes + prompt machinery + parsing helpers in one module) was
split into sibling owner modules behind a thin compatibility facade. No
behavior change; function/class bodies moved byte-for-byte (line-sliced by
verified section banners, cross-imports computed by AST, then pruned by
`ruff --fix F401`).

## New layout

| module | LOC | owns |
| --- | ---: | --- |
| `agents/_support.py` | 744 | prompt-pack handles, shared constants, parsing helpers |
| `agents/planner.py` | 1748 | `PlannerAgent` + planner prompt/contract machinery |
| `agents/replanner.py` | 441 | `ReplannerAgent` + context-budget guards |
| `agents/roles.py` | 694 | 7 ICU worker agents + `RuntimeSupervisor` |
| `agents/coder.py` | 1671 | `CoderAgent` + authority contracts + patch machinery |
| `agents/reporting.py` | 462 | `AnalyzerAgent` + `WriterAgent` |
| `agents/core.py` | ~300 | pure re-export facade (historical import surface) |

Dependency order (acyclic, verified): `_support ← planner ← replanner`,
`_support ← {roles, coder}`, `roles → reporting → coder`.

## Compatibility surface preserved

- `agents.X is agents.core.X is easyicu.research_agent.X` identity contract
  (`test_agent_package_boundary.py` unchanged and passing).
- All historical private-name imports from `agents.core` (e.g.
  `_CODER_GUIDE`, `_build_planner_user_prompt`, `_repair_specialization`,
  `_normalise_plan_payload` identity vs `plan_payload`) re-exported by the
  facade; verified by an AST scan of every `from …agents.core import`
  consumer in src/tests/benchmarks/tools/scripts (197 imports, 0 missing).
- `agents` package stays lazy (facade only loads when requested).
- `pyproject.toml` ruff per-file-ignores: `agents/core.py` F401 entry kept
  (facade); new files are ruff-clean without pins. One dead local
  (`outputs` in the coder product-boundary guard) deleted — it was masked by
  the old core.py F841 pin.

## Verification

- Focused suites: 887 passed, 2 failed —
  `test_outbound_diagnostic_envelope.py::test_replanner_probe_projection_rejects_all_identifier_suffixes`
  and
  `test_deferred_llm_audit_and_runtime_repair.py::test_automatic_contract_repair_does_not_consume_llm_contract_allowance`.
  Both fail identically on the pre-split baseline (git-stash control run),
  i.e. pre-existing on HEAD `223d45c`, not introduced here.
- `python -m ruff check src/easyicu/research_agent/agents/` clean.
- `tools/arch_measure.py`: six new owners appended to `TARGET_FILES`;
  `--diff` OK (core.py 4,933 → ~300, improvement only); baseline re-emitted
  with reason. Other-file deltas absorbed by the emit are committed
  improvements from `26bdcdc..223d45c` (all lower-is-better direction).
- `tools/research_agent_resource_baseline.py`: re-emitted with reason
  (core.py digest changed; planner envelope maxima decreased 61,286→59,785).
- `tools/research_agent_module_graph.py`: `cyclic_scc_count` still 0;
  new modules all reachable via `agents.core`.

## Follow-ups (not this batch)

- `audits/validators.py` (13.8K), `pipeline.py` (11.3K), `execution/phase.py`
  (7.5K), `gates/preflight.py` (7.0K) remain the next P3 batches — each needs
  its own owner contract + characterization tests; NOT authorized by this
  batch's baseline refresh.
- The two pre-existing test failures above belong to their owning lanes.
- `tests/test_repository_hygiene.py` also fails on this HEAD for unowned
  top-level `literature_concepts.py` / `literature_excerpt.py` — verified
  pre-existing via stash control, unrelated to this batch.

## Post-batch fixes (same day, separate commit)

1. **Hygiene**: `literature_concepts.py` / `literature_excerpt.py` registered
   in `tools/arch_baselines/research_agent_top_level_ownership.json` under
   `shared_contract` (introduced by 2ef77e7/0d59ad2 without manifest rows).
2. **Replanner crash on plans without a canonical analysis_type** (surface
   exposed by 2ef77e7's action-guide call): `agents/replanner.py` now omits
   the action guide when `canonical_analysis_family(current_plan.analysis_type)`
   is None instead of raising — a replan must not invent a family the locked
   plan never declared. Synthetic-plan fixture updated to carry the new
   `measurement_audit_spec` contract (`missingness_profile`).
3. **Stale monkeypatch seam**: `test_deferred_llm_audit_...` now patches
   `execution.candidate_loop.deterministic_contract_repair`; the loop moved
   out of `execution.phase` in 1e5182a and the test kept patching the old
   owner, silently skipping the deterministic-repair stage it verifies.

Verification: affected suites 55 passed; combined focused batch 878 passed;
ruff clean.
- Layer taxonomy for authority/gates/audits/repairs/contracts is now
  documented in `docs/research_agent_layer_ownership.md`.
