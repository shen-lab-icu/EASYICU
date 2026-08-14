# Research-agent capability inventory

**What this is.** The reachability registry for package-local modules that look
unreachable to the static import graph. It distinguishes code that a public
entry point demonstrably reaches from experimental or disabled code. "Already
written" is not the same as "available in the product". Each production row
must name the exact public-API-to-executor route and one integration test that
traverses it; every other row must stay visibly experimental or disabled.

**What this is not.** It is not a redundancy audit. Zero-inbound-import is only
the *most visible* kind of unused code; it says nothing about unreachable
branches inside wired modules, string- or template-dispatched entry points
(`run_feasibility_protocol` is called from a generated code template and looks
orphaned to any import scan), or paths that exist but are never traversed in a
real run. Do not compute "% of the module that is live" from this file.

**How to regenerate the candidate list**

```bash
python tools/research_agent_module_graph.py > /tmp/graph.json
```

Modules with in-degree 0 that are not `__init__.py` are candidates; then check
each leaf name against `src/easyicu` (excluding `research_agent`), `tests/`,
`tools/` and `pyproject.toml` before calling anything unused. Console-script
entry points (`cli`, `replication_cli`) are reached through
`[project.scripts]`, not imports.

Measured 2026-08-14 at `26bdcdc`.

## Status vocabulary

| status | meaning |
| --- | --- |
| `production_reachable` | A public API reaches the real runtime owner, and the row binds an exact integration test proving that route. |
| `experimental` | Implemented or tested, but not advertised as a stable product capability and not relied on by the default runtime. |
| `disabled` | Intentionally unavailable on production paths; retained only for a named compatibility/removal boundary. |

## Inventory

| module | LOC | status | owner | activation precondition | tests | review |
| --- | ---: | --- | --- | --- | --- | --- |
| `methods/rmst.py` | 264 | `experimental` | methods | Deterministic supporting-executor wiring, priority 3 of 5 (`ph_schoenfeld → delong_auc → rmst → decision_curve → evalue`). Needs exact typed inputs/outputs and a full data contract for `owns_step`; **no method-string → function mapping**. Also has no external authority comparison (survRM2 absent). | 21 | 2026-10-01 |
| `methods/decision_curve.py` | 228 | `experimental` | methods | Same wiring decision, priority 4 of 5. No external authority comparison (dcurves absent). | 11 | 2026-10-01 |
| `methods/delong_auc.py` | 285 | `experimental` | methods | Same wiring decision, priority 2 of 5. Oracle-checked against `pROC::roc.test`, but not runtime wired. | 4 | 2026-10-01 |
| `methods/conformal.py` | 165 | `experimental` | methods | Same wiring decision; not in the current priority list, so it needs one before it can be scheduled. | 5 | 2026-10-01 |
| `methods/survival_inputs.py` | 219 | `experimental` | methods | Candidate typed-input owner; no production executor currently consumes this module. | 4 | 2026-10-01 |
| `methods/temporal_features.py` | 306 | `experimental` | methods | Trajectory lane; blocked behind the trajectory typed contract. | 9 | 2026-10-01 |
| `evaluation_scorecard.py` | 1628 | `experimental` | evaluation | Optional Tier-2 scoring over completed artifacts. Paper-facing scorer authority lives outside the installed engine, under `benchmarks/`. | 7 | 2026-11-01 |
| `evaluation/tier2_jury.py` | 649 | `experimental` | evaluation | Optional jury/rubric adapter; not enabled on the default production path. | 8 | 2026-11-01 |
| `evaluation/cross_model_panel.py` | 294 | `experimental` | evaluation | Optional cross-model concordance; not enabled on the default production path. | 5 | 2026-11-01 |
| `acquisition/foundation.py` | 603 | `production_reachable` | acquisition | Pi `easyicu_run` → Web `make_research_pipeline_run_runner` → `acquire_universe_for_question` → `ResearchAgentPipeline.run`; the host owns pre-sandbox acquisition. | `tests/test_pi_copilot_research_workflow.py::test_web_runner_delegates_to_research_agent_pipeline` | 2026-11-01 |
| `discovery/idea_mining_source_status.py` | 1144 | `experimental` | idea-mining | Idea-mining lane. Needs the differentiated fresh Idea end-to-end run before this is advertised as a stable product capability. | 1 | 2026-10-01 |
| `discovery/idea_mining_extended_feasibility.py` | 680 | `experimental` | idea-mining | Same lane. | 5 | 2026-10-01 |
| `discovery/idea_mining_data_first_route.py` | 627 | `experimental` | idea-mining | Same lane. | 3 | 2026-10-01 |
| `discovery/concept_proposal.py` | 460 | `experimental` | idea-mining | Same lane. | 5 | 2026-10-01 |
| `discovery/idea_mining_longitudinal.py` | 349 | `experimental` | idea-mining | Same lane. | 2 | 2026-10-01 |
| `discovery/longitudinal_handoff.py` | 263 | `experimental` | idea-mining | Same lane. | 1 | 2026-10-01 |
| `authority/source_status_sdk.py` | 240 | `experimental` | authority | Public SDK candidate for source-status receipts; no production consumer is currently registered. | 1 | 2026-10-01 |
| `reporting/result_card.py` | 281 | `production_reachable` | reporting | Completed pipeline artifacts → Pi result projection → `build_result_interpretation_card`; the card remains aggregate-only and cannot invent numbers. | `tests/test_pi_copilot_research_workflow.py::test_result_interpretation_card_reuses_agent_claims_without_new_numbers` | 2026-11-01 |
| `replication/discovery.py` | 698 | `production_reachable` | replication | Public `discover_easyicu_exports` → native/legacy ExportPackage discovery → verified package result. | `tests/research_agent/test_export_package_integration.py::test_replication_discovery_accepts_native_and_legacy_packages` | 2026-11-01 |
| `cli.py` | 426 | `production_reachable` | agent | `[project.scripts] easyicu-research-agent` → `cli.main` → `ResearchAgentPipeline.run`. | `tests/research_agent/test_cli.py::test_public_cli_reaches_the_research_agent_pipeline` | 2026-11-01 |
| `script_runtime.py` | 106 | `production_reachable` | agent | CoderAgent prompt → generated script import → `script_runtime.write_json` → registered output artifact. | `tests/research_agent/test_script_io_replaces_the_pasted_helper.py::test_the_imported_writer_emits_null_for_every_non_finite_number` | 2026-11-01 |
| `graph.py` | 55 | `disabled` | orchestration | Retired LangGraph builder fails closed; removal target is 2.0 in `docs/deprecation_policy.md`. | 1 | 2.0 |
| `scientific_adapters/*` | 52 | `experimental` | scientific-adapters | Optional dowhy / pandera / sksurv adapters are dynamically probed but may not upgrade current scientific capabilities. | 10 | 2026-11-01 |
| `replication_cli.py` | 228 | `production_reachable` | replication | `[project.scripts] easyicu-research-replication` → `replication_cli.main` → `ResearchAgentPipeline.reproduce_paper`. | `tests/research_agent/test_replication_cli.py::test_replication_cli_paper_mode_dispatches_to_pipeline` | 2026-11-01 |
| `case_plugins/` | 26 | `experimental` | case-plugins | Plugin discovery surface; default production registry remains empty until an explicit plugin activation contract is supplied. | 7 | 2026-11-01 |

## Rules

1. A new module that no production path calls must be added here in the same
   change that introduces it, with a review date.
2. At the review date the owner picks one of: **make it production-reachable**
   with a typed route and integration test, **keep it experimental** with a new
   reason/date, **disable it**, archive it, or delete it.
3. Do not delete an `experimental` row to make a number look better. The
   engineering rule in `CLAUDE.md` stands: zero production references alone
   never justifies deleting a real implementation — a runner, validator or
   authority needs positive evidence it has been superseded.
4. `methods/` kernels wired without exact typed inputs/outputs and a full data
   contract for `owns_step` are a regression even if tests pass. A
   method-string-to-function mapping is not wiring.
5. A `production_reachable` row without an exact
   `tests/...py::test_name` reachability test is invalid and fails the inventory
   audit. Unit tests of the isolated kernel are not sufficient.
