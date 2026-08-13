# Research-agent capability inventory

**What this is.** Modules that exist, are tested, and are **not reached from any
production path** in `src/`. They are inventory, not dead code — but "already
written" is not the same as "keep forever". Each row must name an owner, an
activation precondition, and a review date; a row that goes past its review date
without a decision is a defect in this file, not a licence to keep waiting.

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

Measured 2026-08-13 at `e743498`.

## Status vocabulary

| status | meaning |
| --- | --- |
| `optional-by-design` | Documented in `CLAUDE.md` as an optional layer. Never expected on the default path. |
| `awaiting-wiring` | Implemented and tested; blocked on a named decision before an owner can call it. |
| `entry-point` | Reached through `[project.scripts]` or a generated code template, not an import. |
| `support-surface` | Used by tests/tools as a public surface; no production caller is expected. |
| `external-consumer` | Called by production code outside `research_agent`, so package-local in-degree is intentionally zero. |
| `compatibility` | A documented, fail-closed compatibility surface with an explicit removal target. |

## Inventory

| module | LOC | status | owner | activation precondition | tests | review |
| --- | ---: | --- | --- | --- | --- | --- |
| `methods/rmst.py` | 264 | `awaiting-wiring` | methods | Deterministic supporting-executor wiring, priority 3 of 5 (`ph_schoenfeld → delong_auc → rmst → decision_curve → evalue`). Needs exact typed inputs/outputs and a full data contract for `owns_step`; **no method-string → function mapping**. Also has no external authority comparison (survRM2 absent). | 21 | 2026-10-01 |
| `methods/decision_curve.py` | 228 | `awaiting-wiring` | methods | Same wiring decision, priority 4 of 5. No external authority comparison (dcurves absent). | 11 | 2026-10-01 |
| `methods/delong_auc.py` | 285 | `awaiting-wiring` | methods | Same wiring decision, priority 2 of 5. Oracle-checked against `pROC::roc.test`. | 4 | 2026-10-01 |
| `methods/conformal.py` | 165 | `awaiting-wiring` | methods | Same wiring decision; not in the current priority list, so it needs one before it can be scheduled. | 5 | 2026-10-01 |
| `methods/survival_inputs.py` | 219 | `awaiting-wiring` | methods | Consumed by the survival executors once they take typed inputs. | 4 | 2026-10-01 |
| `methods/temporal_features.py` | 306 | `awaiting-wiring` | methods | Trajectory lane; blocked behind the trajectory typed contract. | 9 | 2026-10-01 |
| `evaluation_scorecard.py` | 1628 | `optional-by-design` | evaluation | Tier-2 scoring over completed artifacts. Paper-facing scorer authority lives outside the installed engine, under `benchmarks/`. | 7 | 2026-11-01 |
| `evaluation/tier2_jury.py` | 649 | `optional-by-design` | evaluation | Jury/rubric adapter; opt-in per run. | 8 | 2026-11-01 |
| `evaluation/cross_model_panel.py` | 294 | `optional-by-design` | evaluation | Cross-model concordance; opt-in per run. | 5 | 2026-11-01 |
| `acquisition/foundation.py` | 603 | `external-consumer` | acquisition | Production Web run launcher and `run_discovery_to_manuscript.py` own the pre-sandbox acquisition call; the core Agent must not import backward into its host. | 10 | 2026-11-01 |
| `discovery/idea_mining_source_status.py` | 1144 | `awaiting-wiring` | idea-mining | Idea-mining lane. Needs the differentiated fresh Idea end-to-end run before any of this becomes a production path. | 1 | 2026-10-01 |
| `discovery/idea_mining_extended_feasibility.py` | 680 | `awaiting-wiring` | idea-mining | Same lane. | 5 | 2026-10-01 |
| `discovery/idea_mining_data_first_route.py` | 627 | `awaiting-wiring` | idea-mining | Same lane. | 3 | 2026-10-01 |
| `discovery/concept_proposal.py` | 460 | `awaiting-wiring` | idea-mining | Same lane. | 5 | 2026-10-01 |
| `discovery/idea_mining_longitudinal.py` | 349 | `awaiting-wiring` | idea-mining | Same lane. | 2 | 2026-10-01 |
| `discovery/longitudinal_handoff.py` | 263 | `awaiting-wiring` | idea-mining | Same lane. | 1 | 2026-10-01 |
| `authority/source_status_sdk.py` | 240 | `support-surface` | authority | Public SDK surface for source-status receipts. One test; confirm whether an external consumer is intended, otherwise reclassify. | 1 | 2026-10-01 |
| `reporting/result_card.py` | 281 | `external-consumer` | reporting | Pi Copilot renders this aggregate-only card from outside `research_agent`; keeping the dependency outward prevents reporting from importing Web. | 2 | 2026-11-01 |
| `replication/discovery.py` | 698 | `entry-point` | replication | Lazily exported by `replication.__getattr__` and called by `replication_cli`; direct package imports are intentionally absent. | 3 | — |
| `cli.py` | 426 | `entry-point` | agent | `easyicu-research-agent` in `[project.scripts]`. | 2 | — |
| `script_runtime.py` | 106 | `support-surface` | agent | Runtime shim imported by generated scripts, not by the engine. | 3 | 2026-11-01 |
| `graph.py` | 55 | `compatibility` | orchestration | Retired LangGraph builder fails closed while human-review model imports remain compatible; removal target is 2.0 in `docs/deprecation_policy.md`. | 1 | 2.0 |
| `scientific_adapters/*` | 52 | `support-surface` | scientific-adapters | Third-party adapters loaded through `importlib.import_module` (dowhy / pandera / sksurv); invisible to import scans by construction. | 10 | 2026-11-01 |
| `replication_cli.py` | 228 | `entry-point` | replication | `easyicu-research-replication` in `[project.scripts]`. | 3 | — |
| `case_plugins/` | 26 | `support-surface` | case-plugins | Plugin discovery surface. | 7 | 2026-11-01 |

## Rules

1. A new module that no production path calls must be added here in the same
   change that introduces it, with a review date.
2. At the review date the owner picks one of: **wire it**, **archive it**
   (move out of the installed package), or **delete it**. "Leave it another
   quarter" is a decision that needs a new date and a reason.
3. Do not delete an `awaiting-wiring` row to make a number look better. The
   engineering rule in `CLAUDE.md` stands: zero production references alone
   never justifies deleting a real implementation — a runner, validator or
   authority needs positive evidence it has been superseded.
4. `methods/` kernels wired without exact typed inputs/outputs and a full data
   contract for `owns_step` are a regression even if tests pass. A
   method-string-to-function mapping is not wiring.
