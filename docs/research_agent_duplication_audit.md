# Research-agent duplication and necessity audit

Measured 2026-08-14 after the P3 decomposition batches and helper
consolidation. Inventory metrics count top-level functions and classes;
duplicate-helper enforcement scans functions, methods, and nested functions in
`src/easyicu/research_agent/**/*.py`. This is not permission to delete a module
based on static in-degree alone.

## Current baseline

| metric | value |
| --- | ---: |
| Python modules | 519 |
| top-level definitions | 5,232 |
| names defined in more than one module | 129 |
| cyclic SCCs | 0 |
| local `_sha256_file` definitions | 0 (16 consolidated) |
| grandfathered local `_finite` definitions | 16 |

## Is every file necessary?

Not every file is a production capability, but every known zero-in-degree file
has a governance route:

1. Console entry points (`cli.py`, `replication_cli.py`) are reached through
   `pyproject.toml`, not package imports.
2. Public package/plugin surfaces may have external consumers and therefore
   cannot be deleted from an internal import graph alone.
3. `docs/research_agent_capability_inventory.md` classifies apparently
   unreachable modules as `production_reachable`, `experimental`, or
   `disabled`, with an owner, activation condition, test, and review date.
4. `graph.py` is an intentional fail-closed compatibility surface for the
   retired LangGraph dispatcher, with removal targeted at 2.0.
5. Idea-mining, cross-model evaluation, scientific adapters, and source-status
   SDK modules are deliberately experimental; they are not advertised as
   stable production capability.

Conclusion: there is no evidence for a bulk deletion pass. Delete a file only
when its public/CLI/plugin reachability is checked and a positive replacement
owner exists. The three-state capability inventory remains the authority.

## Duplicate-helper classification

### A. Same operation, safe central owner

| helper | definitions / distinct bodies before action | decision |
| --- | --- | --- |
| `_sha256_file` | 16 / 4 | **Consolidated now** into `canonical_json.sha256_file`; local definitions prohibited by CI. Four variants differed only by whole-file vs streamed read, `Path()` coercion, and configurable chunk size. The owner accepts `Path | str`, streams in 1 MiB chunks, and preserves the configurable argument. |
| `_canonical_sha256` | 7 / 4 | Candidate for `canonical_json.canonical_sha256`, but each caller's pre-normalization must be reviewed first. Do not replace mechanically. |
| `_atomic_write_text` | 3 / 3 | Create one filesystem-atomic owner after comparing fsync/replace/permission semantics. |
| `_figure_product` | 6 / 3 | Runner templates should use one typed product constructor after output metadata differences are modeled explicitly. |

### B. Same name, divergent semantics (highest risk)

| helper | definitions / distinct bodies | risk / next action |
| --- | --- | --- |
| `_finite` | 16 / 9 | Different accepted input types, scalar coercion, and null/error behavior. Frozen by CI at every function scope: no new copies. Next batch must define a typed finite-number policy in `scalar_utils.py`, migrate one semantic family at a time, and shrink the allowlist. |
| `_method_head` | 17 / 9 | Some normalize aliases, some only split tokens, some depend on plan vocabulary. Route through `research_context.prompt_scope.normalised_method_head` only after call-site semantics are characterized. |
| `_canonical_bytes` | 7 / 7 | The name hides seven schema-specific normalizers. Keep schema normalization local; only the final byte serialization should use `canonical_json_bytes`. |
| `_call_name` | 4 / 3 | AST helpers differ on attributes/subscripts and unknown calls. Merge only after a shared AST-call-name contract and fixtures exist. |
| `_subscript_key` | 4 / 3 | Different Python-version and literal-key behavior. Consolidate with AST compatibility tests, not by name. |

### C. Parallel domain owners, not proven duplication

`intake/materialized_metadata.py` and `intake/materialized_trajectory.py` share
13 top-level names but **zero identical bodies**. They encode related but
different authority envelopes. This is not safe copy-paste cleanup: first
extract a common envelope schema and characterize trajectory-specific identity,
row-count, and time-axis behavior. Until then, the shared names are a drift
risk, not deletion evidence.

## Are we reinventing external libraries?

| area | judgment |
| --- | --- |
| EvidenceStore, typed receipts, plan/run authority, fail-closed gates | Deliberate domain infrastructure. General agent frameworks do not provide these ICU/scientific authority contracts; keep in-house. |
| Workflow engine | Deliberate. LangGraph was retired because phase handoffs were not checkpoint-serializable; `graph.py` refuses dual dispatch. Do not reintroduce a framework without proving exact durable handoffs. |
| Provider retry, budget, call receipts | Domain extension around provider SDKs. Keep the authority/budget layer; delegate HTTP transport to `openai`/`httpx` where already done. |
| Canonical JSON, file hashing, atomic writes, scalar coercion | Commodity infrastructure. Centralize aggressively instead of per-runner copies. |
| Statistical kernels | Prefer SciPy/statsmodels/scikit-learn for fitting and distribution functions. Keep reviewed wrappers, typed contracts, leakage checks, and external-oracle comparisons. A local kernel is justified only when the dependency lacks the exact estimator or reproducibility contract. |
| Figure renderers | Domain-specific publication contracts justify local code, but shared typed product/source-data builders should replace copied runner helpers. |

## Enforcement

`tools/audit_repository_hygiene.py` reads
`tools/arch_baselines/research_agent_duplicate_helpers.json`:

- `_sha256_file`: empty allowlist; any new local definition fails CI.
- `_finite`: exact current file/count upper bounds; deletion is allowed, but a
  new file or second local definition fails CI.

The baseline is an upper bound, not a target. It may shrink without a refresh;
growth requires an explicit reviewed baseline change and should normally be
rejected.

## Recommended order

1. Run E1 fresh Web acceptance before more god-unit refactoring.
2. Migrate `_finite` by semantic family into `scalar_utils.py`; shrink the
   allowlist after each tested batch.
3. Consolidate `_method_head` callers that already match
   `normalised_method_head` exactly; leave divergent planner contracts local.
4. Model one typed figure-product builder and migrate deterministic figure
   runners in small batches.
5. Revisit `run_execute_phase` / `ResearchAgentPipeline` only with
   characterization tests around each extracted state transition/method group.
