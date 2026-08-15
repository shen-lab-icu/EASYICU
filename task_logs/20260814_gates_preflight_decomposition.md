# 2026-08-14 — gates/preflight.py decomposition batch (P3 structure debt, batch 5)

## Scope

One owner, one batch: two cohesive helper families moved out of
`research_agent/gates/preflight.py` into sibling owner modules behind the
preflight facade. `audit_mechanical_code_contracts` and every
seam-entangled checker stay in preflight.py.

## Split

- `gates/preflight_statics.py` (478 lines): scope/assignment name analysis
  (`_scope_nodes`, `_assignment_target_names`, `_names_bound_in_scope`,
  `unresolvable_names`, `_numeric_coercion_sites`, guard/loss-binding
  machinery — the coercion-loss analyzer cluster).
- `gates/preflight_provenance.py` (357 lines): provenance predicate meaning,
  branch-flow outcomes, result-sink ordering, resolved-context payload
  findings, resolved-input binding keys.
- `preflight.py` (6,268 lines): the entry point, the patched seam
  (`_SIGNATURE_DERIVED_HOST_HELPERS`), and 22 stayers the fixpoint ejected
  because they reference non-family names (branch-local unbound analysis,
  provenance fail-closed, typed-dataframe erasure, reconciliation
  swallowing, call-signature findings — these are deeply entangled with the
  vocabularies and each other; splitting them needs a dedicated batch with
  characterization tests).

6,982 → 6,268 loc. Cross-imports between the two new modules were resolved
to a single direction (statics ← provenance); no cycle.

## Splitter bug found and fixed (record for the next batch)

The first emit wrote `head + facades + stayers + tail`, which silently
**dropped the non-family source between two family ranges** (L1232–1573:
`_pre312_fstring_subscript_quote_findings`, `_resolved_input_binding_key_findings`,
`_RECONCILIATION_HELPER_NAME` and friends). Caught by ruff F821 in the
rewritten preflight.py; fixed by rebuilding the middle as an ordered slice
that preserves everything except the moved spans, then verified with an
AST top-level-name diff (original vs merged modules: lost defs = NONE).

## Verification

- Preflight consumer suites (boolean reduction, fstring compat, typed-input
  precedence, suffix slice, cohort summary, coder context repair, host
  helper contracts ×2, interval method, coder output scope, refusal naming,
  binding-lookup): 570 passed.
- Coder-adjacent suites (prompt budget, agentic coder, failed-attempt
  release): 107 passed.
- End-to-end pipeline mock smoke (6 tests): passed.
- Module graph zero SCC; new modules have no edge back to preflight.py.
- Arch ratchet: both new owners appended to TARGET_FILES; baseline
  re-emitted with reason (no growth).
