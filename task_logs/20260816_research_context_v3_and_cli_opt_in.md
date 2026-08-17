# ResearchContext V3 compatibility and CLI AI opt-in closure

Date: 2026-08-16
Task: `FIG2-DEV9-HELDOUT27` review remediation
Branch: `feat/figure2-dev9-heldout27-20260815`
Base before this patch: `50a9b11`

## Authority boundary

This is product-contract and entrypoint remediation. It creates no scientific
result, changes no sealed development run, and has no Figure 2 or publication
authority.

## Confirmed defects

1. The mounted archived H1 `easyicu.research_context/2` artifact failed in
   `ResearchContextV2.model_validate` because the V2 closure validator had
   later started requiring `analysis_window_role=outer_observation_window`.
   The exact corpus-backed regression failed before the patch and passes after
   it.
2. `easyicu-research-agent --llm openai` constructed the provider client
   without first calling the canonical
   `easyicu.ai_optin.check_external_llm_opt_in` policy. The sibling paper-aware
   replication CLI had the same public-entrypoint gap.

## Remediation

- `ResearchContextV2` is now an archived model with its original descriptor
  closure field set. Parsing a V2 document preserves its version and values;
  it does not rewrite immutable evidence.
- `ResearchContextV3` is the current typed contract. It binds the
  materialization-window role to the sealed column binding, and the context
  builder writes only `easyicu.research_context/3` for new typed contexts.
- `migrate_research_context_v2` provides an explicit deterministic upgrade.
  It fills a missing role from the sealed binding but rejects an explicit
  conflicting role instead of overwriting it.
- Typed prompt, raw-input-contract, and scoped-context revalidation preserve
  the concrete schema version, so V2 replay and V3 fresh runs do not silently
  cross versions.
- Both public research-agent CLIs now require
  `--external-llm-opt-in` with `--llm openai` and call the canonical gate
  before provider construction or credential lookup. Offline mock execution
  remains exempt.
- The module README now describes the runtime as role-scoped LLM calls in a
  sequential `plan -> execute -> verify` state machine, not independently
  negotiating autonomous agents.

## Verification

- Corpus-backed regression plus the focused ResearchContext, typed input,
  temporal, endpoint, resume, CLI, and provider-boundary matrix:
  `668 passed`.
- Package direction, module graph behavior, static architecture policy, and
  repository hygiene tests: `27 passed`.
- Changed-file Ruff check: passed.
- `git diff --check`: passed.

The architecture ratchet command still reports pre-existing branch drift in
`pipeline.py`, `agents/planner.py`, and `schema.py`; the module-inventory diff
also sees the five Progressive Planner modules already introduced before this
patch. No ratchet baseline was refreshed and none of those files is changed by
this remediation.

## Still open by design

- The 2,919-line `_execute_one_step` remains an acknowledged characterized
  refactor. It must be split in a separate owner-focused commit, not mixed into
  this compatibility/security hotfix.
- Slow/integration test classification needs measured file/test ownership and
  an explicit CI split before default selection can change. Adding
  `-m 'not slow'` now would silently weaken existing CI coverage, so this patch
  does not claim that recommendation is closed.
