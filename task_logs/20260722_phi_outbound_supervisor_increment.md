# PHI / outbound hardening supervisor increment

Date: 2026-07-22
Branch: `codex/phi-outbound-hardening-20260722`
Base: `9401b3168d29f22ab657842f3a529bad479803d5`
Status: candidate commit awaiting independent review; not merged or pushed

## Scope

This increment closes only the supervisor-confirmed outbound authority gaps.
It did not call a real Provider, Docker, patient data, or Canonical9.

- Provider trust is identity-bound and deny-by-default. Unknown wrappers and
  custom constructors cannot inherit child trust or self-register as offline.
  Reviewed wrappers bind the exact child graph and reject later mutation.
- OpenAI locality is determined from a parsed HTTP(S) hostname/IP. Userinfo,
  query/fragment injection, non-loopback hosts, wildcard `0.0.0.0`, malformed
  ports, and unapproved paths are rejected. Delivery rechecks the live model
  and endpoint against the factory-minted authorization.
- Table 1 keeps opaque labels in the outbound plan and a digest-bound,
  host-only execution binding. Replan normalization re-establishes that binding
  before execution. Contract repair, runtime repair, minimal patch, full
  rewrite, Concept Auditor, Coder, and Replanner receive only reversible opaque
  code tokens; the host restores private literals after the external response.
- One shared outbound context projection removes observed literals/extrema,
  raw logs, patient-derived prose, and arbitrary artifact text while retaining
  host-authoritative scientific contracts: units, plausibility ranges, outcome
  semantics, inclusion/exclusion, temporal constraints and analysis windows,
  fixed trajectory coordinates, missingness semantics, forbidden transforms,
  and explicit user choices.
- Analyzer receives a structured projection that retains numeric aggregates
  and opaque-tokenizes categorical strings. Tier-2 Jury receives identity plus
  safe structured status, errors, evidence, and counts rather than raw text.
- The closed static mocks used by offline tests cannot execute caller-provided
  callbacks. Counting and provider-budget integration therefore remain usable
  without reopening a generic custom-client authorization seam.

## Negative guarantees and tests

- Unknown/custom/wrapper/endpoint rejection:
  `test_custom_client_cannot_self_register_as_offline`,
  `test_unregistered_top_level_wrapper_cannot_inherit_child_trust`,
  `test_registered_wrapper_rejects_mutated_child_graph_before_delivery`,
  `test_remote_openai_transport_cannot_be_authorized_as_local`, and
  `test_registered_openai_transport_mutation_is_rejected_before_delivery`.
- Malicious URL parsing: `test_loopback_url_classification_is_parsed_and_strict`.
- CountingClient production integration:
  `test_counting_client_remains_a_registered_planner_wrapper`.
- Replan-to-execution Table 1 binding and prompt capture:
  `test_private_table_one_labels_never_enter_agent_prompts`.
- Four repair surfaces:
  `test_table_one_contract_repair_never_sends_private_labels`,
  `test_table_one_runtime_repair_never_sends_private_labels`,
  `test_table_one_minimal_patch_never_sends_private_deterministic_script`, and
  `test_table_one_full_rewrite_never_sends_private_deterministic_script`.
- Shared context / Concept Auditor / Analyzer:
  `test_every_agent_context_uses_the_same_outbound_safe_projection`.
- Tier-2 structured review without raw artifacts:
  `test_jury_receives_artifact_identity_not_arbitrary_artifact_text`.
- Delivery surface scan across `src/`, `tools/`, `scripts/`, and `examples/`,
  including `complete` and `complete_with_images`:
  `test_production_prompt_calls_use_the_authorized_delivery_boundary`.
- Wide trajectory preservation and transport budget:
  `test_wide_trajectory_projection_preserves_every_exact_window_coordinate`
  and `test_wide_trajectory_generation_prompts_stay_below_transport_gate`.

## Verification

- `pytest test_validators.py test_trajectory_prompt_compaction.py`:
  **212 passed** (210 validator + 2 trajectory).
- Supervisor regression matrix (original 317 plus new negative tests):
  **326 passed, 2 skipped, 1 intentionally deselected pre-existing test**.
- `pytest test_provider_budget.py`: **44 passed**.
- `pytest test_execution_identity.py`: **14 passed**.
- Call-scoped usage/concurrency selection: **4 passed**.
- Ruff, changed-file Black check, `py_compile`, and `git diff --check`: pass.
- `tools/arch_measure.py --diff tools/arch_baselines/execution_phase.json`:
  no lower-is-better regression.
- `tools/research_agent_module_graph.py --diff
  tools/arch_baselines/research_agent_module_graph.json`: pass, no new cycle.

The architecture baseline was re-emitted after the required three-line
replan-binding call was added. It also freezes already-present reductions in
`agents/core.py`, `execution/phase.py`, `pipeline.py`, and validators; the
post-emit diff is zero. This is a candidate for final incremental review, not
a claim that PHI/outbound hardening has been merged or paper-authorized.
