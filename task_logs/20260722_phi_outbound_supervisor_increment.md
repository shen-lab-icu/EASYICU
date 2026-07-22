# PHI / outbound hardening supervisor increment

Date: 2026-07-22
Branch: `codex/phi-outbound-hardening-20260722`
Original base: `9401b3168d29f22ab657842f3a529bad479803d5`
Current blocker-fix base: `e24d634d635da8efd99d49ea55a02ea2ca5688ba`
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
- Provider trust records bind the exact constructor, concrete client type,
  `complete`, optional `complete_with_images`, and wrapper child graph. An
  instance method override, later class-method mutation, or an exact-type
  `object.__new__` pseudo-OpenAI object is rejected before its callback runs.
- Step summaries now use an explicit host-owned aggregate field schema. Numeric
  values are not safe merely because they are numeric: patient/stay identifiers,
  individual ages/labs, arrays, and unknown nested values are dropped. Tier-2
  parses structured content only from its registered process-manifest filenames;
  arbitrary JSON artifacts remain identity-only.
- Reversible Table 1 code tokens are HMACs under a host-only, context-stable
  binding key rather than unsalted hashes of private labels. Rebinding the same
  stopped/resumed context reproduces the token; a new host context does not.

## Negative guarantees and tests

- Unknown/custom/wrapper/endpoint rejection:
  `test_custom_client_cannot_self_register_as_offline`,
  `test_unregistered_top_level_wrapper_cannot_inherit_child_trust`,
  `test_registered_wrapper_rejects_mutated_child_graph_before_delivery`,
  `test_remote_openai_transport_cannot_be_authorized_as_local`, and
  `test_registered_openai_transport_mutation_is_rejected_before_delivery`.
- Callable/construction attacks:
  `test_registered_mock_instance_method_override_is_rejected_before_callback`,
  `test_registered_mock_class_method_mutation_is_rejected_before_callback`,
  `test_registered_vision_mock_image_method_override_is_rejected_before_callback`,
  and `test_unconstructed_exact_openai_object_cannot_gain_local_authority`.
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
  `test_every_agent_context_uses_the_same_outbound_safe_projection` and
  `test_step_summary_projection_rejects_numeric_phi_and_unknown_structure`.
- Tier-2 structured review without raw artifacts:
  `test_jury_receives_artifact_identity_not_arbitrary_artifact_text`.
- Delivery surface scan across `src/`, `tools/`, `scripts/`, and `examples/`,
  including `complete` and `complete_with_images`:
  `test_production_prompt_calls_use_the_authorized_delivery_boundary`.
- Wide trajectory preservation and transport budget:
  `test_wide_trajectory_projection_preserves_every_exact_window_coordinate`
  and `test_wide_trajectory_generation_prompts_stay_below_transport_gate`.
- Host-keyed stable Table 1 token:
  `test_private_code_tokens_are_host_keyed_and_stable_after_rebinding`.
- Overlapping mock prompt routing:
  `test_pattern_scripted_mock_prefers_later_specific_overlapping_marker`.

## Verification

- The exact canonical eight-file command requested by supervision collected
  **169 tests and passed 169/169**. It includes both durable-budget pipeline
  regressions; neither was deleted, relaxed, or deselected.
- `pytest test_tier2_jury.py`: **13 passed, 2 skipped**.
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
