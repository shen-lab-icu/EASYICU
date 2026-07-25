# PHI / outbound hardening supervisor increment

Date: 2026-07-22
Branch: `codex/phi-outbound-hardening-20260722`
Original base: `9401b3168d29f22ab657842f3a529bad479803d5`
Current blocker-fix base: `7f2e699`
Status: final blocker increment prepared for independent review; not merged or pushed

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
  `complete`, optional `complete_with_images`, callable code objects,
  `__getattribute__`, the actual OpenAI transport object, and wrapper child
  graph. Instance/class method mutation, in-place `complete.__code__` mutation,
  post-authorization `_client` replacement, or an exact-type `object.__new__`
  pseudo-OpenAI object is rejected before its callback runs. Legitimate OpenAI
  connection-pool rotation updates the binding only from the reviewed rebuild
  method.
- Step summaries now use an explicit host-owned aggregate field schema. Numeric
  values are not safe merely because they are numeric: patient/stay identifiers,
  individual ages/labs, arrays, and unknown nested values are dropped. Tier-2
  parses structured content only from its registered process-manifest filenames;
  arbitrary JSON artifacts remain identity-only.
- Reversible Table 1 code tokens are HMACs under an independently random
  32-byte host-only secret rather than unsalted hashes, public plan material,
  or timestamps. The secret is excluded from the public plan/context and is
  restored from a digest-checked mode-0600 private runtime checkpoint, so a
  stopped/resumed run reproduces the token without making it dictionary-
  guessable from candidate labels.

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
  `test_unconstructed_exact_openai_object_cannot_gain_local_authority`,
  `test_authorized_openai_rejects_replaced_transport_before_callback`,
  `test_authorized_openai_rejects_getattribute_dispatch_mutation_before_callback`,
  and `test_authorized_openai_rejects_in_place_complete_code_mutation`.
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
- Replanner probe identifier denial:
  `test_replanner_probe_projection_rejects_all_identifier_suffixes`.
- Tier-2 structured review without raw artifacts:
  `test_jury_receives_artifact_identity_not_arbitrary_artifact_text`.
- Delivery surface scan across `src/`, `tools/`, `scripts/`, and `examples/`,
  including `complete` and `complete_with_images`:
  `test_production_prompt_calls_use_the_authorized_delivery_boundary`.
- Wide trajectory preservation and transport budget:
  `test_wide_trajectory_projection_preserves_every_exact_window_coordinate`
  and `test_wide_trajectory_generation_prompts_stay_below_transport_gate`.
- Random host-keyed stable Table 1 token and private resume checkpoint:
  `test_private_code_tokens_are_host_keyed_and_stable_after_rebinding` and
  `test_private_code_token_secret_restores_from_private_checkpoint`.
- Registry-only mock classification:
  `test_mockish_classification_uses_only_registered_offline_graphs`.
- Overlapping mock prompt routing:
  `test_pattern_scripted_mock_prefers_later_specific_overlapping_marker`.

## Verification

- The exact canonical eight-file command requested by supervision collected
  **175 tests and passed 175/175** outside the nested sandbox. It includes both durable-budget pipeline
  regressions; neither was deleted, relaxed, or deselected.
- Table One / PlanAuthority / resume-controller focused validation passed
  **144 tests** before reaching 15 pre-existing full-pipeline Mock-subclass
  failures; the new private checkpoint tests themselves passed. Those failures
  already exist at the `7f2e699` trust boundary and were not bypassed by
  weakening offline registration.
- Ruff, changed-file Black check, `py_compile`, and `git diff --check`: pass.
- `tools/arch_measure.py --diff tools/arch_baselines/execution_phase.json`:
  no lower-is-better regression.
- `tools/research_agent_module_graph.py --diff
  tools/arch_baselines/research_agent_module_graph.json`: pass, no new cycle.

The architecture baseline was re-emitted because the private stop/resume
checkpoint adds 8 control-plane lines to `run_execute_phase` and 13 lines to
`pipeline.py`. It adds no nested functions, closure captures, or module cycle;
the post-emit diff is zero. This is a candidate for final incremental review,
not a claim that PHI/outbound hardening has been merged or paper-authorized.

## Increment from `debf447`: bind the complete outbound call chain

This final scoped increment addresses two concrete code-object mutation
attacks reported by supervision. It does not broaden provider capability or
change the external-authorization policy.

- Construction and trusted records now retain one immutable callable contract
  for the concrete client type: `complete`, `complete_with_usage`,
  `complete_with_images`, `_rebuild_openai_client`, `__getattribute__`, and
  every available Python code object. Delivery rejects any instance override,
  class replacement, or in-place code mutation before handing over a prompt.
- Transport refresh no longer treats the current mutable rebuild method as its
  authority. It requires the construction-time rebuild function and code
  object, verifies that the actual caller frame is that reviewed code object,
  and rechecks both construction and trusted registry records under the lock
  before updating only the transport identity.
- The reviewed rebuild remains functional. Tests now create OpenAI adapters
  through the real constructor with an injected in-memory SDK transport;
  transient retry/rebuild behavior is exercised without assigning a replacement
  rebuild method after authorization.

New negative regressions:

- `test_authorized_openai_rejects_in_place_complete_with_usage_code_mutation`
  mutates `OpenAIClient.complete_with_usage.__code__` and proves zero callback
  invocations.
- `test_malicious_rebuild_cannot_refresh_transport_authority` mutates the
  rebuild code, swaps the transport, attempts registry refresh, restores the
  original code, and proves both refresh and later delivery fail closed with
  zero malicious transport calls.

Positive preservation:

- `test_reviewed_openai_rebuild_can_rotate_transport` uses the unchanged
  reviewed rebuild path and proves the replacement in-memory transport can be
  authorized and called once.
- Existing retry, retry-after, streaming, response parsing, and provider-budget
  tests use the real rebuild method rather than an authorization-bypassing test
  override.

Verification at the uncommitted increment:

- Supervisor eight-file matrix: **178 passed** (the original 175 plus the three
  tests above).
- Provider factory + concrete LLM adapter focused suite: **77 passed**.
- Ruff, Black check, `py_compile`, and `git diff --check`: pass.
- Architecture diff: no lower-is-better regression.
- Research-agent module graph diff: pass, no new cycle.

This remains an isolated review candidate. It has not been merged or pushed,
and no real Provider, Docker, patient data, or Canonical9 run was used.
