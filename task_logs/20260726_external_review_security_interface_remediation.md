# External review security and interface remediation

- Date: 2026-07-26 08:56 EDT
- Module: `agent`
- Task ID: `AGENT-MCP-REVIEW-SECURITY-20260726`
- Branch: `fix/external-review-20260724-p0-p1`
- Starting point: `639cd9d`
- Final code commit: `d9063b0`
- Upstream observed after code work: `origin/fix/external-review-20260724-p0-p1@e6f5ddc`

## Outcome

The immediate security and interface findings accepted from the external review
are closed in four scoped commits. The implementation keeps Route B: the
explicit `WorkflowEngine` state machine remains the production orchestrator,
LangGraph is not restored, and human-review continuation remains explicitly
limited to the same process.

| Commit | Scope | Result |
|---|---|---|
| `e6f5ddc` | Patient-selector authorization and audit | Any supplied `patient_ids`, including an empty selector, requires `read_patient_data` before a loader is invoked. A fail-closed, PHI-free `patient_access_requested` event is persisted before the read and `patient_access_completed` before any response or evidence write. Raw patient identifiers are not stored in the audit event. |
| `49b08de` | Human-review authority | Review requests bind the complete typed `AnalysisPlan`, canonical plan digest, evidence SHA mapping, configuration digest, submission-profile reference, capability-activation digest, and run-input-capsule SHA. Structured plan, evidence, or execution-identity drift invalidates prior approval. Corrupt or partial typed plan authority fails closed. |
| `a47f7f7` | Pause/resume interfaces and concept-dictionary exports | Interactive CLI displays digest-bound review requests and resumes on the same live pipeline. Non-interactive CLI and MCP return an explicit nonterminal `human_review_pending` result instead of implying completion or durable resumability. Concept exports use collision-resistant physical names, preflight every destination, publish without overwrite, and expose a host-controlled logical-to-physical mapping. |
| `d9063b0` | Secret provenance and debug diagnostics | Secret configuration values and secret-looking values are represented by digests in provenance. LLM diagnostics require an exact opt-in flag plus an explicit directory, are recursively redacted and bounded, and are written with owner-only permissions. No default global raw-response dump remains. |

## Verification

The fixed HEAD remained `d9063b0` across the combined verification window.

- Focused/adjacent functional matrix: **313 passed, 2 skipped, 13 warnings** in
  9.01 seconds.
- Ruff lint over `src/easyicu`, `tests`, and `tools`: passed.
- Ruff format check over the changed owner files: passed.
- Deptry: 491 files scanned, no dependency issue.
- Import Linter: 7 contracts kept, 0 broken.
- Research-agent module graph: no drift from the checked architecture baseline.
- `git diff --check`: passed.

The combined functional matrix covered MCP dispatch and transport, patient
authorization/auditing, workflow review authority, CLI review handling, MCP
pending-state semantics, concept export collisions/no-partial-write behavior,
configuration provenance, provider debug captures, parser error diagnostics,
and neighboring pipeline/configuration contracts.

## Explicit boundaries

- This is Route B. Only **same-process resume** is supported. The interactive
  CLI can resume because it retains the live `ResearchAgentPipeline`; a
  non-interactive CLI invocation or MCP tool call does not retain that object
  and therefore reports `human_review_pending` without a false resume promise.
- No durable job supervisor, job ID/idempotency protocol, multi-worker recovery,
  service-restart recovery, or LangGraph runtime was added.
- The roughly 38-minute full research-agent suite was not rerun after these four
  commits. The shorter layered matrix was chosen because it directly covers the
  modified interfaces and safety boundaries.
- No Provider call, Docker analysis, patient extraction, real patient database,
  `/Volumes/外置硬盘/databases`, or other external-disk dataset was read during
  this remediation. These are code and contract tests, not a real-data
  validation claim.
- Debug diagnostics are local troubleshooting material, not scientific evidence
  or replay authority. A caller that needs them must point the explicit debug
  directory at an appropriate run-local diagnostic location.
- Durable Route A, TLS/reverse-proxy deployment hardening, stronger multi-user
  job isolation, and broader scientific-design contracts remain separate
  product or research-method tasks. They are not silently claimed by this patch.

## Repository handoff

At the end of the code batches, `origin/fix/external-review-20260724-p0-p1`
pointed to `e6f5ddc` and local `HEAD` pointed to `d9063b0`; the local branch was
three commits ahead. This session did not push. A remote CI run should follow
only after the owner elects to publish the remaining local commits.
