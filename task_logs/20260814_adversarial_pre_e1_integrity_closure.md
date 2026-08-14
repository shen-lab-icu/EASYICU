# Adversarial pre-E1 integrity closure

Date: 2026-08-14

## Scope

An adversarial exact-worktree review found that the prior clean release did not
exercise several hostile state transitions. This batch fixes those boundaries;
it does not authorize a Provider run by itself.

## Closed findings

1. Provider usage mappings that carried model provenance but no token counters
   released the full transport reservation. A reservation now settles only when
   at least one explicit token field is reported; metadata-only completion stays
   conservatively accounted as `completed_usage_unreported`.
2. Approval authority hashed EvidenceStore records without re-reading the
   registered bytes. `EvidenceStore.verified_records()` now verifies every
   path/SHA before authority compilation, and publication figure consumers use
   the same verified-path boundary.
3. STRICT numeric binding treated an untyped same-value count as a typed effect
   claim. Explicit prose effect-scale/estimand declarations now require matching
   typed claim identity.
4. A durable human-review checkpoint could reach disk before Provider active
   time was paused. Pipeline now pauses immediately after checkpoint persistence;
   restart reconciles any crash-window `running` ledger to the checkpoint's
   `created_at`, excluding later downtime.
5. Planner could replace a user/host-authorized analysis family. An explicit
   `inferred_analysis_family` is now a closed canonical family authority; only
   contexts without that authority retain Planner family choice.
6. Formal Web review could approve an analysis-only primary capability and
   discover the publication ceiling only after execution. Web config now
   requires a reportable registered scientific capability before approval;
   separately labelled diagnostic callers retain analysis-only execution.
7. Standalone MCP pipeline calls lacked aggregate Provider accounting, and MCP
   timeout could return while the synchronous worker continued. Each MCP run now
   receives a durable Provider ledger injected through PipelineConfig/Services.
   Timeout bounds queue wait only; once dispatch starts, request lifecycle waits
   for the real operation to converge.
8. Capability inventory accepted any existing test symbol as reachability proof.
   Production rows now declare AST-verifiable public calls and downstream trace
   assertions.
9. The manuscript evidence consumer treated `00_probe` as a reportable success
   step, although the host diagnostic probe deliberately has no ordinary
   executor sidecar. Probe records are now excluded from the Writer namespace;
   ordinary successful steps remain sidecar-mandatory.

## Verification

- Focused changed-boundary matrix: 245 passed.
- Durable pause/provider/recovery matrix: 47 passed.
- Analysis-family/scientific-review/capability matrix: 76 passed.
- MCP server/transport matrix: 52 passed.
- Capability inventory governance: 7 passed; standalone audit reports OK.
- Broad pipeline/Web/recovery/config matrix: 391/400 initially passed. The nine
  failures exposed stale focused fixtures plus the probe/Writer boundary above;
  all nine failed node IDs then passed after remediation, and the final
  probe/Writer regression passed independently. The final post-fix broad rerun
  passed 458/458.
- Post-refactor family/scientific-review/Web verification passed 149/149.
- Independent pre-commit adversarial review then closed additional boundary
  cases that the first green matrices did not exercise: partial, malformed,
  zero and internally inconsistent Provider usage receipts; direct-pipeline
  Provider resume and terminal exhaustion; registered complete same-generation
  publication bundles; canonical host-probe evidence; reportability config
  misuse; and MCP cancellation/queue-timeout semantics. The combined focused
  Provider/recovery/MCP/publication/Writer matrix passed 199/199 before the
  final targeted additions, and every added negative node passed independently.
- Strict progress lint passed all six `CURRENT.md` pages with zero warnings.
- The shared family-authority guidance now states that caller-bound family
  authority cannot be replaced and intentionally adds 81 bytes to each
  provider-free Planner fixture. The measured maximum is 60,033 of the 120,000
  byte limit; Provider calls and patient-data reads remain zero. The resource
  baseline records this exact move in append-only history.
- The post-refactor dirty-worktree framework release ran every internal gate successfully:
  resource context, architecture, module graph, and 135 framework tests all
  returned zero. The receipt status remains `failed` solely because the
  integrity closure is still an uncommitted dirty worktree.
- Ruff over every changed Python owner and test: passed.
- `git diff --check`: passed.

## Release boundary

The fixes are currently an uncommitted worktree over `3de8f0e`. Formal Web E1
remains blocked until the reviewed changes are committed and the release tool is
rerun from that clean exact HEAD. The prior `3de8f0e` release receipt predates
this adversarial closure and cannot authorize E1. The latest dirty-worktree
diagnostic receipt is
`/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/opencode/e1-integrity-worktree-release-post-refactor.json`;
it proves the internal gates pass but is not a clean-head authorization.
