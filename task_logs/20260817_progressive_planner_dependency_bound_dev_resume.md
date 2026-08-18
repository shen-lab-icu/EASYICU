# Progressive Planner dependency-bound Dev resume

Date: 2026-08-17  
Task: `FIG2-DEV9-HELDOUT27`  
Implementation commit: `074480b`  
Branch: `integration/figure2-e1-h3-20260816`

## Decision

E1 remains the only Dev9 canary until its complete 11-stage path succeeds. The
next optimization is development-only, dependency-bound Progressive Planner
checkpoint reuse. It does not relax planning, execution, evidence, review, or
publication gates, and it is forbidden for paper-facing profiles and formal
Figure 2 batches.

This decision follows two bounded E1 failures before Execute:

- job `61533f72575f`, run `run_20260817T084213_a57aab`: 12 Planner calls,
  280,876 accounted tokens, estimated cost `$2.95370`; outline, foundation, and
  steps 1-4 were valid before a distribution-contrast compiler finding.
- job `5a3dfa3c42ff`, run `run_20260817T091636_c01ce3`: 4 Planner calls,
  85,550 accounted tokens, estimated cost `$0.90244`; a stochastic foundation
  failure occurred before the previously reached prefix could be recovered.

Neither run entered Execute and neither is publication evidence.

## Implemented owner contracts

1. `planning/progressive_resume.py` owns deterministic replay authority and
   prefix reconstruction. The dependency digest binds semantic ResearchContext
   and article context, scientific/action/variable/literature/know-how/cohort
   authority, exact cohort file SHA-256, provider/model signature, and prompt
   version. Volatile run paths are excluded, while every reused step is checked
   against the current strict schema and recompiled by the current host.
2. `planning/progressive_artifacts.py` owns the typed append-only checkpoint
   chain, raw terminal file digest, canonical internal digests, predecessor
   closure, regular-file/size guards, and delayed import into the current
   EvidenceStore. A source chain is imported only after dependency validation
   and current-host prefix recompilation.
3. `orchestration/progressive_planning.py` owns one Progressive Planner call and
   its optional Dev source-chain lifecycle. A validated prefix remains auditable
   if a later suffix call fails; an invalid prefix is never registered.
4. `PipelineConfig` requires the checkpoint path and SHA-256 together, requires
   `progressive_v2`, forbids deterministic fallback, and accepts replay only in
   explicit diagnostic mode or a registered non-paper `*_dev` profile.
5. `run_research_agent_bench.py` exposes paired development flags only for one
   selected JSONL item and one arm. Repeats, formal batch binding, and Figure 2
   paper-acceptance mode reject cross-run checkpoint reuse.

Old checkpoints lack `resume_dependency_authority_sha256` and intentionally fail
closed. The first exact-head canary after this commit must start fresh; only its
new dependency-bound chain may be used for a later suffix continuation.

## Complexity result

- `ProgressivePlannerAgent.run`: 638 lines during the initial implementation,
  then 441 after extracting suffix materialization and artifact-chain mechanics;
  the pre-change function was 506 lines.
- `ResearchAgentPipeline._generate_or_resume_plan`: 449-line architecture
  baseline versus 433 lines after orchestration extraction.
- Research-agent module graph remains acyclic.

The checked-in architecture ratchet still reports four unrelated, pre-existing
branch drifts: `execution/phase.py` (+5 LOC),
`execution/phase_support.py` (+11 LOC), `authority/typed_binding.py` (+54 LOC),
and `agents/replanner.py` (+272 LOC). This change did not refresh the baseline or
claim those four findings closed.

## Verification

- Progressive replay, config, provider portability, benchmark profile,
  review/egress, and module graph: `197 passed`.
- Progressive replay, config, benchmark profile, module graph, and dependency
  directions after final fail-closed refinements: `122 passed`.
- Package dependency and static architecture boundaries: `9 passed`.
- Targeted Ruff: passed.
- `git diff --check`: passed.
- Benchmark CLI help exposes both paired development checkpoint flags.

No full exact-head CI was run because E1 is not 11/11 and this is not a freeze,
merge, release, or formal-experiment checkpoint.

## Next gate

Build a uniquely tagged runner from the exact post-documentation HEAD and run
one fresh E1 backend canary with the registered Dev profile. If it fails after
producing a dependency-bound prefix, retry only from the exact terminal
checkpoint and file digest. Run one Web smoke only after the backend path is
valid. Do not start E2 before E1 reaches the complete 11-stage path.
