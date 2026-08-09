# Research-agent orchestration decision

Status: accepted for the paper-first runtime
Decision date: 2026-07-26

## Current decision

EasyICU uses one explicit workflow engine for:

`plan → human_review → execute → write → finalise`

The stable application seam is
`easyicu.research_agent.orchestration.workflow.WorkflowEngine`. The current
implementation is `PipelineWorkflow`; callers must not depend on its private
state. Human-review requests and decisions remain schema-versioned,
digest-bound models.

Paused work is currently resumable only by the same pipeline instance in the
same operating-system process. `HumanReviewPending.resume_scope` therefore
reports `same_process` and records `resume_pid`. The UI or an API must not
describe such a pause as restart-safe.

## Why the previous graph was retired

The former LangGraph state stored handles to live phase objects kept in a
process-local dictionary. An `EvidenceStore`, provider resolver and run-scoped
services could not be serialized or reconstructed from its checkpoint. It
therefore added a second dispatcher without providing cross-process recovery.

## Post-paper durable route

A future durable implementation may replace `PipelineWorkflow` behind
`WorkflowEngine`; it must not restore the former process-local registry.
Before advertising durable resume it must:

1. persist only `run_id`, artifact paths, immutable digests, phase/status and
   schema-versioned review data;
2. rehydrate evidence and services through explicit repositories/factories;
3. verify reconstructed artifacts and authority against their stored digests;
4. pass an integration test that pauses a run, terminates the service, starts a
   new service process and resumes to the same evidence outcome.

Until all four conditions hold, `same_process` is the only supported resume
scope.
