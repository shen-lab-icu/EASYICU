# DockerRunner output-mount lifecycle investigation

Date: 2026-07-20

Branch: `window2/sandbox-output-lifecycle-20260720`

Baseline: `6aeecc001fd950227c11b4314898fe840f09c781`

Scope: sandbox/container output directory lifetime only; no provider calls and no online E2/E3 runs.

## Outcome

The archived `analysis_cohort.parquet` failure is associated with an immediate
second execution of the same step over Docker's overlapping read-only run-root
and writable nested-output bind mounts. The generated script created
`STEP_OUT_DIR` before doing any work, and the only code change between the
successful execution and the failing execution changed three attrition-rule
identifiers. No output path, directory creation, or write statement changed.

One runner-owned lifecycle defect was proven: after a successful
`docker run --rm`, `DockerRunner.run()` treated `returncode == 0` as proof that
the container and its nested mounts were gone. Non-zero and timeout paths, by
contrast, explicitly confirmed teardown before collecting or reusing output.
The exact Docker Desktop internal unmount sequence is not observable from the
archive, so this report does not claim a lower-level daemon bug. It does establish
that the host crossed an authority boundary without proving the output mount was
quiescent.

The minimal fix makes the successful path obey the same invariant. It first
inspects the unique container id/name. A normal `--rm` "no such container"
response confirms removal. A still-present or ambiguous container is routed
through the existing stop/kill/remove/wait teardown. If absence still cannot be
proved, output collection is fail-closed and the cleanup sentinel remains for
the next attempt.

## Archived evidence and exact sequence

Affected run:

```text
research_output/_development_postqc_n1000_20260720_v15/
  bench_e3_gpt56luna/E3_kdigo_gradient/aware/
  run_20260720T065322_97b3a6
```

Affected step: `01_define_analysis_cohort`.

Sequence from `audit_log.jsonl`:

```text
07:35:12.850817Z  running resumed script
                    -> returncode 0, duration 0.559 s
07:35:13.598629Z  deterministic repair
                    attrition_rule_id_canonicalization_v1
07:35:13.917251Z  running repaired script
07:35:14.621423Z  execution failed
```

The second run began about 0.32 seconds after the deterministic repair event.
Its container ran for 0.501 seconds. The first output write failed:

```text
analysis_cohort.to_parquet(analysis_cohort_path, index=False)
FileNotFoundError: /easyicu-run/steps/01_define_analysis_cohort/
                   outputs/analysis_cohort.parquet
```

The sealed candidate executes
`step_out_dir.mkdir(parents=True, exist_ok=True)` before cohort work. A diff of
the successful and failing scripts changes only three `criterion_id` literals
and their matching comparisons. It does not change `STEP_OUT_DIR`, `mkdir`, the
parquet path, or the write call.

The failing Docker command used the topology:

```text
<run root>                         -> /easyicu-run (read-only)
<run root>/steps/<step>/outputs    -> /easyicu-run/steps/<step>/outputs (read-write)
```

The archive contains one relevant `FileNotFoundError` at this first parquet
write. Host-side output reconstruction after failure explains why the archived
directory now contains runner provenance files even though the in-container
write failed.

## Lifecycle trace

- `DockerRunner.prepare_step_dir()` ensures a real step and output directory.
- `_clear_step_outputs()` removes children but does not replace the output
  directory.
- `build_command()` overlays the writable output directory below the read-only
  run-root mount.
- Timeout and non-zero exits call `_teardown_container()` before scanning
  output.
- Before this patch, a zero exit set `teardown_confirmed = True` without any
  Docker control-plane check.
- `materialize_sealed_run_result()` can atomically exchange the whole output
  directory, but it is selected instead of runner execution in the same step
  control path. No evidence shows it overlapping the affected container.
- No other production owner was found deleting or renaming the live output
  directory during this execution.

## Reproduction and regression evidence

Before the source change, a no-API stress loop using the same parent-read-only /
nested-writable bind topology completed six iterations and then left the next
`docker run` container present and hung well past the 0.25-second payload. The
diagnostic container was force-removed; no archived run was touched.

A red unit test then demonstrated the owner defect: a successful Docker process
performed no teardown/absence confirmation before output collection.

With the fix applied, the actual `DockerRunner` repeatedly executed the same
step id 20 times against image
`easyicu-research-agent:1.0.0`. Each payload recreated `STEP_OUT_DIR`, slept,
and wrote `analysis_cohort.parquet`. Results:

```text
iterations: 20
successful: 20
outputs_safe_to_collect: 20/20 true
analysis_cohort.parquet present: 20/20
```

This is an offline runner stress check only. It did not invoke the Planner,
Coder, provider/API, or any E2/E3 workflow.

## Tests added

1. Successful `--rm` execution must prove the container is absent before
   collecting outputs.
2. A nominally successful generated program whose container absence cannot be
   proved must expose no artifacts, must report
   `outputs_safe_to_collect=False`, and must retain its sentinel.

Existing non-zero, timeout, stale-sentinel, mount-security, provenance, and
runner-factory tests remain covered by the focused runner suite.

## Scope and non-claims

- No archived run output was modified.
- No online experiment or model/provider call was made.
- No forbidden central orchestration, repair, preflight, typed-input, prompt,
  or figure-renderer file was edited.
- The patch does not redesign the overlapping mount topology. It closes the
  proven quiescence gap at its owning boundary. A future mount-layout redesign
  would be a broader security/compatibility change and is not justified by this
  single archived incident.
