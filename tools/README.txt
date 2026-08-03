# EasyICU tools directory

This directory contains repository-level command entrypoints. These scripts are
not installed as public Python APIs; they are convenience commands for
benchmarking, validation, and maintenance from the repository root.

## Canonical six-database extraction

`reextract_native_export_v2.py` is the single maintained full-six extraction
entrypoint. It requires one clean Git checkout, pins every database worker to
that checkout's `src` tree, and publishes a database only after all 19 native-v2
Parquet files pass manifest/SHA validation. The run root must not already exist.

```bash
PYTHONPATH=src python tools/reextract_native_export_v2.py \
  --output-root /new/private/full6_run \
  --data-root /path/to/databases
```

`--data-root` must contain `mimiciv`, `mimiciii`, `eicu`, `aumc`, `hirid`, and
`sic`; repeat `--data-path DATABASE=PATH` to override individual locations.
The output contract is:

```text
full6_run/
  exports/{database}/{19 modules}.parquet
  run_manifest.json
  database_extraction_timing.csv
  .orchestration/attempts/...       # logs/specs; no runnable code clone
```

Each database runs in a clean child interpreter. On a 16 GB machine, or when
currently available memory is low, databases run serially. A larger server can
run up to three smaller databases concurrently; eICU remains exclusive because
it is the largest source. Batch size is computed continuously from stay count
and currently assigned memory: smaller cohorts stay one-shot when they fit,
while full eICU normally uses about three 67k-stay batches on a sufficiently
large server. At about 8 GB available it starts around 40k rather than falling
off a fixed 10k cliff. The extraction core can adjust from measured worksets;
a process-level OOM retries only unpublished staging data at a smaller batch.

Interrupted or failed runs retain completed databases. Resume with the exact
same clean commit, data paths, database order, and resource policy:

```bash
PYTHONPATH=src python tools/reextract_native_export_v2.py \
  --output-root /new/private/full6_run \
  --data-root /path/to/databases \
  --resume
```

The default `--resource-policy strict` requires `psutil>=5.9` and real
process-tree RSS/PSS support. It fails before creating a fresh run root when
that evidence is unavailable; RSS is never relabelled as PSS. The explicit
`--resource-policy allow-unsealable` fallback can extract on a platform without
PSS, but marks the run and timing rows unsealable. Install with
`python -m pip install 'psutil>=5.9'` for release runs.

The command and path handling use only cross-platform Python APIs. On Windows,
run the equivalent command with `py -3` and quote drive paths. Advanced users
may set a per-source override such as
`--database-batch-size eicu=50000`; the automatic policy should normally be
left in control.

## Benchmark tools

| File | Purpose |
|---|---|
| `run_research_agent_bench.py` | Main research-agent benchmark runner. Use `--arms aware` for paper-facing EasyICU workflow runs. |

## Maintenance and validation tools

| File | Purpose |
|---|---|
| `reextract_native_export_v2.py` | Canonical resumable six-database native-v2 extraction, resource measurement, and atomic publication. |
| `run_analysis_bench_overnight.py` | Batch wrapper around `run_research_agent_bench.py` for long runs. |
| `run_openrouter_fullflow_validation.py` | Real-provider full-flow validation for OpenRouter-compatible models. |
| `evaluate_concept_usage_auditor.py` | Helper for checking concept-usage audit behavior. |
| `fetch_baselines.py` | Fetches external baseline systems into ignored local checkout directories. |

## Archived locally, not uploaded

Historical experiment runners, local-model launchers, manuscript/submission
package builders, and submission-specific result aggregators are kept out of
the tracked repository (see the manuscript/submission entries in `.gitignore`
and `tools/legacy/`). They remain available locally for provenance but are not
part of the clean GitHub surface. Manuscript source-data, figures, and the
draft itself live in a separate private writing workspace, not in this package.
