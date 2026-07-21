# Research-agent development execution policy

This document defines the fast, non-paper execution mode used while validating
the EasyICU research-agent workflow. It is deliberately separate from a frozen
submission run: development results demonstrate that the workflow can execute
and enforce its contracts, but they are not manuscript estimates.

## Canonical pre-extracted six-database input

The current internal Canonical9/Extension3 experiment program consumes the
already completed six-database, all-module EasyICU export at:

```text
/Volumes/外置硬盘/easyicu_data/full6_20260717
```

Do **not** re-extract the six databases for Agent experiments. The benchmark is
defined as the downstream user journey after a user has already completed data
extraction with EasyICU. Experiments may build locked cohorts, post-QC
development samples, and typed artifacts from this export, but they must not
rerun raw conversion/extraction merely to start an Agent task. A newer export
may replace it only through an explicit data-foundation version change with a
new fingerprint and experiment ID.

## Post-QC sample: 1,000 stays

Use `--development-sample-size 1000` for Canonical9 development runs.

The sample is created only after the Agent has:

1. interpreted the question and locked the scientific cohort definition;
2. applied inclusion and exclusion criteria;
3. materialized the analysis cohort and completed cohort quality control; and
4. identified a unique stay-level identity column.

The runtime then ranks the locked identities by a seeded SHA-256 digest and
selects the first 1,000. It does not sample by exposure, outcome, database,
cluster, or result direction. When a longitudinal trajectory is present, the
trajectory is filtered to exactly the same selected identities before analysis.
The full parent cohort remains immutable and both parent and child digests are
recorded.

If fewer than 1,000 eligible stays remain, all eligible stays are retained and
the achieved size is recorded. If no verified post-QC cohort exists, sampling
fails before Coder spend. A sampled run is always marked `paper_authority=false`
and cannot satisfy the Figure 2 paper-acceptance gate.

For trajectory clustering, the 1,000-stay child cohort is sufficient for
development only when the Agent's ordinary cluster-quality contracts remain
estimable (for example, valid aligned trajectories and non-degenerate cluster
sizes). The runtime must not silently change the selected number of clusters,
method, features, window, or estimand to make the sample pass. An insufficient
sample is a structured development limitation, not permission to fall back to
the full cohort during the same run.

## Database availability

EasyICU does not require a user to possess all six supported databases.

- A single-database question may run against any one verified EasyICU export.
- Databases that are unrelated to the declared question do not block the run.
- A cross-database or database-specific task must name its required targets.
- If a required export, module, concept, or typed authority is absent, the run
  stops before scientific execution with the missing resource identified.
- The engine never substitutes another database or relabels its source to make
  a requested comparison appear complete.

This lets ordinary users work with the data they are licensed to access while
keeping cross-database claims fail-closed.

## New methods and Python packages

The Planner retains scientific ownership of the method. Package availability is
an execution constraint, not a reason for the runtime to choose a different
exposure, outcome, cohort, estimand, or method family.

Before the first planning provider call, the runner validates its actual Python
environment. The Coder receives only this verified allow-list. During generated
code execution the sandbox has no network and cannot install packages.

There are three supported outcomes when a planned method needs a package:

1. **Verified package available:** execute it and bind the package version and
   immutable runtime identity into provenance.
2. **Registered scientifically equivalent fallback available:** use the named
   fallback and record the degradation. A fallback may change implementation,
   not the Planner-owned estimand or method family.
3. **No valid implementation available:** fail closed with a runtime-capability
   limitation before spending repair attempts.

To add a package for future runs, maintainers add it to the curated method
package registry, pin it in the build dependency set, rebuild/version the Docker
image, and pass its import and smoke tests. Users may supply a custom immutable
runner image only if it implements the same capability and provenance contract.
Runtime `pip install`, unpinned downloads, and Coder-authored environment changes
are not supported.

## Development command pattern

```bash
PYTHONPATH=src .venv/bin/python tools/run_research_agent_bench.py \
  --bench-kind analysis \
  --arms aware \
  --provider openai \
  --runner docker \
  --development-sample-size 1000 \
  ...
```

After the shared engine, task authority, model, prompt pack, dictionary, runtime,
and evaluator are frozen, paper-facing runs omit `--development-sample-size` and
use the full locked cohort under a submission profile.
