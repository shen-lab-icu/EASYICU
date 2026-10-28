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
| `report_export_plausibility.py` | Aggregate-only clinical screening of one native-v2 database export, optionally compared with an older export. |
| `walk_easyicu.py` | Resumable HTTP walk of one configured source; saves aggregate results and never downloads data artifacts. |
| `run_analysis_bench_overnight.py` | Batch wrapper around `run_research_agent_bench.py` for long runs. |
| `run_openrouter_fullflow_validation.py` | Real-provider full-flow validation for OpenRouter-compatible models. |
| `evaluate_concept_usage_auditor.py` | Helper for checking concept-usage audit behavior. |
| `fetch_baselines.py` | Fetches external baseline systems into ignored local checkout directories. |

Before publishing an export, an administrator can run:

```bash
PYTHONPATH=src python tools/report_export_plausibility.py /path/to/export \
  --compare /path/to/older-export --out report.json --threads 2 --memory-limit 8GB
```

The root is the database export containing `_manifest.json`. The tool scans
only declared parquet files with DuckDB, applies the memory/thread settings,
and prints a Chinese/English summary and JSON (also saved with `--out`). All
thresholds and their rationale are in its `SCREENING_RULES` table. Rates use
fractions; coverage uses all distinct stays across the declared modules;
SOFA comparisons use stays with both maxima; Sepsis-3 reports both all-stay
and observed-stay denominators. Quantiles are approximate. Missing inputs warn.
It returns zero for screening findings and unreadable inputs and never prints
identifiers, patient rows, file listings or extreme-stay examples. Clinical
review of findings remains necessary; this report is not a publication gate.

## HTTP walk of a configured source

Run `walk_easyicu.py` against an already-running EasyICU server with a verified
provider and a registered source. It never launches a server, changes settings
or provider keys, or opens a database locally. Choose an output directory
outside this checkout:

```bash
python tools/walk_easyicu.py --base http://127.0.0.1:8780 \
  --source <configured-source-id-or-name> --out /private/easyicu-walk \
  --max-workflows 2
```

The default walk stops at plan review. Add `--approve-test-runs` to approve the
server's current digest-bound review offer and wait for execution. This flag
also permits selecting an option explicitly offered by a design card and
generating and reviewing its replacement plan. Other pending scientific
choices stay refused with their codes. Candidate plan submissions always carry
`candidate_plan_only=true`, including when the approval flag is present.

Use `--only environment,mapping,catalogue,literature,ideas,workflows,interactions`
to select phases. Each item has a deadline, configurable with
`--timeout-<phase> SECONDS`; `--request-timeout` caps individual requests.
`--max-workflows` caps runnable examples in catalogue order; items outside the
limit and method components without complete example questions are recorded
as skipped. Workflow questions come from their public worked-example packages.

Each example gets a recognisably named `EasyICU walk ...` project and a fresh
conversation: EasyICU binds scientific state to a project, so sharing one
project between examples would reuse another question's study. Interaction
rename/delete probes use a separate scratch conversation. Completed study
conversations and projects remain available for the researcher to inspect.
The task-table probe checks the public project workflow projection. This HTTP
tool does not claim browser rendering or keyboard interaction coverage.

`walk_report.json` (`easyicu.walk/1`) and `walk_report.md` contain terminal
statuses, reason codes, timings and explicitly projected aggregates. Artifact
names and counts are retained; data artifacts, tables, raw replies, paths,
error text, identifiers and provider credentials are never saved. The only
artifact preview requested is `literature_evidence.json`, projected immediately
to retrieval/screening/inclusion counts, source presence and role coverage.
Source presence counts mean one per returning connector, not article totals.
Idea Mining uses a fictional abstract with metadata-only, network-disabled
feasibility; evidence verdicts describe its published prior-art status.

Repeat with `--resume` and the same base/source/output to reuse successful,
refused and skipped receipts. Failed/timed-out items retry using saved owned
conversation/job coordinates. A known execution job is polled without another
approval. Adding the approval flag revisits plan-only results; increasing the
limit revisits previously limited examples. Source names are resolved again
and cannot silently point a resumed walk at a different registration.
Read-only source resolution and example discovery refresh before receipts are
reused; a discovery outage therefore gets its own current failure result.

Missing HTTP facts stay explicit: this server currently exposes mapping
metadata rather than source-observed concept coverage, extraction prefills
rather than a dedicated 19-module smoke receipt, and a planned-method count
rather than all planned identities. Build commit/image facts also remain
`not_reported` when the server does not publish them. A metadata projection or
a finished job alone never counts as a successful extraction/analysis.

Fast offline tests live in `tests/core/test_walk_easyicu*.py`; direct HTTP
schema-owner checks live in `tests/webserver/test_walk_easyicu_contracts.py`.
The slow ASGI contract smoke uses the installed `providers.mocks` offline
provider and synthetic conversation/execution services in a temporary
`EASYICU_HOME`. Run it on a machine that permits TestClient:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider -m slow tests/webserver/test_walk_easyicu_smoke.py
```

That smoke checks transport and public request shapes. A live source walk is
still needed to validate real scientific execution and runner availability.

## Archived locally, not uploaded

Historical experiment runners, local-model launchers, manuscript/submission
package builders, and submission-specific result aggregators are kept out of
the tracked repository (see the manuscript/submission entries in `.gitignore`
and `tools/legacy/`). They remain available locally for provenance but are not
part of the clean GitHub surface. Manuscript source-data, figures, and the
draft itself live in a separate private writing workspace, not in this package.

## Native web UI fit audits

`audit_web_popovers.py` presses every menu and disclosure of the running web
UI (three viewport sizes) and fails when a floating menu lands outside the
viewport, is clipped or covered, or does not close on Escape / an outside
press / its own opener. `audit_web_fit.py` checks each route at rest for
covered controls, floating elements outside the viewport, silently clipped
text, controls under 24 px, and horizontal overflow. Both need a running
server and Playwright with a Chrome channel; pass `--session
"pi_project=…&pi_session=…"` so the Guided conversation and entry composers
are covered, and `--out` to keep the JSON report. Both read their routes from
`web_ui_audit_routes.py`; `tests/webserver/test_web_ui_audits.py` fails when a
screen registered on `window.SCREENS` is neither audited nor listed as an
exclusion, and runs both tools live (marker `requires_web_server`) when
`EASYICU_WEB_AUDIT_BASE` names a running server (plus
`EASYICU_WEB_AUDIT_SESSION` for the Guided composers).

```bash
python tools/audit_web_popovers.py --base http://127.0.0.1:8765 --session "pi_project=<id>&pi_session=<id>"
python tools/audit_web_fit.py --base http://127.0.0.1:8765 --session "pi_project=<id>&pi_session=<id>"
```
