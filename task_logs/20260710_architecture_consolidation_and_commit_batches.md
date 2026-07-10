# 2026-07-10 EasyICU architecture consolidation and commit batches

Task IDs: `WEBAPP-FASTAPI-NATIVE-QA`, `IDEA-MINING-DISCOVERY-MODULE`, and `DATA-FIX1`.

## Outcome

The high-level product split is sound: the data foundation, native Web platform,
Idea Mining, and Research Agent are distinct capabilities with real contracts.
The main architectural risk was inside those boundaries: duplicated provider and
database rules, a Web handoff parallel to the canonical Agent contract, large
registration files, and many related statistical modules left at one package
root. This pass made a first compatibility-preserving consolidation. It did not
attempt a big-bang rewrite of the remaining 4,000-6,000-line orchestration files.

Before starting new refactors, the existing reviewed repairs were verified and
committed in four independent batches, as requested:

| Commit | Scope |
|---|---|
| `0de31a7` | `fix(data)`: cohort, bounds, conversion, outcome and table contracts |
| `bc1a684` | `fix(web)`: local services, SSRF/Host controls, bounded workspace flows |
| `85ecc7c` | `fix(agent)`: runtime isolation, EvidenceStore and MCP transport |
| `e9cce4c` | `fix(discovery)`: handoff, outcome and figure provenance |

No commit was pushed.

## Architecture changes

| Commit | Change and boundary |
|---|---|
| `aca08a0` | Moved 13 statistical implementations into `easyicu/research_agent/methods/`; added a package-boundary regression. |
| `ad6f998` | Added `research_agent/providers/factory.py` as the single key/base-URL/client construction policy for MCP, discovery and benchmark entry points. |
| `3e06904` | Made `PipelineConfig` and constructor kwargs reflect the same field set and fail fast on unknown keys. |
| `a68d363` | Kept one documented `temporal_features` compatibility shim for generated scripts; other moved method modules have no root duplicates. |
| `86709b0` | Adapted native Web Idea Mining to the canonical `DiscoveryHandoffPacket`, persisted a fixed hash-bound artifact, and required pre-seed revalidation. |
| `a301072` | Created `webserver/routes/system.py` and moved 13 system/settings/capability adapters out of `app.py` while preserving route names and order. |
| `4b80294` | Added typed `databases/profiles.py`; six public databases and two demo sources now derive stay table/ID, label, alias and order from `data-sources.json`. |
| `98dab6c` | Updated method-suite runner validation and comments to the new `research_agent/methods/` owner path after the broad sweep caught the stale assumption. |
| `02783c5` | Corrected PEP 621 license metadata so the declared `setuptools>=68` baseline can build a wheel. |

The database consolidation also fixed a real drift bug: `mimic_demo` previously
fell back to `stay_id` in normalization, explicit batching and total-count paths;
it now consistently uses `icustays/icustay_id`. `DATABASE_ID_CONFIG` remains a
publicly importable, lazy, read-only `Mapping` with compatible `[]`, `get`,
`items`, and `dict(...)` behavior.

The canonical Web handoff remains locked after creation:
`human_confirmed=false`, `analysis_ready=false`, `reportable=false`, and
`draft_unlocked=false`. Unknown source databases are recorded as `unspecified`,
not silently called MIMIC-IV. Artifact, envelope, partial-field, identity, and
"tamper plus legitimate re-plan" cases all fail closed before Agent Project
creation; only a genuinely pre-canonical legacy envelope is refreshed.

## Verification

Focused verification performed before each commit:

```text
Existing data batch: 39 passed, 9 real-data tests skipped
Existing Web batch: 51 passed
Existing Agent batch: 61 passed
Existing discovery batch: 162 passed, 1 warning

Methods package: 75 passed, 7 skipped; final boundary/method registry: 28 passed
Provider factory + PipelineConfig contract: 56 passed
Web canonical handoff: 25 passed; adjacent discovery/security suite: 62 passed
System route extraction: 17 passed
Database profiles + resources/API cache/full workspace summary: 137 passed

Final changed-boundary combination:
343 passed, 7 skipped, 1 warning in 502.06s

ruff check src tests tools:
All checks passed

python -m compileall -q src/easyicu:
passed

git diff --check:
no output
```

A repository-wide 3,222-test run first found the stale method-runner path check;
that defect was fixed in `98dab6c`. The rerun then completed 125 passed / 2
skipped with no failures before it was intentionally stopped after 17m37s in an
unrelated cache-off pipeline group that repeatedly executes full generated
analysis scripts. The 343-test changed-boundary matrix above is the completed
final regression; this pass does not claim a completed all-repository run.

Packaging verification used the repository's declared build backend with
`setuptools 72.1.0`:

```text
wheel: /tmp/easyicu-build-20260710/easyicu-1.0.0-py3-none-any.whl
size: 2.6 MB
sha256: 942ef3cb198b2216223fba88a417227376b5c4b418dc07525aa1a340639b8935
wheel import smoke: ok
mimic_demo: icustays/icustay_id
system routes: 13
provider factory: easyicu.research_agent.providers.factory
```

## Deliberately deferred boundaries

- `api.py` is still about 4,800 lines. A later phase should introduce an
  `easyicu/api/` package behind the existing public import facade, split by
  loader/query, batching, extraction, cohort, and compatibility surfaces.
- `research_agent/pipeline.py` and `pipeline_execute.py` remain large. Their next
  split should follow explicit state/execution/reporting seams, not copy closures
  or introduce another catch-all module.
- `webserver/app.py` still owns many feature routes. Future batches can migrate
  one domain at a time into `webserver/routes/`, each with a method/path/name
  snapshot like the new system router.
- `database_config.py`, deep datasource join branches, legacy cohort labels, and
  Research Agent concept aliases still contain older database-specific metadata.
  They were left unchanged to keep this profile migration bounded.
- The 3,222-test full suite, live Docker daemon, real provider, and six-database
  real-export checks remain separate validation tasks.

## Next action

Use the same phased pattern for the next structural change: first freeze public
imports and route/API contracts, then extract one internal domain, run focused
and adjacent regressions, and commit it independently. The safest next targets
are the remaining FastAPI route groups and a compatibility-facade split of
`api.py`; the Research Agent pipeline monolith should follow only after its state
ownership is documented.
