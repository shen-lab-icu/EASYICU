# Aggregate study replay integration

The replay is an offline projection, not execution, publication approval, or a
replacement scientific review. No Planner contracts, review finding codes,
scientific schema versions, execution owners, or UI files change.

## Owners and calls

`easyicu.webserver.study_replay.build_study_replay(run_dir, *, conversation=None)`
returns `dict[str, bytes]`. `export_study_replay(run_dir, destination, *,
conversation=None)` validates first and publishes a new directory. An existing
destination is refused. These are host calls; no HTTP endpoint is introduced.

`run_dir` is the Web wrapper already produced by `agent_pipeline_runs`:

| Input | Use |
| --- | --- |
| `run_context.json` | Question, run and source identity |
| `source_run_manifest.json` | Matching source/run, status, trace timings |
| `agent_plan.json` | Recorded scientific plan, operational fields removed |
| `manuscript_draft.json` | Answer text, claims, sentences and recorded status |
| `scientific_readiness.json` | Optional existing review projection |
| `scientific_plan_review.json` | Optional existing plan review |
| `result_tables.json` | Digest-pinned table coordinates, not preview contents |
| `figure_gallery.json` | Registered PNG coordinates, captions and labels |
| `conversation_replay.json` | Optional Pi lifecycle projection |
| `evidence/evidence_index.json` | Registry, here or under `pipeline/<run_id>/` |

The optional conversation argument accepts the same
`easyicu.pi-conversation-replay/1` shape. Only `host_action` turns whose
`child_job_id` joins a child job with this `result.run_id` become decisions.
Private Pi JSONL, user messages, assistant transcripts and tool payloads are
never exported. Missing decisions or durations stay absent/null. Timestamp
durations explicitly identify the span since the first recorded step event.

Dependency edges are gallery -> replay -> evidence/IO/policy; evidence uses the
existing host figure privacy constants, and policy uses the external egress
floor plus the existing subject-count name rule. These owners live in the Web
layer; they add no Research Agent capability or module-graph entry.

## Output contracts

The directory contains `index.html`, `replay.json`, `manifest.json`, and optional
`figures/<evidence_id>.png`. HTML has no script, external assets or network
dependency. Text is escaped; the lead owns the final UI and user copy.

`easyicu.study-replay/1` contains `run_id`, `source`, `question`, `status`,
`reviewed_plan`, `trace_steps`, `decisions`, `result_answer`, `scientific_review`,
`plan_review`, `aggregate_tables`, and `figures`. Table rows retain the existing
renderer contract (`renderer`, `headers`, `rows`, truncation booleans), with
evidence ID and source digest. Reviews retain their original authority; absent
review files become null.

`easyicu.study-replay-manifest/1` records the source identity, sharing policy,
input-record digests, declared source databases (including validation sources),
the inspected evidence closure, decision-projection digest,
and byte lengths/SHA-256 for all bundle members except the manifest itself.
No clock, random ID, host root or absolute path enters the output. The manifest
is an integrity receipt, not a digital signature or a new disclosure approval.

## Disclosure boundary

- A complete registered result table is checked before presentation. CSV, TSV
  and JSON `{headers, rows}` tables are accepted; raw columnar files, code,
  HTML, PDFs, arbitrary JSON and unregistered files are not copied. Every row
  must carry inspectable subject-count evidence. Ambiguous tables fail closed.
- Identifier columns/values, event timestamps and nested row payloads are
  refused. Paths, URLs and credential-shaped text are scrubbed. Operational
  provider, environment and source-code fields are removed recursively. Opaque
  credential values already present in operational records are also scrubbed
  if echoed into scientific text; no credential file is opened.
- The existing external-sharing floor, `MIN_DISCLOSED_GROUP_SIZE` (20), applies
  to all sources. Zero is retained. Any positive count below the floor refuses
  the entire bundle, including prose and figures, avoiding disclosure through
  totals or derived results. This is stricter than manuscript signoff's review
  policy and does not change it.
- PNGs require the trusted host producer, current host audit schema/version,
  matching figure identity, registered contract, and exact source digests.
  Every source table is rehashed and reinspected. Text that would need
  scrubbing in the rendered contract or its source labels makes the PNG
  unshareable. Only supported, bounded PNG pixel chunks survive; metadata,
  animations, other image formats and unsupported PNG encodings are refused.
  The receipt retains `mark_count_verified`; this exporter does not claim OCR
  or independent visual reconstruction of a trusted host renderer's pixels.
- Reads are bounded and reject links/traversal; output is staged only after
  validation. Counts above a trillion are refused before integer conversion;
  write failures expose only a stable code. The filesystem registry remains the trusted host authority, as
  for existing figure egress. No caller flag can relax the export policy.

Errors are `ReplayExportError` with a stable `.code` and code-only text. Key
codes are `replay_patient_level_forbidden`, `replay_small_cell_forbidden`,
`replay_aggregate_unproven`, `replay_evidence_class_forbidden`,
`replay_digest_mismatch`, `replay_figure_uncleared`,
`replay_figure_text_unsafe`, `replay_path_forbidden`, `replay_size_limit`,
`replay_source_identity_mismatch`, and `replay_record_invalid`.

## Gallery handoff

`build_replay_gallery(root, registry)` returns a byte bundle containing
`gallery.json` plus `replays/<entry_id>/...`. `export_replay_gallery` publishes
it to a fresh directory. `website/replay_registry.json` explicitly registers
two aggregate-only **synthetic fixtures**, not results of real research runs.
No demo downloads, studies, exports from clinical sources, or conversions are
needed to exercise them.

Registry schema `easyicu.replay-gallery-registry/1` has `entries` with `id`,
`run_id`, `run_dir` (relative to root), `source_manifest_sha256`, `title`, and
`study_type`. Each registered snapshot must match before and after replay
construction. Source classification comes from recorded source identity, not
registry labels. Credentialed runs can produce private aggregate replays but
cannot enter this public gallery.

Gallery entries expose `id`, `title`, `question`, `database`, `study_type`,
`source_kind`, `real_data_acceptance`, `run_id`, `replay`, and
`replay_manifest_sha256`. Every fixture/demo is `not_accepted` on real data;
the site must retain this code when it supplies translated labels.

Host CLI entry points (not run against clinical data in this worker):

```sh
PYTHONPATH=src "$PY" tools/export_study_replay.py WRAPPER NEW_DESTINATION
PYTHONPATH=src "$PY" website/scripts/build_replay_gallery.py --output NEW_DESTINATION
```

The website builder and download route can consume these owners without
starting a runner or requesting provider credentials. The initial renderer is
intentionally plain English scaffolding for the lead's UI work.
