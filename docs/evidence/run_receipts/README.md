# Research-agent run receipts

Machine-generated records that a given research-agent run happened, what it
produced, and what governance status it carried. Build one with:

```bash
python tools/preserve_run_receipt.py <run_dir> \
  --out docs/evidence/run_receipts/<run_id>.json
```

Re-check a receipt while its run tree is still on disk:

```bash
python tools/preserve_run_receipt.py <run_dir> \
  --verify docs/evidence/run_receipts/<run_id>.json
```

## Why receipts exist

Run trees live under `output/` and `research_output/`, which `.gitignore`
excludes as regenerable scratch and which get pruned. The submission plan cites
those runs as evidence anyway — the active WebApp row cites
`run_20260829T024326_bfcbf6` ("11/11 analysis steps, 12 tables, 3 figures,
evidence-bound article draft") and that directory no longer exists anywhere on
disk. A prose paragraph in a task log is exactly the unauditable claim the
evidence machinery exists to prevent.

A receipt keeps the run provable after the tree is gone. On a measured run it is
about 4% of the tree it describes (1.7 MB run → 73 KB receipt, 183 artifacts).

## What a receipt actually is

The run's **own** decisions, copied verbatim, plus a SHA-256 inventory:

| field | source in the run directory |
|---|---|
| `run_id`, `research_question`, `started_at`, `finished_at` | `manifest.json` |
| `status`, `strict_fail_closed`, `gates`, `code_version` | `run_status.json` |
| `evidence_authority_head` | `.easyicu_evidence_authority_head.json` |
| `steps` | `steps/*/step_summary.json` |
| `artifacts` | every regular file, with size and SHA-256 |

The tool adds no gate and no runtime branch. It never upgrades a status and
never asserts publication readiness on a run's behalf — a run recorded as
`analysis_only` stays `analysis_only` in its receipt.

`--verify` first checks the receipt's own digest, then rebuilds the complete
artifact inventory and fails closed on changed, missing, newly added, duplicate,
or unsafe paths. It is meaningful only while the tree still exists; once a run
is pruned its receipt stays readable evidence but can no longer be re-verified
against source.

## Privacy

No artifact contents are copied. A receipt holds the research question,
run-generated relative artifact paths, byte sizes and digests — never row values
or cohort contents. Run directories whose filenames themselves contain patient
identifiers are not eligible for preservation and must be remediated first.

## Current records

None yet. The first receipt should be taken the next time a run is cited as
evidence in `EasyICU_当前投稿主控计划.md`.
