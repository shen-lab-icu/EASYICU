# Research-agent run receipts

The runtime preserves a digest-bound snapshot whenever its workflow returns a
terminal `PipelineResult`. Receipt publication happens after the final durable
review-checkpoint transition. Human-review pauses do not create terminal
receipts. A missing terminal manifest, corrupt source, or publication conflict
fails closed with a named receipt error.

The default destination is outside the run scratch tree:

```text
<EASYICU_HOME or user home>/.easyicu/run_receipts/<run_id>/<receipt_sha256>.json
```

`EASYICU_RUN_RECEIPT_ROOT` can select a retained storage root. The shared
`easyicu.state_paths` owner defines `EASYICU_HOME` consistently for Web state,
extension registries and receipts. Receipt paths inside the run tree are
rejected to prevent recursive inventories. Identical publication is idempotent;
different bytes require a different path. This is application-enforced
immutability, not storage-level WORM protection.

The discovery launcher preserves another version after its final package
assessment, because those files are written after pipeline completion. Other
post-run artifact/signoff updates require another call to
`preserve_terminal_run_receipt` or the CLI. The automatic hook does not inventory
the surrounding Web wrapper or intercept arbitrary filesystem writes. Process
crashes before a terminal manifest require the owning runtime/formal harness's
failure receipt; a summary must never manufacture a terminal run.

## Retain or verify an explicit copy

```bash
python tools/preserve_run_receipt.py <run_dir> --out docs/evidence/run_receipts/<run_id>-<version>.json
python tools/preserve_run_receipt.py <run_dir> --verify docs/evidence/run_receipts/<run_id>-<version>.json
```

## Recorded facts and limits

| Field | Source |
|---|---|
| Run identity, question, timestamps, current plan authority | `manifest.json` |
| Status, gates, strict mode, code identity | `run_status.json` |
| Evidence authority head | `.easyicu_evidence_authority_head.json` |
| Step outcomes | `steps/*/outputs/step_summary.json`, with the legacy step-root fallback |
| Artifact paths, byte sizes and SHA256 | Regular files in the run tree |

The inventory is checked before and after reading the projected facts; observed
source drift is rejected. Verification checks the self digest, complete artifact
inventory and facts reconstructed from source. It rejects changed, missing,
unrecorded, duplicate and unsafe artifact paths. An `analysis_only` run remains
`analysis_only`. A digest is not a human signature or publication authorization.

A receipt retains metadata and commitments to artifact bytes. It does not
retain those bytes, demonstrate scientific correctness, authenticate a signer,
or make a deleted run reproducible. Without source, its internal digest can be
checked but its relationship to the original run cannot be independently
reverified. Preserve the source tree and the receipt in backed-up storage before
citing a run in a demo, README, benchmark table or paper.

## Disclosure

No data tables are copied, but questions, paths, gates, plan references and
authority-head metadata are copied from the run. Review those fields before
publicly committing or sharing a receipt; the receipt builder is not a privacy
redaction gate.

## Current committed records

None. The earlier retention audit reported the cited
`run_20260829T024326_bfcbf6` tree as unavailable. This engineering change cannot
recover its missing artifacts and has not fabricated a retrospective receipt.
Synthetic test receipts validate software behavior only.
