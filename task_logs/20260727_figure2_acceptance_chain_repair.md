# Figure 2 formal acceptance-chain repair

Date: 2026-07-27
Task: `FIG2-CANONICAL9-ACCEPTANCE-CLOSURE-20260727`
Status: offline code repair and focused verification complete; no Provider call
or Canonical9 experiment launched.

## Reproduced failures

- A replay-verified task with `analysis_only` could still contribute to an
  `accepted` nine-task terminal receipt.
- The paper-acceptance artifact could be written before batch-ledger failure
  was known, leaving a stale accepted file.
- Formal execution read `figure2_safety_receipt.json`, but no production path
  issued it after the Agent completed.
- The EHRFlow adapter loaded a full cohort solely to obtain row and column
  counts, then passed the path to the pipeline for a second read.

## Repair

- Paper acceptance now requires every task to be `gate_reportable`.
- Batch-ledger and frozen-input verification run before the one terminal
  acceptance artifact is written; any mismatch is encoded into that artifact.
- The evaluator owns a typed, evaluator-only safety transport, prompt,
  response parser, issuer, verifier, atomic receipt writer, and reuse path.
  The research Agent does not import the evaluator or see its rubric.
- Formal E1 issues a missing safety receipt and deterministically rescores
  before the canary decision. Development diagnostics never spend this extra
  evaluator call.
- Parquet shape comes from footer metadata. CSV/TSV shape uses a header read
  plus bounded single-column chunks. Full cohort loading remains owned by the
  pipeline.

## Launcher boundary

The project launcher now:

- passes `adaptive_v1` explicitly by default;
- requires either one non-paper development receipt or all four formal
  authority files, never a mixture;
- passes the formal paper-acceptance and authority arguments together;
- stamps a launcher fingerprint only when this invocation created or changed a
  run, so a pre-run failure cannot relabel an older run.

The launcher is project-level operational state outside the `EASYICU` Git
root, so it is recorded here but not part of this repository commit.

## Verification

- acceptance, safety issuer/runner, bench integration, and real-run authority:
  `177 passed`;
- Ruff on all touched benchmark source and tests: passed;
- launcher `bash -n`: passed;
- bounded Parquet and CSV shape tests prove the pre-pipeline adapter does not
  call `pandas.read_parquet` or materialize all delimited columns.

## Freeze boundary

The scorer-tree digest is intentionally not changed inside this implementation
commit. After the evaluator source is committed and its tests are stable,
refresh the v3 scorer digest once in a separate, reviewable freeze commit.
That freeze must happen before a formal paid run, not after it; changing the
scorer after observing experiment results would make the evaluation post hoc.
