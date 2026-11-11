# Changelog

## 0.1.0 — 2026-11-11 — first public pre-release

### Highlights

- Standardized concept extraction across the supported public ICU databases
  (MIMIC-IV, MIMIC-III, eICU-CRD, AmsterdamUMCdb, HiRID, SICdb), plus mapping-only
  support for four community sources (NWICU, Zhejiang eICU, Jinhua, Zigong).
- Local web app and desktop packages with a guided Copilot conversation for
  study setup, plan review and governed execution.
- Evidence-bound research agent (Plan, Build, Analyze, Gate) that drafts a
  manuscript only after the evidence gate passes, with number tracing,
  a scientific-review list and downloadable study replay.
- Analysis methods including cross-database external validation, REML pooling,
  g-methods, mixed models and GEE, and multiple imputation pooled by Rubin's rules.
- Model request records, a no-data problem report, and counts of 1 to 10
  withheld from what is sent to the writing model.

The full list of user-visible changes is in
`src/easyicu/data/release_notes.json`.

### Hardware

Default extraction, 4 workers, minimum 4 CPU cores.

- MIMIC-IV-scale databases: 16 GB RAM recommended, about 66–70 minutes on
  4 cores. Under a 14 GB cap the peak was 12.45 GiB and the run took 65.6
  minutes, completing after one automatic halve-on-out-of-memory retry.
- eICU needs a high-memory server: about 88 GiB peak with 4 workers and no
  cap; under a 32 GB cap it completes with 3 automatic retries.
- Peak memory with 4 workers and no cap: MIMIC-IV 19.4, AUMC 17.4, HiRID 17.6,
  eICU 87.7, SICdb 7.4, MIMIC-III 14.5 GiB.

See [docs/local_run_guide.md](docs/local_run_guide.md).

### Known limits

- Fitting extraction within 14 GB without retries is planned for 0.1.1.
- Manuscript number binding is heuristic. It may reject ambiguous but correct
  sentences, and it does not catch values swapped within multi-group list
  sentences.
- Small-cell suppression applies to what is sent to the model. The manuscript
  keeps exact counts behind the publication sign-off; automatic suppression of
  published text is planned for 0.1.1.
- Python 3.11 and 3.12 are tested; 3.13 is not yet.

### Repository history

Repository history was rewritten to remove database identifiers and data files
that had been committed by mistake.

### Earlier note

- Formal receipts now bind an optional dictionary overlay; main runs are unchanged.
