# Changelog

## 0.1.3 — unreleased (release candidate)

The full list of changes since 0.1.0 is in the v0.1.3 release notes. Recorded here:

- On the official demo's raw data with a first-ICU-stay cohort, the next run after you
  confirm the demo source again no longer stops at once because "the first-stay
  restriction cannot be verified" (B14). The run's own working copy of the study now
  asks the registered official export through the study it was launched from.
- When a run adds its launch timing and re-signs the confirmed demo choice, the data
  the run prepares now carries the re-signed choice. Before, it kept the earlier one,
  so the run's source checks could refuse a valid study.
- When a study planned from the database dictionary uses a reviewed candidate whose
  exposure is summarized per stay (for example its maximum), preparing the data no
  longer stops because that summarized exposure "is not in the catalog" (B15). The
  summarized column is a coordinate the study carries; its base concept is still
  checked against the catalog.
- On the official demo's raw data, starting a run no longer undoes a demo choice you
  just confirmed (B16). The default for repeated ICU stays is no longer written into a
  study whose signed choice it would void, so the run does not ask you to confirm the
  demo source again.
- Behavior change since 0.1.2: the BibTeX file (`manuscript_scaffold.bib`) is a plan
  deliverable ("references"). A run writes it when the plan's confirmed deliverables
  include references, for example when the question asks for a bibliography. Otherwise
  the LaTeX manuscript keeps an inline bibliography and no separate `.bib` is written.
  This came in with 8165c4913 ("Honor reviewed article deliverables and explain tier
  omissions"); a `.bib` by default when the manuscript has citations is planned for
  0.1.4.
- A plan no longer stops with no way forward when a study's cohort note (for example
  "first-stay handling remains for the formal plan") was read as a first-stay rule of
  the other scope than the study's setting (B17). When the words do not themselves name
  that scope (a note, or a plain "first ICU stay"), the plan goes back to the Planner,
  which omits the note or follows the study's own scope. Words that do name the other
  scope ("first ICU stay per hospitalization") are never retyped: the plan card states
  the conflict and points to "Request changes".
- A population condition counts as stated by the study only when its words are in the
  research question. Wording found only in the study's cohort notes (which can hold
  the assistant's own remarks) is shown as a system proposal, never as the study's
  own words.
- A plan review card that no automatic revision can resolve (review findings only an
  owner can close, or an automatic revision that has stopped) no longer shows no
  buttons at all. It says why in one sentence and offers "Request changes (back to
  the conversation)".
- On a raw database source with no extracted analysis package, a candidate plan no
  longer offers "Confirm plan and prepare data" (B18). Preparing from raw files read no
  data, so the run only planned again on metadata and the same card came back. The card
  now says the study's data must be extracted first and offers to ask for that
  extraction in the conversation; a direct submission of that preparation is refused
  with `research_pipeline_prepare_requires_extracted_package`.
  A data preparation that ends asking for preparation again (it ran on metadata only and
  built no cohort) never counts as done: the card states that the last preparation read
  no data, and offers extraction or, when the source now has a package, preparing again
  from it. B18 does not come from the B15 candidate change; it is the budget a raw
  source resolves to.

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

The repository history was rewritten to remove files that should not have been
published. Earlier 0.1.x pre-releases were withdrawn; this release replaces them.

### Earlier note

- Formal receipts now bind an optional dictionary overlay; main runs are unchanged.
