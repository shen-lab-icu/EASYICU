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
- A data preparation that stopped before it wrote a run (for example a brief model-service
  sign-in failure) no longer leaves the same "Confirm plan and prepare data" card with no
  failure line. The card says the last preparation did not finish, gives the reason, and
  offers "Prepare data again" (or extraction on a source with no package). A preparation
  that wrote a run keeps that run's own failure message.
- A first-stay cohort setting is never applied silently (S1639). When the study keeps
  each patient's first ICU stay (or the first stay within each hospitalization) and the
  plan's population conditions do not list it, the plan card names it as "Keep each
  patient's first ICU stay only · Study setting (system default)" with "Change in the
  conversation". The plan review now reads the same scope: on a metadata-only plan
  (for example the raw official demo) it no longer asks to retain all ICU stays while
  the extraction keeps one stay per patient.
- On a study planned from the official Demo's raw data, asking in the conversation for
  this study's extraction now starts it (B19). The card's drafted request confirms the
  extraction for the study's current cohort, so sending it grants the extraction; the
  study's extraction uses the Demo's full module scope when the study names no modules;
  preparing the same Demo again is redirected to the study's extraction instead of
  re-registering the generic Demo export; and a generic Demo export is never described
  as the study's own data package. When you send that request on the raw-data plan
  card, EasyICU starts the study's extraction itself, once, before the assistant
  replies, and the assistant reports the result. A request that does not confirm the
  extraction, or one sent in any other state, is left to the assistant as before.
- Extracting a MIMIC-IV or MIMIC-III study that keeps each patient's first ICU stay no
  longer fails with "criterion 'first_icu_stay' is unavailable" (B20). Extraction now
  selects the first ICU stay within the database's record (icustays ordered by intime
  within each patient, MIMIC has no outside or earlier stay history), the same host rule
  planning and the run apply, and the export's cohort report names that rule with
  aggregate counts. A tied or missing admission time, or a missing stay table, still
  stops the extraction with a typed reason. On MIMIC the plan card's first-stay row
  names the same scope.

- Preparing the data of a reviewed plan no longer fails with
  "caller_bound_population_mode_conflict" when the study's own extraction already kept
  only adults and the plan read age within the first hours of the ICU stay (B22). Age is
  one value per stay, so the extraction's age step proves the criterion and the prepared
  plan keeps every input row.
- Switching the model connection in setup (for example from a saved API connection to
  the ChatGPT account) while a conversation is open now starts a conversation on the
  chosen connection. Before, the open conversation stayed on the connection it was
  created with, so the next messages went to the model and account you had just left.
  The earlier conversation stays in the project.
- When the model, asked how the study groups its exposure, names something that is not
  a measured value (for example the grouping's own id) as the grouping's concept, the
  plan no longer stops at once telling you to change the grouping in your question
  (B24). The model is asked once more with the reason and the values the data hold. If
  it names no known value again, planning stops as a planning error, says it is not a
  problem with your question, and points to "Generate a fresh plan". A concept the data
  lack but Data Extraction defines still reports the extraction it needs. The message
  for a grouping that cannot be formed on the data no longer assumes the question is at
  fault.
- A plan no longer silently studies every input row when the question names a narrower
  population (for example "septic patients") and the plan neither reads nor selects on
  it. Such a plan is not approvable: it goes back to the model to state the population
  as a condition, and a condition that cannot be applied is shown on the plan card as
  not applied.
- When the question both groups stays by a measured value and names that value in its
  wording (for example "maximum lactate on the first ICU day above 2 mmol/L"), the plan
  no longer also turns the same words into an inclusion condition that cannot be
  applied and asks you to accept it as a limitation (B25). The model is asked again; if
  it repeats the error, planning stops as a planning error that points to "Generate a
  fresh plan".
- "Extend to a full article" on a first-result card now prepares this study's article
  extension for review, the same as the matching follow-up suggestion (B26). Before, it
  only drafted a request into the conversation that started nothing, and the reply said
  no article run had begun. The button appears only when the study's result can take an
  article extension.
- When EasyICU was restarted between preparing a study's data and approving its plan,
  the approved analysis no longer stops at its first modelled step because "the
  landmark spline authority has not sealed its Planner-selected adjustment roster"
  (B27). The restored review now seals the plan's adjustment variables again before the
  analysis runs, as an uninterrupted run does.
- On a first-result card, a deliverable held back for an article no longer says it
  "requires a separate article run", which nothing on the card could start. It says that
  "Extend to a full article" appears on the card when the study's design supports it, and
  that landmark and other current-case designs cannot yet be extended to an article in the
  desktop app (planned for 0.1.4).
- A question that restricts its patients to a condition EasyICU records (for example
  「在脓毒症 ICU 患者中…」, "among patients with sepsis", "septic patients", AKI or
  mechanical ventilation) now names that population to the plan. The plan's requirement
  list must account for it, and a plan that neither reads nor selects on it goes back to
  the model to state the population as a condition. Before, the condition was not read
  as something the plan must answer, so the plan could study every input row without
  saying so.
- When the plan review asks which of the question's exposures to analyse and you choose
  one arrangement (for example "Analyse only the primary exposure"), the revised plan no
  longer comes back with the same question (B29). EasyICU records your choice for those
  exposures, carries it into the revised plan and its later revisions, and judges the plan
  on it: it is covered once the plan analyses the exposures you chose. A plan that names
  other exposures is asked again.
- "Generate candidate plan" on a landmark association study whose exposure the researcher
  verified in the conversation no longer starts nothing with
  "study_concept_selection_authority_server_owned" (B30). The launch staged the host's
  24-hour landmark together with the researcher's verified exposure receipt it had just
  read, and then refused that unchanged receipt. An unchanged receipt is now accepted
  on that path as it is everywhere else; a receipt a client creates or changes is still
  refused.
- When the question states how its exposure is summarized (for example 「暴露只取入科时的首次
  乳酸，不使用 24 小时峰值」 or "the first lactate at ICU admission"), the study can no
  longer store a different summary such as the 24-hour maximum (B31). Such a write is
  refused with both readings, so the assistant must correct it instead of only saying it
  did; a summary that a new question supersedes is dropped, and the question's own words
  decide. A population condition (「Sepsis-3 脓毒症成人 ICU 患者中」) is no longer read as the
  study's exposure.
- A question that restricts its patients to a population EasyICU has no condition for
  (for example 「在心衰患者中」, "among patients with heart failure", 「ARDS 患者的」,
  "trauma patients") is no longer silent when the plan does not list it: the plan card
  says 「问题限定了人群“心衰”，方案未包含对应的人群条件，请确认」 and offers to add it in the
  conversation. Sepsis, AKI and mechanical ventilation are the conditions EasyICU selects
  on itself.

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
