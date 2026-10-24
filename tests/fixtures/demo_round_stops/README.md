These fixtures are reduced from the public-demo rounds recorded on 2026-10-23
with images dev9-c77e87260 and dev9-e81c40797. They contain no patient rows, credentials, model
prompts, or raw model outputs.

- `dh1_prompt_metadata.json` and `dh3_prompt_metadata.json` retain the primary
  exposure, outcome, two outcome companions, and their trajectory metadata
  from each saved `research_context.json`. Variables and bindings were scoped
  together using `project_research_context_variables`. Observation summaries
  were removed, counts set to zero, and the question, cohort identity and
  local input path replaced. Digests and physical metadata remain to exercise
  typed validation; tests never resolve their referenced files.
- `da6_cohort.json` retains the locked predicate, grouped variable descriptor,
  and the Table 1 contract that requires the declared levels. The diagnostic
  exception type and message digest identify the original failure. Tests
  apply counterfactual numeric predicates only to four synthetic values.
- `da6_stop.json` retains the browser-safe child-job projection and the saved
  wrapper's status/reason. The job has neither an output run ID nor artifact
  references; the wrapper still has the `run_<job_id>` identity assigned by
  `RunDirectory.create`. The UI regression uses those identities and progress
  receipts to reproduce the two visible stop sentences.
- `dd3_descriptive_summary.json` retains the failed primary step's typed
  declarations, its two variable descriptors (including closed category
  labels), and the saved executor stop reason. The intent is synthetic;
  observed outcome values and all population counts are omitted. Tests use
  five synthetic fractional-day values to reproduce the same exact-value
  validation failure after CSV readback, for grouped and ungrouped summaries.
  dD3 passed measurement auditing and has a separate serialization defect
  from dH1/dH3's bounded prompt projection.
