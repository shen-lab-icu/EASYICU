# E1 scientific closure and architecture handoff — 2026-07-28

## Status

- Branch: `fix/external-review-20260724-p0-p1`
- Code endpoint: `0bf6842`
- Old E1 with 13 resumes remains `diagnostic_only`; it must not be resumed or promoted.
- Figure 2 paper-facing score remains 0/9. The frozen scorer digest is intentionally not refreshed during development.
- The code repairs themselves were verified without Provider calls. One fresh
  development E1 canary at `b94ffab` used the local Provider and terminated
  fail closed; no Provider call was made after its three failures were
  localized.

## Closed findings

The repair series converts the latest E1 audit findings into owner-module contracts:

1. Event missingness distinguishes measurement absence, event absence and conditional non-applicability.
2. Stay counts are no longer labelled as patient counts when subject identity is unavailable.
3. SOFA first-time derivatives carry time units; representation rules no longer contradict generated means.
4. Statsmodels Wald intervals and underflowed P values are labelled accurately.
5. Critic review receives a bounded structured scientific summary instead of relying on file presence.
6. Provider costs aggregate across resumes and separate reported use from unknown/reserved upper bounds.
7. Durable heartbeats expose phase and step progress.
8. Finalized attempt history is externalized and the manifest points to bounded history records.
9. Evidence aliases and manifests bind the current plan authority.
10. E1 protocol and acceptance enforce Table 1 SMD, typed event semantics, typed cohort consumption, invalid event-time review, landmark handling, readmission sensitivity and nonlinear sensitivity.
11. Planner-owned exposure labels replace generic `Category 0/1` output.
12. Step 06 deterministic robustness execution and its plausibility receipt share one immutable obligation scope and can pass without a Provider call.

## Architecture boundary closure

Commit `d7fe639` moved policy decisions back to owner modules:

- Critic semantics: `research_agent/review/step_semantics.py`
- Plan manifest projection: `research_agent/authority/plan_input_closure.py`
- Runner timeout: `StepExecutor.runner_timeout`
- Repair preflight composition: `research_agent/repairs/preflight.py`
- Cohort display semantics: `research_context/cohort_granularity.py`

No architecture baseline was refreshed. The existing architecture guard stayed green, as did module graph, seven Import Linter contracts, Deptry, Ruff and diff-check.

## Verification

- 158 targeted research-agent tests passed.
- 22 focused owner-contract tests passed.
- 4 exact resume regressions passed.
- A broader resume/provider batch reached 89 passed before being stopped because it had expanded into repetitive coverage; no failure occurred before the stop.
- Earlier focused validation included 201 Step 06 tests, 148 formal safety/acceptance/plausibility tests, 263 Table 1 tests and 35 E1 acceptance/figure tests.

## Fresh5 diagnostic and local closure

The exact-SHA image `easyicu-research-agent:dev-b94ffab-20260728`
(`sha256:3ed84abb0a2c...`) ran a fresh aware-only `adaptive_v1` E1 at:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260728_luna_miiv_dev_b94ffab_e1_fresh5/e1_sepsis3_prevalence_mortality/aware/run_20260728T102246_0f1738`

It ended `diagnostic_only` with 3 of 11 required steps complete. The run made
23 completed Provider calls and correctly refused to promote incomplete
evidence. It also proved that the former 65 KB local repair-envelope limit was
removed: an approximately 82 KB full-rewrite request reached the transport.

Three owner-attributable failures remained:

1. The plausibility gate did not follow a named literal column list into a
   generic validation helper. `0f90509` now resolves that one assignment edge;
   60 focused tests pass, an omitted column remains blocked, and the exact
   quarantined Step 04 script now returns no finding.
2. Generated Step 05 code sent the `StrictNumericInput` envelope into pandas
   instead of projecting `.values`. `0bf6842` adds a narrow traceback- and
   AST-bound repair owner. The exact archived script then ran over 94,458 stays
   without a Provider call and wrote primary, landmark, non-readmission and
   flexible-form estimates. The repair neighborhood is 126 passed; 13
   architecture tests also pass.
3. Step 02 figure code selected rows only by non-null exposure/outcome labels,
   so two zero-count `missingness` rows were treated as joint cells. This item
   is intentionally left open for the next handoff; no speculative patch was
   committed.

No fresh6 image or run was started. The next agent must close item 3, replay
all three exact artifacts without Provider calls, then create a new SHA/image
and a brand-new development batch.

## Fresh E1 policy

1. Use a clean detached worktree at `d7fe639`.
2. Materialize E1 into a new external directory; do not mutate or reuse the earlier `a3d8508` materialization.
3. Run zero-Provider materialization, planner and runtime preflight first.
4. Build an exact-SHA Docker image and require source identity match plus `network=none`.
5. Run only the aware arm with `adaptive_v1` against the local loopback Provider.
6. Treat the first fresh run as a development canary. Do not refresh the formal Figure 2 freeze or start E2–E9 unless E1 passes the complete execution and scientific closure.
7. On a framework failure, stop, assign it to one owner module, repair with focused tests, create a new SHA/image and start a fresh run. Do not spend repair calls repeatedly inside a contaminated run.

## Intentionally deferred

- Exact actual cost for the interrupted historical call cannot be reconstructed; reports must preserve it as unknown and show a conservative upper bound separately.
- Historical duplicate artifacts are not deleted. Storage deduplication is a separate migration and must not mutate evidence-of-record.
- Submission-grade modelling choices and figure polish are validated through the new E1 sensitivity/acceptance contracts; the old E1 figures remain diagnostic.

---

# Round J — Codex corrections to the owner-coverage tool, Step 02 receipt, figure capability

Zero Provider calls. No image rebuilt, no run started, nothing pushed.

## J1–J3. The pre-run coverage replay was over-stating three ways

Codex ran `tools/owner_coverage_replay.py` and got a stderr note followed by a
precise coverage number. Both defects reproduced locally.

**Scoring a plan it had modified.** On a validation failure the tool deleted
`robustness_specs` and scanned the remainder. Reproduced with a plan whose only
defect was an unresolvable robustness spec: the tool printed the note to stderr
and `05_… -> owned` to stdout. A caveat on one stream and a step-precise answer
on the other is not a warning. `load_plan` now raises `PlanNotScannable`
(`invalid_plan`); `main` returns 2 and prints no rows and no tally — asserted on
both streams.

**Reporting `coder` for what it could not decide.** Executors whose readable
schema is fixed by the producing step need the host's `resolved_bindings`.
Offline they decline by construction. New verdict `unknown_runtime_binding`,
derived structurally: a step whose declared inputs include a typed product
another step in the same plan promises, and for which the snapshot carries no
bindings. `artifact:` inputs are excluded — the locked cohort's schema is host
knowledge. On the real E1 plan this moves 3 of 10 steps out of a verdict the
tool could not support.

**Typed `SelectionContextSnapshot`** carries plan, per-step parent bindings and
real receipt obligations. Obligations are compiled by
`compile_flag_only_plausibility_scope` — the production compiler — never
re-derived. Supplying them turns `conditional_receipt` into a definite answer in
either direction (both asserted).

**Coverage is not a gate.** `--require-deterministic <step_id>` is repeatable and
makes it one (exit 1 on shortfall; exit 2 if the protocol names a step the plan
does not contain). Without it the report says it is advisory: an open-ended
scientific step may legitimately go to the Coder. 20 tests.

## J4. Step 02 — the decline was about attributability, not capability

`descriptive_cohort_summary` can compute the flag-only receipt. It declined
because `flag_only_plausibility_obligation_findings` proves the obligation from
the source that will run, and this executor's entrypoint is a single import
call ⇒ `plausibility_check_not_attributable`. Verified against the real gate.

The receipt comparisons are now rendered into the entrypoint over the same frame
the summary is built from (`load_cohort_summary_frame` exported so the cohort is
read once), and the entrypoint performs the `step_summary.json` write itself —
the gate checks the destination as directory **and** filename **and** key, so a
write inside an imported helper is not attributable to it. Correctness stays
host-side: `_verified_plausibility_audit` raises before any summary exists.
Both the obligation gate and the post-execution receipt gate return 0 findings.

### Receipt had no denominator

An entirely missing column and a fully observed, entirely in-range column emit
byte-identical receipts (`below=0, above=0, out_of_range_n=0`); the gate only
checks `total == below + above`. `plausibility_obligation`'s own docstring
already states that "no out-of-range rows" and "we never looked" are different
facts — the receipt lost that distinction one level down. All four host
renderers now emit `compared_n` and `observed_n`.

**Not done, deliberately:** the gate does not yet *require* `compared_n`.
Coder-written receipts predate the field, and making it mandatory would
introduce a fail-close that was not requested. Recorded as an open decision.

14 tests, including a partly-recorded column (`compared_n == 2` of 4 rows, cohort
still 4) and an entirely missing one (`compared_n == 0`).

## J5. Figure ownership by required/optional typed input capability

All four figure executors decided ownership with `tuple(step.inputs) == <its
constant>` — order-sensitive, while every renderer looks bindings up by key and
compares the manifest as a set. New owner
`execution/runners/figure_input_capability.py`: `TypedInputCapability(required,
optional)` with `admits`/`admits_step`. Order-independent; still refuses an
unknown input (a step naming an extra table is asking for a figure that reads
it), a missing required input, duplicates, and any contract/declaration
mismatch.

**Every renderer's `optional` is empty today**, asserted by a test. Each indexes
every binding it declares; declaring one optional would turn a clean decline
into a sandbox crash. So this does not by itself make E1's
`04_missingness_and_measurement_audit_figure` ownable — that plan declares one
audit table and the renderer needs two. That is the plan not promising the
second table, which the E1 protocol's three-product requirement addresses;
the renderer must not cover for it.

**Genuine remaining gap:** `05_primary_adjusted_association_figure` (forest) and
`06_robustness_sensitivity_figure` have no renderer at all. That is missing
capability, not a matching rule.

19 tests.

## Attribution of test failures

`tests/research_agent` full sweep: 32 failed / 7686 passed (52 min); the prior
recorded baseline was 74 failed. Every failure visible in the captured tail is
in the recorded baseline list.

For the focused suites, a detached worktree at `8e038d7` (never `git stash`) was
used as the baseline. Before fixing: 4 newly failing, all mine (3 receipt-shape
assertions from `compared_n`, 1 obsolete abstention test). The 9
`test_flag_only_plausibility_repair` failures (`ImportError:
_records_out_of_range_evidence`) are present at baseline. After fixing, the
focused suites are green.

Two tests changed meaning rather than being deleted, because their principle
still holds and only their instance went stale:

- `test_standard_executor_abstains_when_it_cannot_emit_required_receipt` and
  `test_the_report_cannot_claim_an_owner_the_selector_declined` moved from the
  cohort summary (now selected) to a figure step, which still genuinely cannot
  emit a receipt.
- `test_owner_rejects_reordered_or_widened_contract` split: reordering is now
  asserted *accepted*, widening/narrowing still refused.

`tests/test_owner_coverage_replay.py`'s receipt-gated fixture moved for the same
reason, and gained a test asserting the E1 Step 02 shape is now owned.

---

## Round K — Codex's three corrections, all of them real

Two of them overturned conclusions I had already reported.

### K0. The 12-step plan was never missing

I said I could not locate it. It is at
`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260728_luna_miiv_dev_8e038d7_e1_fresh6b/e1_sepsis3_prevalence_mortality/aware/run_20260728T131514_c54ead/`.
Both `analysis_plan.json` (22,602 B) and the executed
`analysis_plan_revision_3.json` (25,590 B) hold **12 steps**. The 10-step plan I
scanned belongs to a different run. Every conclusion I drew from it about E1 has
to be re-derived, and one of them was wrong (K3).

### K1. `invalid_plan` was a false statement about a run that happened

Reproduced verbatim before changing anything:

```
not scannable [invalid_plan]: analysis_plan.json does not validate as an AnalysisPlan.
  No coverage is reported: a plan the pipeline would reject is
  not a plan whose ownership means anything.
robustness_specs.1.cohort_override.inclusion.0
  Value error, unknown concept_id: icu_readmission
```

`icu_readmission` is not a packaged dictionary concept. It is a column of the
materialised cohort, and `tools/run_research_agent_bench.py:1366` calls
`register_cohort_concept_ids(cohort_columns)` **before** the pipeline plans, for
exactly this reason. The run's own authority
(`cohort_authority.sha256-5bb2a148…json`, reached through the capsule's
`materialized_cohort_authority_ref`) lists **104 columns** and contains it. The
pipeline accepted this plan and ran it.

So the tool answered a question it did not hold — in the strict direction this
time. Same defect as J1's optimism, mirrored. Now:

* `missing_validation_context`, and the message never says "would reject".
* The classification is **proved, not pattern-matched**: the unknown ids are
  extracted, registered, and the plan re-validated. Only if that succeeds is the
  failure attributed to the missing registry. A plan broken any other way stays
  `invalid_plan` (test).
* Registration goes through a new `cohort_concept_id_scope` context manager in
  the owner module (`planning/cohort_contract.py`), which restores the exact
  prior set. `clear_cohort_concept_ids` empties it wholesale, so asking a
  hypothetical previously meant destroying someone else's registration.

### K2. Restoring the real context, bound by digest

`--run-dir` loads:

| fact | source | binding |
|---|---|---|
| cohort column registry | `cohort_authority.sha256-*.json` | capsule `materialized_cohort_authority_ref` **file + sha256**, re-hashed and compared; authority's `cohort_sha256` must equal the capsule's |
| typed product bindings | `resolved_inputs/<step_id>.json` → `inputs` | complete by construction: `_write_resolved_inputs_manifest` raises unless the binding keys equal the declared typed inputs |
| receipt obligations | same file → `raw_input_contracts` | compiled by the **production** `compile_flag_only_plausibility_scope`; a step with no recorded contracts is omitted, not given an empty obligation |

Refused: a tampered authority, an authority describing a different cohort, a
capsule with no reference. Picking whichever `cohort_authority.*.json` happens
to be in the directory would let the tool validate a plan against a cohort the
run never used.

### K3. "The missingness figure only declares one table" was wrong

Step 7 of the real plan declares **both**:

```
07. 05_missingness_measurement_process_audit_figure
    inputs=['table:missingness_measurement_audit', 'table:measurement_process_audit']
```

That claim came from the 10-step plan. Withdrawn.

### The real 12-step owner matrix

```
 1. 01_define_analysis_cohort                        -- coder --
 2. 02_cohort_definition_summary                     owned    descriptive_cohort_summary
 3. 03_table_one_by_sepsis3                          owned    grouped_table_one
 4. 04_prevalence_mortality_distribution             -- coder --
 5. 04_prevalence_mortality_distribution_figure      -- coder --
 6. 05_missingness_measurement_process_audit         owned    declared_missingness_audit_products
 7. 05_missingness_measurement_process_audit_figure  UNKNOWN  (parent binding)
 8. 06_primary_adjusted_association                  -- coder --
 9. 06_primary_adjusted_association_figure           UNKNOWN  (parent binding)
10. 07_e1_scientific_sensitivity_table               -- coder --
11. 08_robustness_sensitivity                        -- coder --
12. 08_robustness_sensitivity_figure                 UNKNOWN  (parent binding)

3 owned / 3 unknown_runtime_binding / 6 coder
```

Codex's replay gave 3 / 4 / 5. The one difference is step 5: it *has* a recorded
binding, so the decline is a real verdict rather than an unknown. Reading the
selector's own trace, it is a **capability gap, not a matching rule**:
`exposure_outcome_distribution_figure` requires
`table:cohort_summary` **and** `table:exposure_outcome_distribution` (it needs
the locked denominator); the plan's figure step declares only the latter. Not
fixed here — whether the protocol should require both inputs or the renderer
should gain a single-table variant is a design decision, not a patch.

### K4. The regression was a false green

I reported "exit code 0" for a run that was `15 failed / 2051 passed`. Cause:
`pytest ... | tail` returns **`tail`'s** status. Two `pgrep -f "pytest
tests/research_agent"` watchers also matched their own command lines and would
have waited forever; killed (pids 17078, 18484).

Replaced with `scratchpad/regress.sh`: each leg writes pytest's **own** exit
code to `<leg>.rc` and its complete FAILED set to `<leg>.failed`, and the two
legs are the same selection against a clean baseline worktree
(`/private/tmp/easyicu-k-base` @ `0df6d26`) and the working tree.

Separately, the 15 failures were checked and none are mine:
`test_flag_only_plausibility_repair.py` (9) fails on
`ImportError: cannot import name '_records_out_of_range_evidence'` — the symbol
is absent from `audits/validators.py` at `8e038d7`, i.e. **committed-RED before
this session**; the trajectory-contract failures are bench exit-5 behaviour. My
five commits touch neither file.

### Overfitting check (Codex's separate note)

`git diff 8e038d7..HEAD` over `src/` and `tools/`: the case tokens
(`sepsis|sep3|sofa2|miiv|mimic|e1_`) appear in three files, and **my diffs added
one occurrence** — a docstring line in `planning/method_literature.py` naming
what the module does *not* supply. No case-specific branches were introduced.
That does not rebut the broader point: every shape being fixed still comes from
E1, so E1 is a development sample now, not a held-out test.

### K5. "All digest-bound" was not yet true

Only the cohort authority was verified. Bindings were found by listing
`resolved_inputs/*.json`, and the executed plan was a stderr note comparing
**file names**. Both now go through the manifest:

* **Plan authority.** `manifest.current_plan_authority` names the revision;
  its bytes are re-hashed against the declared sha256, and the EvidenceStore
  record for the same `evidence_id` must repeat both path and digest (exactly
  one record; zero or many is a refusal). Scanning any other file with that
  context is `plan_not_authority` — a **digest** check, which matters because
  the run root's `analysis_plan.json` and the executed
  `analysis_plan_revision_3.json` really are different bytes (22,602 vs 25,590).
  With `--run-dir` the plan argument is now optional: the run knows which
  revision it ran.
* **Bindings.** Each comes from `per_step_records[].resolved_inputs_path` at
  `resolved_inputs_sha256`. Fail closed on: digest mismatch, a path without a
  digest (half a receipt), a duplicate `step_id`, a capsule whose own `step_id`
  disagrees with the record filing it, a path escaping the run directory, and a
  capsule in `resolved_inputs/` that **no manifest record claims** (a stale
  attempt has no authority). A step with neither path nor digest is absent, not
  empty — `05_..._figure` in the real run is exactly that.

11 new fail-closed tests; the real matrix is unchanged at 3 / 3 / 6.

### K6. The scoped registry was not concurrency-safe

Snapshot-and-restore on a process-global set: A snapshots, B snapshots (now
including A's ids), A restores (dropping its own), B restores — and A's
hypothetical is in the process permanently. Now guarded by a re-entrant lock, so
one scoped question runs at a time; nesting in a thread still works.

**Mutation-verified.** With the scope's lock replaced by `if True:`, the
interleaved-threads test fails with a thread losing its *own* id inside its own
scope — real corruption, not a hypothetical. Restored, both tests green.

The lock is the honest fix for shared mutable state, not the good one. The
better long-term shape is an explicit immutable registry passed down; the
docstring says so rather than implying the global is now safe to use in
parallel.

### K7. Labelled as replay, not preflight

The tool is a **post-run replay**. Its report and module docstring now say so,
and name what a real preflight would need: prospective bindings compiled from
the producing step's Planner-declared typed product contract, and no `unknown`
left at the end. Not built here.

### K8. The read side of the registry, and a test that proved nothing

Codex: `concept_id_exists()` read the global set unguarded, so a thread that
never enters a scope could still observe another thread's temporary ids — a
hypothetical asked in one place becoming a real answer in another. The read now
takes the same `RLock`.

**The first test for it was worthless.** Two threads racing, one scoping and one
polling: it passed *with the read lock removed*. The scope enters and exits in
microseconds, so the bare reader never sampled inside it and the assertion never
had anything to assert. Rewritten to hold the window open deliberately — the
scope signals, sleeps a fixed interval, exits; the reader is released into that
interval. Mutation now behaves correctly:

```
read lock removed  -> FAILED: a thread outside the scope observed a temporary
                      concept id: ['scoped_only']
read lock restored -> 56 passed
```

The scoped thread never waits on the reader, because with the lock working the
reader is blocked on *it* — waiting would deadlock instead of failing.

**Recorded as debt, not fixed:** the lock closes the scoped-replay race only.
`register_cohort_concept_ids` is a permanent process-wide registration, so two
real runs in one process still accumulate each other's cohort columns and each
would validate a plan naming a column only the other materialised. There is one
set to mix into; a lock cannot unmix it. The fix is an explicit immutable
registry threaded through planning validation. Written into the module beside
the global.

### K9. Case detail removed from a shared gate module

`audits/step_summary_integrity.py` carried "the 2026-07-28 E1 run ... 94,458
rows" in a docstring. Nothing read it, but a shared gate should not narrate one
benchmark item. Rewritten to the shape of the defect with no case identity and
no counts.

A sweep found five other `E1` references in `src/`. Three are **provenance for a
measured constant** (`providers/prompt_budget.py`'s bytes/token calibration,
`orchestration/config.py`'s replan default) — naming the run a number was
measured on is traceability and stays. Two are one-line incident pointers with
no data (`cohort/repair.py`, `missingness_measurement_figure_executor.py`); left
as-is rather than sweeping other people's comments in an unrelated change.

Verified: **no E1 result value appears anywhere in `src/`.**
`33,997` / `OR 1.608` / `94,458` / the per-group counts return nothing.

## K10 — regression comparison, both legs, real exit codes

Never judged by a pipeline's status. Each leg wrote pytest's own `$?` and its
complete `FAILED` set.

| leg | pytest exit | failed | passed | skipped | wall |
|---|---|---|---|---|---|
| baseline `0df6d26` (clean worktree) | 1 | 33 | 7718 | 17 | 49:32 |
| changed (working tree) | 1 | 33 | 7754 | 16 | 43:25 |

`comm -13` and `comm -23` are **both empty**: zero new failures, zero fixed. The
33 are pre-existing on the clean tree. The background harness reported the
*script's* exit 0; the legs' own codes are both 1.

`+36 passed` = 26 mine (21 replay + 5 concurrency) + 7 from two **gitignored**
test files (`.gitignore:134-135`, `test_case_b_bootstrap.py`,
`test_pilot_exit_status_capture.py`) that exist only on this machine and run in
no clean checkout or CI + 3 that skip in a bare worktree. `-1 skipped` is two
dirty-tree guards in that same gitignored file, minus one. Every extra test
passed, so the asymmetry does not touch the verdict.

Baseline cached at `~/.cache/easyicu-regress/baseline-0df6d26.failed` with a
`.meta` recording the sha, flags, counts and invalidation rule. Later small
patches run the changed leg only.

## K11 — typed-contract gap map (read-only, zero Provider)

Authority: `analysis_plan_revision_3__analysis_plan_revision_3.json`, reached
through `manifest.current_plan_authority` + the matching EvidenceStore record,
both digests re-hashed. The run reached **7 of 12 steps**, stopping after
`06_primary_adjusted_association`.

Current replay: **3 owned / 3 unknown / 6 coder.**

| # | step | typed contract for the science | pre-run schema | today | target |
|---|---|---|---|---|---|
| 1 | `01_define_analysis_cohort` | full `CohortSpec` | yes | coder | `deterministic_owned` |
| 2 | `02_cohort_definition_summary` | inputs + host receipt | yes | owned | owned |
| 3 | `03_table_one_by_sepsis3` | `easyicu.table_one/1` — the reference standard | yes | owned | owned |
| 4 | `04_prevalence_mortality_distribution` | **none** — in `inputs` order + intent prose | no | coder | **`unsupported_contract`** |
| 5 | `04_..._figure` | consumption typed; parent untyped | no | coder | **`unsupported_contract`** |
| 6 | `05_missingness_measurement_process_audit` | declared inputs + outputs | yes | owned | owned |
| 7 | `05_..._figure` | both inputs from the owned #6 at a fixed schema | yes | unknown | `deterministic_owned` — **string only** |
| 8 | `06_primary_adjusted_association` | partial: no covariates, form, reference level, CI method, clustering, missingness | partial | coder | **`unsupported_contract`** |
| 9 | `06_..._figure` | no forest renderer exists | no | unknown | **`unsupported_contract`** |
| 10 | `07_e1_scientific_sensitivity_table` | **prose only**; `flexible_age_charlson` has no typed home | no | coder | **`unsupported_contract`** |
| 11 | `08_robustness_sensitivity` | 2 cohort axes typed; refit model + output columns not | partial | coder | **`unsupported_contract`** |
| 12 | `08_..._figure` | no robustness renderer exists | no | unknown | **`unsupported_contract`** |

**Binding rule for the next phase.** Covariates, model form and sensitivity
design must never be inferred from intent prose, from input-column set
arithmetic (`inputs` minus exposure/outcome/`_measured`/`_n`), or from an
analysis id. Design the typed spec first, then implement the owner. An executor
that picks its own covariates has taken a scientific decision that belongs to
the Agent.

### The step-7 finding, proved not read

Every `owns_step` clause passes except the figure product **name**:

```
capability.admits_step       : True     <- inputs match the renderer exactly
role / method / no specs     : True
declared figure product      : missingness_event_timing
renderer product allowlist   : data_quality, missingness_measurement,
                               missingness_measurement_audit
product in allowlist         : False    <- the only failing clause
renamed to an allow-listed spelling -> owns_step = True
```

`figure_product` is used only as that guard and as a filename stem
(`{product}.png`, `.figure_contract.json`, `_source_data.csv`, `figure_id`,
`out_dir / figure_product`). It never selects a panel, variable or transform.

Two consequences. (a) The allowlist has no semantic content — selection must
bind on the typed input contract and output kind, which are already checked.
(b) It was **accidentally doing path sanitisation**: three fixed strings cannot
traverse a directory, a Planner-chosen name can. So the fix is a deletion **plus
an explicit safe-`figure:<id>` validator**, not a deletion.

### What the fix will and will not show on this run

| supplied | matrix | step 7 |
|---|---|---|
| as shipped | 3 owned / 0 cond / 3 unknown / 6 coder | `unknown_runtime_binding` |
| allowlist off | 3 owned / **1 cond** / 2 unknown / 6 coder | `conditional_receipt` |
| allowlist off + step 7's real obligation | **4 owned / 0 cond / 2 unknown / 6 coder** | `owned` |

The middle row exists because this run died at step 06 and never recorded
contracts for step 7, so the tool refuses to invent an obligation and probes
with a deliberately harsher non-empty one. That the real obligation is empty is
measured, not assumed: the figure step that *did* run compiled to
`expected_columns=()`, because a rendering step reads typed products, not raw
concept columns. **Replayed against this run the honest report is
`3 owned / 1 conditional_receipt / 2 unknown / 6 coder`**; it reads `4 owned`
once any run reaches step 7, or once bindings can be compiled prospectively.
This gap will not be closed by fabricating an obligation.

### Coverage inventory

Deterministic owners that exist: table one, cohort summary, missingness audit,
source availability audit, trajectory stability, 4 figure renderers. **None**
for cohort definition, association models, scientific sensitivity, robustness
sensitivity, forest plots, robustness plots. `deterministic_robustness.py` is
2,478 lines but its `__all__` is `[replay_locked_memberships,
robustness_sensitivity_preflight_code]` — it validates Coder output, it does not
own the step.

Of the 6 steps that fall to the Coder, **none is open-ended science**;
`allowed_coder` is the right verdict for zero of the 12. They reach the Coder
because nobody typed their contract.

## L0 — a figure claimed by its contract, not by its label (`b9b1427`)

Removed the semantic-free product-name allow-list from the
missingness/measurement renderer, selector **and** runtime.

Correction to K11: the allow-list was **not** what stopped a path traversal on
the selector side -- `_figure_product` already parsed through
`[a-z][a-z0-9_]*`, so `figure:../../etc/passwd`, `figure:a/b` and
`figure:.hidden` were already rejected. The claim holds only for the public
`run_...` entry point, whose sole guard the allow-list was, and which
interpolates the id into `out_dir / figure_product`. So the change is a
deletion plus a validator **in one place**, not two.

Mutation-verified: restoring the allow-list fails 5, dropping the runtime check
fails 6, widening the id rule fails 14; restored, 49 pass. The new
case-token guard caught a case name I had written into a comment of this very
change.

Replay moved to `3 owned / 1 conditional_receipt / 2 unknown / 6 coder`, as
predicted in K11.

## L1 — declare the distribution, then own it (`f3358bc`)

`ExposureOutcomeDistributionSpec`: exposure, closed `exposure_levels`, outcome,
the exact `outcome_positive_value`, `denominator_policy`, `interval_method`.
Refuses eight malformed shapes. Planner is taught to emit it and **refused** if
it declares the output without it.

Against the real plan: the distribution step is **not** owned as the Planner
wrote it, and **is** owned once the design is declared. Exposure/outcome for
that check were taken from the plan's own `model_requirements`, not guessed.

The product is self-contained -- per-row denominator, missing count, events,
rate, interval, plus an overall row -- which is what lets a figure's input
contract close before its parent runs.

Level matching is by the declared level's own type. A declared *number* matches
the same number whatever dtype the export produced; a declared *string*
compares as a string, so `0`/`1` never absorb a `yes`/`no` column. **An earlier
version of that test asserted the opposite; the test was wrong, not the code**
-- refusing a string-typed column would fail-close a correct study.

`typed_cohort_binding.py` extracted as the owner of "which bytes may this step
read" (contained path, no symlink segment, digest, and columns/row-count equal
to the `product_contract` -- joined, because a digest proves the file is
unchanged, not that it is the table promised). It was already copy-pasted into
`deterministic_robustness.py`; the summary executor now imports it and its 14
tests pass unchanged. **The third copy is still there** -- migrating it is its
own change.

Regression: 33 failed / 7791 passed, pytest's own exit 1, identical 33-item set
to the cached baseline. Zero new, zero fixed.

## L2 — a renderer that needs one table

`exposure_outcome_distribution_render.py` consumes exactly the L1 product and
nothing else: no covert second dependency on a cohort summary. It re-checks the
arithmetic before drawing it (levels partition the cohort, events sum, observed
plus missing equals the row count, the rate lies inside its own interval) --
mutation-verified by deleting the partition check.

The end-to-end test builds the table with the **real producer** and hands only
that to the renderer, rather than hand-writing a fixture that might not match
what the producer emits.

## Method notes from this session

* **`nohup … &` reports the shell's exit code, not pytest's.** The harness
  said "exit code 0" for a run that had not written its summary line yet, and a
  failure diff read from that file meant nothing. Same family as
  `pytest | tail`; third disguise this session. Every leg now writes its own
  `$?` and its complete `FAILED` set, diffed against
  `~/.cache/easyicu-regress/baseline-0df6d26.failed`.
* **A case-token guard test belongs in every new owner**, and it earns its
  place immediately -- it caught the author, not a hypothetical future one.

## M — correction round on L1/L2 (external review, 5 reproduced defects)

The reviewer reproduced five real defects in the code I had just landed. All
five were confirmed against the source before any edit. Recorded here because
four of them share one shape: **a check that reads as complete but is not**.

1. **An undeclared outcome value was silently a non-event.** `outcome_levels`
   did not exist, so a `2` in a column believed to be 0/1 was observed, was not
   the declared event, and was counted in the denominator as a non-event. The
   rate deflates, the table still balances against every total, and nothing
   downstream can detect it. This is the worst of the five: it produces a
   *wrong number*, not a crash.
2. **`True` matched a declared numeric level `1`,** because
   `isinstance(True, int)` is true in Python. A boolean column would answer a
   0/1 declaration and the study would report a different variable from the one
   it declared.
3. **The distribution owner still fell back to bare `COHORT_PARQUET`.**
   `_typed_cohort_input` returned `None` when a step declared no typed input,
   and `owns_step` tested `!= ""`, so `None` passed. A table counted from
   unverified bytes was being labelled deterministically owned.
4. **The renderer's "re-check the arithmetic" was hollow.** It asserted the
   rate fell inside its own interval -- which a wrong rate usually does. 30.0%
   altered to 31.0% passed every check it had.
5. **L2 copied the binding loader** it was supposed to share with L1.

Closed by one commit:

* `ExposureOutcomeDistributionSpec` → schema version `/2`: closed
  `outcome_levels`; `outcome_positive_value` must be a **typed** member of it
  (`1` and `"1"` are different declarations); explicit `level_match_policy`
  (`exact_typed` / `numeric_string_equivalent`); explicit
  `missing_outcome_policy` cross-validated against `denominator_policy`
  (complete-case and carry-the-missing are opposite denominators, so the
  contradictory pairs are refused at validation); Planner-owned
  `confidence_level`, from which the `z` multiplier is derived rather than
  written down.
* Level matching is a partition, not a lookup: a value matching **two**
  declared levels is as fatal as one matching none. The refusal reports the
  number of undeclared rows and the number of distinct undeclared values and
  **not the values** -- a mis-declared column could be a continuous
  measurement, and the count is what makes the failure actionable.
* `typed_cohort_binding.py` → `typed_input_binding.py`, now the only binding
  loader: manifest `step_id`, exclusivity, `declared_kind`, product identity,
  `identity_row` self-agreement, consumption contract, containment, digest
  **before and after** the read, and the product contract. Failures carry a
  stable `reason_code`. L2's copy is deleted.
* The product carries its own design on every row, so a consumer can check it
  without being told anything. `outcome_positive_index` rather than the value:
  a CSV cell cannot carry a typed scalar -- `1` and `"1"` are written
  identically and both read back as a number -- so the event is identified by
  position in the levels array, which does survive the round trip.
* The renderer re-derives: each percentage from the counts on its own row, each
  interval rebuilt by the declared method at the declared confidence level, the
  strata summed against every total, and the denominators checked against the
  declared policy.

### Mutation results (and one that came back honest)

Each new check was deleted and the corresponding test re-run. Three failed
immediately as intended. Two came back green and needed diagnosis rather than
acceptance:

* The confidence-interval mutation was **my own precedence bug** --
  `if False and A or B` still evaluates `B`. Rewritten as `if False:`, two
  tests failed. The check was live all along.
* The boolean guard was **genuinely duplicated**: one copy in `_matches_scalar`
  and one in `_number`, each individually removable with tests still green.
  That is how a check goes dead later -- one copy is deleted, nothing notices,
  and the second goes the same way. Consolidated onto `_number`, which is the
  semantic home ("a boolean is not a number"), with a comment saying why there
  is only one. Removing it now fails a test.

### Boundary left open, stated rather than papered over

`load_typed_cohort` requires a consumption contract only when the *step*
declared one, because production attaches it only in that case. Requiring it
unconditionally on the cohort path would fail-close plans I cannot show
production satisfies -- I could not find a real cohort capsule on this machine
to check. The figure path requires it unconditionally, because its owner
requires the step to declare one. Raising the cohort path is a separate change
that needs a real capsule as evidence.

`deterministic_robustness.py` still holds the third copy of
`_contained_regular_file`. Unchanged from L1; still its own migration.

### A collision I found while auditing my own diff

`exposure_outcome_distribution_figure_executor.py` (the **pre-existing**
two-table renderer) already claims the product name
`table:exposure_outcome_distribution`, and expects a completely different
schema for it: `row_type, exposure_variable, exposure_category,
outcome_variable, outcome_category, count, percentage_of_locked_cohort,
denominator_n`. L1 introduced a second meaning for that name.

Checked in both directions rather than assumed:

* the old two-table executor, handed my 22-column contract, returns
  `owns_step=False`;
* my renderer, handed the old 8-column contract, refuses with
  `contract_columns_mismatch`.

It is survivable because **both owners bind on the schema the host recorded,
not on the name** -- which is the same principle that motivated L0 from the
other side. Now pinned by a test, because a name collision resolved by
structure is precisely what starts mis-rendering the day someone relaxes a
check to "make the shapes compatible".

### Two more defects the correction's own tests found

Both are the same shape as the one Codex reported about missing outcomes: the
code refused correctly but **named the wrong cause**, which sends a reader to
the wrong place.

* A **missing exposure** was swept into the undeclared-level bucket, so a row
  with no exposure value at all was reported as carrying "a value that is not
  one of the declared exposure levels". Someone reading that would go hunting
  for a stray category code that does not exist. Missing exposure now refuses
  on its own terms.
* With an exclusive binding, a **missing** input reported as
  `binding_widened`, because the set-inequality check ran before the presence
  check. "The host gave this step something extra" and "the thing it needs
  isn't there" are opposite problems. Presence is now checked first, and the
  widened message names what was added.

The second was found only because the binding owner got its own test file
rather than being tested through its two consumers -- both of which happened
to exercise the widened case and neither the absent one. A shared owner needs
tests at the owner, not coverage inherited from whoever calls it.

### A correction to how tonight's regressions were reported

Earlier entries in this log say "33 failed, identical to the cached baseline,
zero new". That is true **of the research-agent leg**, which is the only leg
`baseline-0df6d26.failed` covers. The core leg (all of `tests/` minus
`research_agent`) had **no** cached baseline, and when first measured it showed
**136 failures** -- which looked like a large regression.

It is not. Running the same nine files in a clean detached worktree at
`acbba99` produces the **identical 136-item set**; both `comm` directions are
empty. They are overwhelmingly figure2 canonical9 scorer-tree / frozen-authority
digest staleness, which any core `src/` edit invalidates by design and which
must not be "fixed" by refreshing the freeze. Now cached as
`baseline-acbba99-core-subset.failed` with a `.meta` recording why it exists,
so the next patch can diff it in three minutes instead of rediscovering it.

The lesson is narrow and worth keeping: **a cached baseline is only a baseline
for the selection it recorded.** Saying "zero new failures" without naming the
selection invites exactly the reading it does not support.

### Re-measured after the correction, and what 4/12 does *not* mean

The zero-Provider replay on the real authority plan gives the **same** matrix
after the correction as before it -- 4 owned / 1 conditional / 2 unknown /
5 coder -- so tightening the distribution owner (it now requires a real typed
cohort binding) did not cost the figure claim.

Two things found by reading that run's own artifacts rather than reasoning
about them, both of which qualify the number:

**The Coder had already computed this product, under a third set of names.**
`04_prevalence_mortality_distribution` completed, and the table it wrote for
`table:exposure_outcome_distribution` has columns
`classification, classification_level, n_stays, prevalence_denominator_n,
prevalence_pct, n_deaths, mortality_denominator_n, mortality_pct,
mortality_ci_low_pct, mortality_ci_high_pct`. That is the *same content* as the
typed product -- role, level, count, denominator, percentage, events, rate,
interval -- spelled differently. It is the strongest evidence yet for the whole
direction: the step was deterministically computable all along and went to an
LLM only because nobody had written the contract down. It is also now the
**third** schema living under that one product name (this one, the two-table
renderer's, and mine), which is exactly why every owner must bind on schema.

**The figure step produced nothing.** Its directory holds only a quarantined
`concept_draft.py` -- no outputs, no summary. So my renderer claiming it is not
converting a working Coder step into a failure; it is converting a step that
already failed into one that will fail *closed with a precise reason* until the
plan declares an `exposure_outcome_distribution_spec`.

That is the honest reading of the number: **4/12 owned is 4 steps whose
execution no longer depends on an LLM writing correct code, not 4 steps that
will now succeed on this plan.** On *this* plan the figure's parent is still a
Coder step, so the figure will refuse the parent's bytes. The ownership pays
off only once the plan declares the spec -- which is a Planner-instruction
change that has landed, and will show up on the next freshly planned run, not
on this frozen one.
