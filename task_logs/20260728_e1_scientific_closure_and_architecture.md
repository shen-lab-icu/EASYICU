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
